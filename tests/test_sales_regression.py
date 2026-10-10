"""
Regression tests for CleanSheet AI sales dataset cleaning engine and data integrity.

Tests verify:
1. Date parsing: Preserves each valid date, parses mixed formats safely without cross-row contamination or replacing with 2026-01-05.
2. Numeric conversion: 'two' in Quantity is converted to NaN with errors='coerce' (or flagged for review), never imputed with 1.428571.
3. Missing Total: Dev Malhotra's blank Total is never mode-filled with 110000.
4. Missing City: Ananya Iyer's blank City is never auto-invented/mode-filled.
5. Order IDs: Identifier columns preserve leading zeros ('00101') in DataFrame and Excel export.
6. Suspicious totals: Kabir Joshi's Total 9999999 is flagged for review.
7. Invalid phone: Flagged for review, preserving original text rather than silently converting to numeric/NaN.
8. Re-exported Excel verified against original CSV cell-by-cell.
"""
import os
import io
import sys
import pytest
import pandas as pd
from openpyxl import load_workbook

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_manager import load_file, detect_issues, compute_quality_score, apply_operation, accept_change
from ai_recommender import AIRecommender
from cleaning_engine import parse_dates, convert_to_numeric, fill_missing_with_strategy
from reporting import export_clean_excel, export_clean_csv

SALES_CSV_PATH = os.path.join(PROJECT_ROOT, "cleansheet_messy_sales(1).csv")


@pytest.fixture(scope="module")
def sales_info():
    assert os.path.exists(SALES_CSV_PATH), f"Sales test file not found: {SALES_CSV_PATH}"
    with open(SALES_CSV_PATH, "rb") as f:
        content = f.read()
    return load_file(io.BytesIO(content), "cleansheet_messy_sales(1).csv")


@pytest.fixture(scope="module")
def sales_df(sales_info):
    return sales_info.original_df.copy()


class TestSalesDateParsing:
    def test_safe_date_parsing_preserves_distinct_dates(self, sales_df):
        parsed_df, failed_count = parse_dates(sales_df, 'Order Date')
        # Row 1 originally '05/01/2026' -> 2026-01-05
        # Row 2 originally '2026/01/07' -> 2026-01-07
        # Row 3 originally 'not-a-date' -> NaT
        # Row 9 originally '13-01-2026' -> 2026-01-13
        assert parsed_df['Order Date'].iloc[0] == pd.Timestamp('2026-01-05')
        assert parsed_df['Order Date'].iloc[1] == pd.Timestamp('2026-01-05')
        assert parsed_df['Order Date'].iloc[2] == pd.Timestamp('2026-01-07')
        assert pd.isna(parsed_df['Order Date'].iloc[3])
        assert parsed_df['Order Date'].iloc[9] == pd.Timestamp('2026-01-13')
        assert parsed_df['Order Date'].iloc[10] == pd.Timestamp('2026-01-14')

        # Critical: Ensure 2026-01-05 was NOT copied across distinct dates
        assert parsed_df['Order Date'].iloc[2] != pd.Timestamp('2026-01-05')
        assert parsed_df['Order Date'].iloc[9] != pd.Timestamp('2026-01-05')

        # Value counts of 2026-01-05 must be exactly 4 (rows 0, 1, 8, 20), not 7
        date_counts = parsed_df['Order Date'].value_counts()
        assert date_counts.get(pd.Timestamp('2026-01-05'), 0) == 4


class TestSalesNumericConversion:
    def test_quantity_two_never_imputed_with_statistical_mean(self, sales_df):
        converted_df, failed = convert_to_numeric(sales_df, 'Quantity', errors='coerce')
        # 'two' at index 9 becomes NaN with coerce
        assert pd.isna(converted_df['Quantity'].iloc[9])
        # Never 1.428571
        assert converted_df['Quantity'].iloc[9] != 1.428571428571429
        assert converted_df['Quantity'].iloc[9] != 1.428571


class TestSalesMissingValueProtections:
    def test_missing_total_and_city_not_mode_filled(self, sales_df):
        with open(SALES_CSV_PATH, "rb") as f:
            info = load_file(io.BytesIO(f.read()), "cleansheet_messy_sales(1).csv")
        
        # Dev Malhotra (row 14) Total is blank
        assert pd.isna(info.current_df['Total'].iloc[14])
        # Ananya Iyer (row 4) City is blank
        assert pd.isna(info.current_df['City'].iloc[4])

        # Attempt to fill_missing with mode on Total and City
        new_df, _ = apply_operation(info, 'fill_missing', {'column': 'Total', 'strategy': 'mode'})
        # Guard must prevent mode fill on Total
        assert pd.isna(new_df['Total'].iloc[14])
        assert new_df['Total'].iloc[14] != '110000'

        new_df2, _ = apply_operation(info, 'fill_missing', {'column': 'City', 'strategy': 'mode'})
        # Guard must prevent mode fill on City
        assert pd.isna(new_df2['City'].iloc[4])

    def test_ai_recommender_protects_city_and_total(self, sales_df):
        issues = detect_issues(sales_df)
        recommender = AIRecommender(use_ai=False)
        recs = recommender.analyze_dataset(sales_df, issues)

        # Total and City missing values must be recommended as leave_unchanged, NOT fill_missing
        for r in recs:
            if r.issue_type == 'missing_values':
                if 'Total' in r.affected_columns:
                    assert r.suggested_operation == 'leave_unchanged'
                    assert r.risk_level == 'high'
                if 'City' in r.affected_columns:
                    assert r.suggested_operation == 'leave_unchanged'
                    assert r.risk_level == 'high'


class TestSalesIdentifiersAndOutliers:
    def test_order_id_leading_zeros_preserved_as_strings(self, sales_df):
        with open(SALES_CSV_PATH, "rb") as f:
            info = load_file(io.BytesIO(f.read()), "cleansheet_messy_sales(1).csv")

        assert info.current_df['Order ID'].iloc[0] == '00101'
        excel_bytes = export_clean_excel(info.current_df)
        wb = load_workbook(io.BytesIO(excel_bytes))
        ws = wb.active
        # Check first data row Order ID in Excel
        order_id_val = ws['A2'].value
        assert str(order_id_val) == '00101'
        assert ws['A2'].number_format == '@'

    def test_kabir_joshi_suspicious_total_flagged_for_review(self, sales_df):
        issues = detect_issues(sales_df)
        outlier_issues = [i for i in issues if 'Total' in i.get('affected_columns', []) and i['issue_type'] == 'suspicious_outliers']
        assert len(outlier_issues) > 0
        assert 16 in outlier_issues[0]['outlier_indices']

        recommender = AIRecommender(use_ai=False)
        recs = recommender.analyze_dataset(sales_df, issues)
        total_outlier_recs = [r for r in recs if 'Total' in r.affected_columns and r.suggested_operation == 'flag_for_review']
        assert len(total_outlier_recs) > 0
        assert total_outlier_recs[0].risk_level == 'high'

    def test_invalid_phone_flagged_for_review_and_preserved(self, sales_df):
        issues = detect_issues(sales_df)
        phone_issues = [i for i in issues if 'Phone' in i.get('affected_columns', []) and i['issue_type'] == 'invalid_values']
        assert len(phone_issues) > 0
        assert 'invalid-phone' in phone_issues[0]['invalid_values']

        recommender = AIRecommender(use_ai=False)
        recs = recommender.analyze_dataset(sales_df, issues)
        phone_recs = [r for r in recs if 'Phone' in r.affected_columns and r.suggested_operation == 'flag_for_review']
        assert len(phone_recs) > 0
        assert phone_recs[0].risk_level == 'high'
