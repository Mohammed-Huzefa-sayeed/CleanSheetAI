"""
Regression tests for CleanSheet AI using cleansheet_messy_customers.csv.

These tests verify the six core data-integrity requirements:
  1. Unique/contact fields (email) are NEVER auto-filled from another record.
  2. Invalid values ('unknown' in Age) are flagged.
  3. Out-of-range values (Age 145) are flagged as suspicious outliers.
  4. C001/C002 and C003/C004 are detected as potential duplicate pairs.
  5. Quality score reflects unresolved issues and is NOT 100.
  6. Rejected operations leave current_df unchanged (undo safety).
"""
import os
import io
import sys
import pytest
import pandas as pd

# Ensure the project root is on sys.path regardless of how pytest is invoked
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_manager import load_file, detect_issues, compute_quality_score, apply_operation, accept_change, undo_last_change
from ai_recommender import AIRecommender
from duplicate_detector import DuplicateDetector

CSV_PATH = os.path.join(PROJECT_ROOT, "cleansheet_messy_customers.csv")


@pytest.fixture(scope="module")
def dataset_info():
    """Load the messy customers CSV once for all tests in this module."""
    assert os.path.exists(CSV_PATH), f"Test fixture not found: {CSV_PATH}"
    with open(CSV_PATH, "rb") as f:
        content = f.read()
    info = load_file(io.BytesIO(content), "cleansheet_messy_customers.csv")
    return info


@pytest.fixture(scope="module")
def df(dataset_info):
    return dataset_info.current_df.copy()


@pytest.fixture(scope="module")
def issues(df):
    return detect_issues(df)


# ---------------------------------------------------------------------------
# Requirement 1 — Missing unique fields must NEVER be auto-filled from mode
# ---------------------------------------------------------------------------

class TestMissingEmailProtection:
    """C006 (Meera Nair) has a blank Email.  The recommender must NOT
    suggest filling it with another customer's email (mode strategy)."""

    def test_ai_recommender_does_not_suggest_mode_fill_for_email(self, df, issues):
        recommender = AIRecommender(use_ai=False)
        recs = recommender.analyze_dataset(df, issues)
        email_fill_recs = [
            r for r in recs
            if any(col.lower() in ('email', 'e-mail') for col in r.affected_columns)
            and r.suggested_operation == 'fill_missing'
            and r.operation_params.get('strategy') in ('mode', 'mean')
        ]
        assert len(email_fill_recs) == 0, (
            "AI recommender must NEVER suggest fill_missing with mode/mean for the Email column. "
            f"Found: {email_fill_recs}"
        )

    def test_ai_recommender_flags_email_missing_as_high_risk(self, df, issues):
        recommender = AIRecommender(use_ai=False)
        recs = recommender.analyze_dataset(df, issues)
        # The recommendation generated for the MISSING EMAIL issue must be high risk.
        # (Normalising email case is legitimately low risk since emails are case-insensitive.)
        email_missing_recs = [
            r for r in recs
            if any(col.lower() in ('email', 'e-mail') for col in r.affected_columns)
            and r.issue_type == 'missing_values'
        ]
        assert len(email_missing_recs) > 0, (
            "There should be a recommendation for the missing Email value."
        )
        for rec in email_missing_recs:
            assert rec.risk_level == 'high', (
                f"Missing email recommendation must be 'high' risk, got '{rec.risk_level}'. "
                f"Operation: {rec.suggested_operation}"
            )
            assert rec.suggested_operation != 'fill_missing', (
                "Missing email must never generate a fill_missing recommendation."
            )


    def test_email_column_missing_row_preserved_after_rejected_op(self, dataset_info):
        """Applying and then rejecting an operation must leave Email intact."""
        import copy
        # Deep-copy to avoid mutating the module-scope fixture
        import pandas as pd
        info_copy = load_file(
            io.BytesIO(open(CSV_PATH, "rb").read()),
            "cleansheet_messy_customers.csv",
        )
        original_email_values = info_copy.original_df['Email'].tolist()

        # Attempt a fill_missing with mode on Email — user rejects (does NOT call accept_change)
        new_df, _ = apply_operation(info_copy, 'fill_missing', {'column': 'Email', 'strategy': 'mode'})
        # current_df must be unchanged because we never called accept_change
        current_emails = info_copy.current_df['Email'].tolist()
        assert current_emails == original_email_values, (
            "current_df must not be modified when a change is rejected (apply_operation must not mutate in-place)."
        )

    def test_exported_csv_does_not_overwrite_missing_email(self, dataset_info):
        """The original dataset's Email for Meera Nair must remain NaN/blank."""
        meera_rows = dataset_info.original_df[dataset_info.original_df['Full Name'].str.strip() == 'Meera Nair']
        assert len(meera_rows) > 0, "Meera Nair row not found in dataset."
        meera_email = meera_rows['Email'].iloc[0]
        assert pd.isna(meera_email) or str(meera_email).strip() == '', (
            f"Meera Nair's original email must be blank/NaN, got '{meera_email}'"
        )


# ---------------------------------------------------------------------------
# Requirement 2 — 'unknown' Age must be flagged as invalid
# ---------------------------------------------------------------------------

class TestInvalidAgeDetection:

    def test_unknown_age_is_flagged(self, issues):
        age_issue_types = {i['issue_type'] for i in issues if 'Age' in i.get('affected_columns', [])}
        assert 'invalid_values' in age_issue_types, (
            f"'unknown' in Age column must be detected as invalid_values. Detected issue types: {age_issue_types}"
        )

    def test_unknown_age_description_mentions_unknown(self, issues):
        age_issues = [
            i for i in issues
            if 'Age' in i.get('affected_columns', []) and i.get('issue_type') == 'invalid_values'
        ]
        assert len(age_issues) > 0
        desc = age_issues[0].get('description', '')
        assert 'unknown' in desc.lower() or 'unknown' in str(age_issues[0].get('invalid_values', [])), (
            f"The invalid_values issue for Age should mention 'unknown'. Got: {desc}"
        )


# ---------------------------------------------------------------------------
# Requirement 3 — Age 145 must be flagged as suspicious/out-of-range
# ---------------------------------------------------------------------------

class TestOutlierAgeDetection:

    def test_age_145_is_flagged_as_suspicious_outlier(self, issues):
        outlier_issues = [
            i for i in issues
            if 'Age' in i.get('affected_columns', [])
            and i.get('issue_type') == 'suspicious_outliers'
        ]
        assert len(outlier_issues) > 0, (
            "Age 145 (Sara Khan) must be flagged as a suspicious_outliers issue."
        )

    def test_age_145_description_or_indices_includes_sara_row(self, df, issues):
        # Sara Khan is the last row (index 6 in 0-based)
        sara_idx = df[df['Full Name'].str.strip() == 'Sara Khan'].index.tolist()
        assert len(sara_idx) > 0, "Sara Khan row not found."
        outlier_issues = [
            i for i in issues
            if 'Age' in i.get('affected_columns', [])
            and i.get('issue_type') == 'suspicious_outliers'
        ]
        assert len(outlier_issues) > 0
        issue_indices = outlier_issues[0].get('outlier_indices', [])
        assert sara_idx[0] in issue_indices, (
            f"Sara Khan's row index {sara_idx[0]} must be in outlier_indices. Got: {issue_indices}"
        )


# ---------------------------------------------------------------------------
# Requirement 4 — Fuzzy duplicate detection: C001/C002 and C003/C004
# ---------------------------------------------------------------------------

class TestFuzzyDuplicateDetection:

    def _run_detection(self, df):
        detector = DuplicateDetector(similarity_threshold=80.0)
        # Compare on Full Name and Email (both are informative)
        cols = ['Full Name', 'Email']
        groups = detector.find_fuzzy_duplicates(df, columns=cols)
        return groups

    def test_c001_c002_detected_as_fuzzy_duplicates(self, df):
        groups = self._run_detection(df)
        c001_idx = df[df['Customer_ID'] == 'C001'].index.tolist()
        c002_idx = df[df['Customer_ID'] == 'C002'].index.tolist()
        assert c001_idx and c002_idx, "C001 and C002 rows not found."
        matched = any(
            c001_idx[0] in g.indices and c002_idx[0] in g.indices
            for g in groups
        )
        assert matched, (
            "C001 (Mohammed Ali) and C002 (Mohamed Ali) must be detected as fuzzy duplicates."
        )

    def test_c003_c004_detected_as_fuzzy_duplicates(self, df):
        groups = self._run_detection(df)
        c003_idx = df[df['Customer_ID'] == 'C003'].index.tolist()
        c004_idx = df[df['Customer_ID'] == 'C004'].index.tolist()
        assert c003_idx and c004_idx, "C003 and C004 rows not found."
        matched = any(
            c003_idx[0] in g.indices and c004_idx[0] in g.indices
            for g in groups
        )
        assert matched, (
            "C003 and C004 (both Priya Sharma, emails differ only in case) must be detected as fuzzy duplicates."
        )

    def test_fuzzy_groups_have_match_type_fuzzy(self, df):
        groups = self._run_detection(df)
        assert len(groups) > 0, "Expected fuzzy duplicate groups."
        for g in groups:
            assert g.match_type == 'fuzzy', f"Groups from find_fuzzy_duplicates should have match_type='fuzzy', got '{g.match_type}'."

    def test_no_auto_merge_of_fuzzy_duplicates(self, df):
        """Fuzzy duplicates must be reported, not automatically merged into the DataFrame."""
        groups = self._run_detection(df)
        # The DataFrame must not have fewer rows just because fuzzy groups were found
        assert len(df) == 7, "Finding fuzzy duplicates must not remove or merge rows from the DataFrame."


# ---------------------------------------------------------------------------
# Requirement 5 — Quality score must not be 100 for the messy test file
# ---------------------------------------------------------------------------

class TestQualityScoring:

    def test_quality_score_is_not_100(self, df):
        score = compute_quality_score(df)
        overall = score['overall_score']
        assert overall < 95.0, (
            f"Quality score for the unclean test file must be < 95, got {overall}. "
            "Consistency and validity must be calculated from actual data."
        )

    def test_validity_component_penalised(self, df):
        score = compute_quality_score(df)
        validity = score['component_scores']['validity']
        assert validity < 1.0, (
            f"validity component must be < 1.0 due to 'unknown' Age and Age 145. Got {validity}."
        )

    def test_consistency_component_penalised(self, df):
        score = compute_quality_score(df)
        consistency = score['component_scores']['consistency']
        assert consistency < 1.0, (
            f"consistency component must be < 1.0 due to mixed date formats and casing. Got {consistency}."
        )

    def test_score_components_present(self, df):
        score = compute_quality_score(df)
        for key in ('completeness', 'uniqueness', 'consistency', 'validity'):
            assert key in score['component_scores'], f"Missing component '{key}' in quality score."
            val = score['component_scores'][key]
            assert 0.0 <= val <= 1.0, f"Component '{key}' value {val} must be between 0 and 1."


# ---------------------------------------------------------------------------
# Requirement 6 — Undo / rejected-operation safety
# ---------------------------------------------------------------------------

class TestUndoSafety:

    def _fresh_info(self):
        with open(CSV_PATH, "rb") as f:
            return load_file(io.BytesIO(f.read()), "cleansheet_messy_customers.csv")

    def test_apply_operation_does_not_mutate_current_df(self):
        info = self._fresh_info()
        before_hash = pd.util.hash_pandas_object(info.current_df).sum()
        # Apply but do NOT accept
        apply_operation(info, 'trim_whitespace', {'columns': None})
        after_hash = pd.util.hash_pandas_object(info.current_df).sum()
        assert before_hash == after_hash, "apply_operation must not mutate current_df."

    def test_undo_restores_previous_state(self):
        info = self._fresh_info()
        original_len = len(info.current_df)
        new_df, record = apply_operation(info, 'drop_rows_with_missing', {'columns': ['Email']})
        accept_change(info, new_df, record)
        assert len(info.current_df) < original_len, "Row with missing email should have been dropped."
        success = undo_last_change(info)
        assert success, "undo_last_change should return True."
        assert len(info.current_df) == original_len, "Undo must restore the original row count."

    def test_undo_on_empty_history_returns_false(self):
        info = self._fresh_info()
        result = undo_last_change(info)
        assert result is False, "undo_last_change with empty history must return False."


# ---------------------------------------------------------------------------
# Requirement 7 — AI Assistant recommendation mappings & Apply execution
# ---------------------------------------------------------------------------

class TestAIAssistantApplyExecution:

    def test_recommendation_mappings_correct(self, df, issues):
        recommender = AIRecommender(use_ai=False)
        recs = recommender.analyze_dataset(df, issues)

        rec_ops = {r.issue_type: r.suggested_operation for r in recs}
        
        # Verify appropriate operations, NOT trim_whitespace for invalid or outlier values
        assert rec_ops.get('inconsistent_case') == 'normalize_case'
        assert rec_ops.get('inconsistent_categories') == 'normalize_categories'
        assert rec_ops.get('invalid_values') == 'convert_to_numeric'
        assert rec_ops.get('suspicious_outliers') == 'flag_for_review'
        assert rec_ops.get('missing_values') == 'leave_unchanged'

    def test_apply_normalize_case_changes_casing(self):
        with open(CSV_PATH, "rb") as f:
            info = load_file(io.BytesIO(f.read()), "cleansheet_messy_customers.csv")
        new_df, record = apply_operation(info, 'normalize_case', {'column': 'City', 'case': 'lower'})
        accept_change(info, new_df, record)
        assert all(c == c.lower() for c in info.current_df['City']), "All cities should be lowercased"
        assert len(info.change_history) == 1

    def test_apply_normalize_categories_changes_intended_values(self):
        with open(CSV_PATH, "rb") as f:
            info = load_file(io.BytesIO(f.read()), "cleansheet_messy_customers.csv")
        new_df, record = apply_operation(info, 'normalize_categories', {'column': 'City', 'mapping': {}})
        accept_change(info, new_df, record)
        # 'HYDERABAD' should be mapped to 'Hyderabad'
        assert 'HYDERABAD' not in info.current_df['City'].values
        assert (info.current_df['City'] == 'Hyderabad').sum() == 4
        assert len(info.change_history) == 1

    def test_converting_unknown_age_produces_missing_value_safely(self):
        with open(CSV_PATH, "rb") as f:
            info = load_file(io.BytesIO(f.read()), "cleansheet_messy_customers.csv")
        new_df, record = apply_operation(info, 'convert_to_numeric', {'column': 'Age', 'errors': 'coerce'})
        accept_change(info, new_df, record)
        # Arjun Reddy's age should now be NaN, and Sara Khan's age remains 145.0
        arjun_row = info.current_df[info.current_df['Full Name'].str.strip() == 'Arjun Reddy']
        sara_row = info.current_df[info.current_df['Full Name'].str.strip() == 'Sara Khan']
        assert pd.isna(arjun_row['Age'].iloc[0])
        assert float(sara_row['Age'].iloc[0]) == 145.0

    def test_suspicious_age_flag_for_review_leaves_age_unchanged(self):
        with open(CSV_PATH, "rb") as f:
            info = load_file(io.BytesIO(f.read()), "cleansheet_messy_customers.csv")
        new_df, record = apply_operation(info, 'flag_for_review', {'column': 'Age'})
        accept_change(info, new_df, record)
        # Sara Khan's age remains '145'
        sara_row = info.current_df[info.current_df['Full Name'].str.strip() == 'Sara Khan']
        assert str(sara_row['Age'].iloc[0]).strip() == '145'
        assert len(info.current_df) == 7

    def test_exported_files_reflect_accepted_changes(self):
        from reporting import export_clean_csv, export_clean_excel
        with open(CSV_PATH, "rb") as f:
            info = load_file(io.BytesIO(f.read()), "cleansheet_messy_customers.csv")
        new_df, record = apply_operation(info, 'normalize_case', {'column': 'City', 'case': 'lower'})
        accept_change(info, new_df, record)
        csv_bytes = export_clean_csv(info.current_df)
        csv_str = csv_bytes.decode('utf-8')
        assert 'hyderabad' in csv_str
        assert 'HYDERABAD' not in csv_str


# ---------------------------------------------------------------------------
# Requirement 8 — Duplicate comparison view on non-consecutive / filtered index
# ---------------------------------------------------------------------------

class TestDuplicateComparisonNonConsecutiveIndex:

    def test_get_comparison_view_non_consecutive_index(self):
        df_filtered = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Alice'],
            'email': ['a@a.com', 'b@b.com', 'a@a.com']
        }, index=[10, 20, 30])
        
        detector = DuplicateDetector()
        exact_groups = detector.find_exact_duplicates(df_filtered, ['name', 'email'])
        assert len(exact_groups) > 0
        comp_df = detector.get_comparison_view(df_filtered, exact_groups[0])
        assert len(comp_df) == 2
        assert list(comp_df.index) == [10, 30]

    def test_resolve_group_non_consecutive_index(self):
        df_filtered = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Alice'],
            'email': ['a@a.com', 'b@b.com', 'a@a.com']
        }, index=[10, 20, 30])
        
        detector = DuplicateDetector()
        exact_groups = detector.find_exact_duplicates(df_filtered, ['name', 'email'])
        resolved = detector.resolve_group(df_filtered, exact_groups[0], 'keep_first')
        assert len(resolved) == 2
        assert list(resolved.index) == [10, 20]

