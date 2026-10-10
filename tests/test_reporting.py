import unittest
import pandas as pd
from reporting import (
    generate_pdf_report, export_clean_csv, export_clean_excel,
    sanitize_dataframe_for_export, generate_comparison_data
)

class TestReporting(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'formula_inj': ['=cmd|"/c calc"!A0', '+123', '@SUM(A1:A5)', '-456', 'Normal'],
            'price': [10.5, 20.0, 30.2, 40.0, 50.5],
            'notes': ['ok', 'fine', 'good', 'great', 'excellent']
        })

    def test_sanitize_dataframe_prevents_formula_injection(self):
        sanitized = sanitize_dataframe_for_export(self.df)
        self.assertTrue(sanitized['formula_inj'].iloc[0].startswith("'="))
        self.assertTrue(sanitized['formula_inj'].iloc[1].startswith("'+"))
        self.assertTrue(sanitized['formula_inj'].iloc[2].startswith("'@"))
        self.assertTrue(sanitized['formula_inj'].iloc[3].startswith("'-"))
        self.assertEqual(sanitized['formula_inj'].iloc[4], "Normal")

    def test_export_clean_csv(self):
        csv_bytes = export_clean_csv(self.df, sanitize=True)
        self.assertIsInstance(csv_bytes, bytes)
        self.assertGreater(len(csv_bytes), 0)
        csv_str = csv_bytes.decode('utf-8')
        self.assertIn("'=cmd", csv_str)

    def test_export_clean_excel(self):
        xlsx_bytes = export_clean_excel(self.df, sheet_name="Clean Data", sanitize=True)
        self.assertIsInstance(xlsx_bytes, bytes)
        self.assertGreater(len(xlsx_bytes), 0)

    def test_generate_pdf_report(self):
        pdf_bytes = generate_pdf_report(
            dataset_info={'name': 'sales.csv', 'rows': 5, 'columns': 3},
            quality_before={'missing_cells': 5, 'duplicate_rows': 2, 'quality_score': 65.0},
            quality_after={'missing_cells': 0, 'duplicate_rows': 0, 'quality_score': 95.0},
            issues_detected=[{'description': 'Duplicates found', 'severity': 'high'}],
            operations_applied=[{'description': 'Removed duplicate rows'}],
            validation_warnings=['Checked leading zeros safely']
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b'%PDF'))

    def test_generate_comparison_data(self):
        df_clean = self.df.iloc[:3].copy()
        comp = generate_comparison_data(self.df, df_clean)
        self.assertEqual(comp['rows_original'], 5)
        self.assertEqual(comp['rows_cleaned'], 3)
        self.assertIn('cells_changed', comp)

if __name__ == '__main__':
    unittest.main()
