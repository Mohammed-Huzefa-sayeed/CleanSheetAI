import unittest
import pandas as pd
import numpy as np
from validation_engine import ValidationEngine

class TestValidationEngine(unittest.TestCase):
    def setUp(self):
        self.df_before = pd.DataFrame({
            'customer_id': ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010'],
            'amount': [100.0, 200.0, 150.0, 300.0, 250.0, 400.0, 350.0, 500.0, 450.0, 600.0],
            'status': ['active'] * 10
        })

    def test_check_row_loss_warning_and_critical(self):
        validator = ValidationEngine()
        # Loss of 2 rows (20%) -> warning
        df_after_warning = self.df_before.iloc[:8]
        warnings = validator.check_row_loss(self.df_before, df_after_warning, 'drop_rows')
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].severity, 'warning')

        # Loss of 6 rows (60%) -> critical
        df_after_critical = self.df_before.iloc[:4]
        crit_warnings = validator.check_row_loss(self.df_before, df_after_critical, 'drop_rows')
        self.assertEqual(len(crit_warnings), 1)
        self.assertEqual(crit_warnings[0].severity, 'critical')

    def test_check_column_loss(self):
        validator = ValidationEngine()
        df_after = self.df_before.drop(columns=['status'])
        warnings = validator.check_column_loss(self.df_before, df_after)
        self.assertEqual(len(warnings), 1)
        self.assertIn('status', warnings[0].affected_columns)

    def test_check_missing_increase(self):
        validator = ValidationEngine()
        df_after = self.df_before.copy()
        df_after.loc[0, 'amount'] = np.nan
        warnings = validator.check_missing_increase(self.df_before, df_after)
        self.assertEqual(len(warnings), 1)
        self.assertIn('amount', warnings[0].affected_columns)

    def test_check_identifier_damage(self):
        validator = ValidationEngine()
        df_after = self.df_before.copy()
        # Strip leading zeros
        df_after['customer_id'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        warnings = validator.check_identifier_damage(self.df_before, df_after)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].category, 'identifier_damage')

    def test_check_numeric_totals(self):
        validator = ValidationEngine()
        df_after = self.df_before.copy()
        # Significant total sum shift
        df_after['amount'] = df_after['amount'] * 0.5
        warnings = validator.check_numeric_totals(self.df_before, df_after)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].category, 'total_change')

if __name__ == '__main__':
    unittest.main()
