import unittest
import pandas as pd
import numpy as np
from cleaning_engine import (
    fill_missing_with_strategy, drop_rows_with_missing,
    trim_whitespace, normalize_case, normalize_categories,
    normalize_column_names, convert_to_numeric, parse_dates,
    normalize_booleans, rename_column, drop_columns,
    split_column, merge_columns, detect_numeric_outliers
)

class TestCleaningEngine(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['  Alice ', 'BOB', 'charlie  ', 'Alice'],
            'age': ['25', '30', 'invalid', '40'],
            'salary': [50000, 60000, np.nan, 200000],
            'active': ['yes', 'no', 'TRUE', '0'],
            'date': ['2023-01-15', '15/02/2023', '2023/03/20', 'invalid_date'],
            'category': ['Electronics', 'electronics', 'ELECTRONICS', 'Home']
        })

    def test_trim_whitespace(self):
        cleaned = trim_whitespace(self.df, columns=['name'])
        self.assertEqual(cleaned['name'].iloc[0], 'Alice')
        self.assertEqual(cleaned['name'].iloc[2], 'charlie')

    def test_normalize_case(self):
        cleaned = normalize_case(self.df, column='name', case='title')
        self.assertEqual(cleaned['name'].iloc[1], 'Bob')

    def test_fill_missing_numeric(self):
        cleaned = fill_missing_with_strategy(self.df, column='salary', strategy='mean')
        self.assertFalse(cleaned['salary'].isnull().any())
        self.assertAlmostEqual(cleaned['salary'].iloc[2], (50000 + 60000 + 200000) / 3)

    def test_convert_to_numeric(self):
        cleaned, failed = convert_to_numeric(self.df, column='age', errors='coerce')
        self.assertEqual(failed, 1)
        self.assertTrue(pd.isna(cleaned['age'].iloc[2]))
        self.assertEqual(cleaned['age'].iloc[0], 25)

    def test_normalize_column_names(self):
        dirty_cols_df = pd.DataFrame(columns=[' First Name ', 'Order-ID#', 'total $'])
        cleaned, mapping = normalize_column_names(dirty_cols_df)
        self.assertIn('first_name', cleaned.columns)
        self.assertIn('order_id', cleaned.columns)
        self.assertEqual(mapping['first_name'], ' First Name ')

    def test_normalize_booleans(self):
        cleaned = normalize_booleans(self.df, column='active')
        self.assertEqual(cleaned['active'].iloc[0], True)
        self.assertEqual(cleaned['active'].iloc[1], False)
        self.assertEqual(cleaned['active'].iloc[2], True)
        self.assertEqual(cleaned['active'].iloc[3], False)

    def test_parse_dates(self):
        cleaned, failed = parse_dates(self.df, column='date')
        self.assertEqual(failed, 1)
        self.assertTrue(pd.notna(cleaned['date'].iloc[0]))
        self.assertTrue(pd.isna(cleaned['date'].iloc[3]))

    def test_split_and_merge_columns(self):
        df_split = pd.DataFrame({'fullname': ['John Doe', 'Jane Smith']})
        split_df = split_column(df_split, column='fullname', delimiter=' ', new_column_names=['first', 'last'])
        self.assertIn('first', split_df.columns)
        self.assertIn('last', split_df.columns)
        self.assertEqual(split_df['first'].iloc[0], 'John')

        merged_df = merge_columns(split_df, columns=['first', 'last'], new_column_name='recombined', separator='-')
        self.assertEqual(merged_df['recombined'].iloc[0], 'John-Doe')

    def test_detect_numeric_outliers(self):
        df_outliers = pd.DataFrame({'value': [10, 11, 12, 10, 11, 10, 500]})
        outliers = detect_numeric_outliers(df_outliers, column='value', method='iqr', threshold=1.5)
        self.assertEqual(len(outliers), 1)
        self.assertEqual(outliers['value'].iloc[0], 500)

if __name__ == '__main__':
    unittest.main()
