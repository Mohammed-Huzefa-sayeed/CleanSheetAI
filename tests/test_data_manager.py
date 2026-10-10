import unittest
import io
import pandas as pd
from data_manager import load_file, get_dataset_profile, get_column_profile, detect_issues, compute_quality_score, apply_operation, accept_change, undo_last_change

class TestDataManager(unittest.TestCase):
    def setUp(self):
        self.csv_data = (
            "id,name,amount,notes\n"
            "00123,  Alice ,100.5,test\n"
            "00124,Bob,200.0,\n"
            "00123,  Alice ,100.5,test\n"
            ",Charlie,invalid,last\n"
        ).encode('utf-8')

    def test_load_csv_preserves_identifiers(self):
        file_obj = io.BytesIO(self.csv_data)
        dataset_info = load_file(file_obj, "test.csv")
        self.assertEqual(dataset_info.row_count, 4)
        # Leading zero must be preserved
        self.assertEqual(dataset_info.current_df['id'].iloc[0], "00123")

    def test_empty_file_handling(self):
        empty_obj = io.BytesIO(b"")
        with self.assertRaises(ValueError):
            load_file(empty_obj, "empty.csv")

    def test_dataset_profiling(self):
        file_obj = io.BytesIO(self.csv_data)
        dataset_info = load_file(file_obj, "test.csv")
        profile = get_dataset_profile(dataset_info.current_df)
        self.assertEqual(profile['row_count'], 4)
        self.assertEqual(profile['col_count'], 4)
        self.assertGreater(profile['total_missing'], 0)

    def test_detect_issues(self):
        file_obj = io.BytesIO(self.csv_data)
        dataset_info = load_file(file_obj, "test.csv")
        issues = detect_issues(dataset_info.current_df)
        issue_types = [i['issue_type'] for i in issues]
        self.assertIn('missing_values', issue_types)
        self.assertIn('duplicate_rows', issue_types)

    def test_apply_and_undo_operation(self):
        file_obj = io.BytesIO(self.csv_data)
        dataset_info = load_file(file_obj, "test.csv")
        initial_len = len(dataset_info.current_df)
        
        # Remove duplicates
        new_df, change_rec = apply_operation(dataset_info, 'remove_duplicates', {'keep': 'first'})
        self.assertEqual(len(new_df), initial_len - 1)
        
        # Accept change
        accept_change(dataset_info, new_df, change_rec)
        self.assertEqual(len(dataset_info.current_df), initial_len - 1)
        self.assertEqual(len(dataset_info.change_history), 1)
        
        # Undo change
        success = undo_last_change(dataset_info)
        self.assertTrue(success)
        self.assertEqual(len(dataset_info.current_df), initial_len)

    def test_quality_score_calculation(self):
        file_obj = io.BytesIO(self.csv_data)
        dataset_info = load_file(file_obj, "test.csv")
        qs = compute_quality_score(dataset_info.current_df)
        self.assertIn('overall_score', qs)
        self.assertTrue(0 <= qs['overall_score'] <= 100)
        self.assertIn('completeness', qs['component_scores'])

if __name__ == '__main__':
    unittest.main()
