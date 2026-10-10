import unittest
import pandas as pd
from duplicate_detector import DuplicateDetector, DuplicateGroup

class TestDuplicateDetector(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['John Smith', 'John Smith', 'Jon Smith', 'Jane Doe', 'Jane Doe'],
            'email': ['john@example.com', 'john@example.com', 'john.smith@example.com', 'jane@example.com', 'jane@example.com'],
            'city': ['New York', 'New York', 'New York City', 'Chicago', 'Chicago']
        })

    def test_find_exact_duplicates(self):
        detector = DuplicateDetector()
        groups = detector.find_exact_duplicates(self.df)
        self.assertEqual(len(groups), 2)
        # Verify group match type
        for g in groups:
            self.assertEqual(g.match_type, 'exact')
            self.assertGreaterEqual(len(g.indices), 2)

    def test_remove_exact_duplicates(self):
        detector = DuplicateDetector()
        new_df, removed_cnt = detector.remove_exact_duplicates(self.df, keep='first')
        self.assertEqual(removed_cnt, 2)
        self.assertEqual(len(new_df), 3)

    def test_find_fuzzy_duplicates(self):
        detector = DuplicateDetector(similarity_threshold=80.0)
        groups = detector.find_fuzzy_duplicates(self.df, columns=['name'])
        # 'John Smith' and 'Jon Smith' should match fuzzily
        self.assertGreater(len(groups), 0)
        found_jon_john = False
        for g in groups:
            matched_names = self.df.loc[g.indices, 'name'].tolist()
            if 'John Smith' in matched_names and 'Jon Smith' in matched_names:
                found_jon_john = True
        self.assertTrue(found_jon_john)

    def test_get_field_match_details(self):
        detector = DuplicateDetector()
        details = detector.get_field_match_details(self.df, 0, 2, columns=['name', 'city'])
        self.assertEqual(len(details), 2)
        name_detail = next(d for d in details if d['column'] == 'name')
        self.assertGreaterEqual(name_detail['similarity_score'], 80)

    def test_blocking_fuzzy_duplicates(self):
        detector = DuplicateDetector(similarity_threshold=80.0)
        groups = detector.find_fuzzy_duplicates_blocked(
            self.df, columns=['name'], blocking_column='name', blocking_method='first3'
        )
        self.assertIsInstance(groups, list)

if __name__ == '__main__':
    unittest.main()
