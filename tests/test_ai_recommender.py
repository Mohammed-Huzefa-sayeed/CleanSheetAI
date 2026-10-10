import unittest
import pandas as pd
from ai_recommender import AIRecommender, CleaningRecommendation

class TestAIRecommender(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['  Alice ', 'Bob', 'Alice'],
            'amount': ['100', 'invalid', '200'],
            'zip': ['00123', '00456', '00789']
        })
        self.issues = [
            {'issue_type': 'missing_values', 'affected_columns': ['amount'], 'affected_rows': 1, 'description': 'Missing values found'},
            {'issue_type': 'duplicate_rows', 'affected_columns': ['name', 'amount', 'zip'], 'affected_rows': 1, 'description': 'Duplicate rows found'},
            {'issue_type': 'whitespace_issues', 'affected_columns': ['name'], 'affected_rows': 2, 'description': 'Whitespace issues detected'},
            {'issue_type': 'mixed_types', 'affected_columns': ['amount'], 'affected_rows': 1, 'description': 'Mixed types detected'}
        ]

    def test_rule_based_fallback_generation(self):
        # AIRecommender with use_ai=False should generate rule-based recommendations reliably
        recommender = AIRecommender(use_ai=False)
        recs = recommender.analyze_dataset(self.df, self.issues)
        self.assertGreater(len(recs), 0)
        
        # Verify recommendation structure
        for rec in recs:
            self.assertEqual(rec.source, 'rule_based')
            self.assertIn(rec.risk_level, ['low', 'medium', 'high'])
            self.assertIsInstance(rec.confidence, float)
            self.assertTrue(hasattr(rec, 'suggested_operation'))

    def test_answer_dataset_question_without_ai(self):
        recommender = AIRecommender(use_ai=False)
        ans = recommender.answer_dataset_question(self.df, "How many rows and columns are there?")
        self.assertIsInstance(ans, str)
        self.assertIn("3", ans) # 3 rows

    def test_recommendations_summary(self):
        recommender = AIRecommender(use_ai=False)
        recommender.analyze_dataset(self.df, self.issues)
        summary = recommender.get_recommendations_summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

if __name__ == '__main__':
    unittest.main()
