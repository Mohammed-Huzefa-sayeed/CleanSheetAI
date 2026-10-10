import unittest
import os
import json
import shutil
from recipe_manager import RecipeManager, CleaningRecipe, RecipeStep

class TestRecipeManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_recipes_tmp'
        os.makedirs(self.test_dir, exist_ok=True)
        self.rm = RecipeManager(recipes_dir=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_create_and_save_recipe(self):
        recipe = self.rm.create_recipe(
            name="Sales Standardizer",
            description="Cleans sales datasets",
            expected_columns=["order_id", "customer_name", "amount"],
            steps=[
                {'operation': 'trim_whitespace', 'params': {'columns': ['customer_name']}, 'description': 'Trim name'},
                {'operation': 'remove_duplicates', 'params': {'keep': 'first'}, 'description': 'Drop dups'}
            ]
        )
        self.assertEqual(len(recipe.steps), 2)
        saved_path = self.rm.save_recipe(recipe)
        self.assertTrue(os.path.exists(saved_path))

        # Reload recipe
        loaded = self.rm.load_recipe(saved_path)
        self.assertEqual(loaded.name, "Sales Standardizer")
        self.assertEqual(len(loaded.steps), 2)

    def test_recipe_validation_rejection_of_unallowed_ops(self):
        with self.assertRaises(ValueError):
            self.rm.create_recipe(
                name="Malicious Recipe",
                description="Tries unallowed op",
                expected_columns=["a"],
                steps=[{'operation': 'execute_arbitrary_code', 'params': {}}]
            )

    def test_recipe_compatibility_check(self):
        recipe = self.rm.create_recipe(
            name="Test Recipe",
            description="Requires specific cols",
            expected_columns=["col_a", "col_b", "col_c"],
            steps=[
                {'operation': 'fill_missing', 'params': {'column': 'col_a'}, 'description': 'Fill A'},
                {'operation': 'trim_whitespace', 'params': {'column': 'col_z'}, 'description': 'Trim missing Z'}
            ]
        )
        compat = self.rm.check_compatibility(recipe, df_columns=["col_a", "col_b"])
        self.assertFalse(compat['compatible'])
        self.assertIn('col_c', compat['missing_columns'])
        self.assertIn(1, compat['compatible_steps'])

if __name__ == '__main__':
    unittest.main()
