import json
import os
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict
import hashlib

@dataclass
class RecipeStep:
    step_number: int
    operation: str
    params: dict
    description: str
    validation_rules: list[str] = field(default_factory=list)

@dataclass 
class CleaningRecipe:
    name: str
    description: str
    expected_columns: list[str]
    steps: list[RecipeStep]
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = '1.0'
    recipe_id: str = ''
    
    def __post_init__(self):
        if not self.recipe_id:
            self.recipe_id = hashlib.md5(f"{self.name}{self.creation_date}".encode()).hexdigest()[:12]

ALLOWED_OPERATIONS = [
    'trim_whitespace', 'normalize_case', 'fill_missing', 'drop_rows_with_missing',
    'remove_duplicates', 'convert_to_numeric', 'parse_dates', 'normalize_booleans',
    'normalize_categories', 'rename_column', 'drop_column', 'split_column',
    'merge_columns', 'normalize_column_names', 'convert_blanks_to_nan',
    'fill_with_value', 'leave_unchanged', 'flag_for_review'
]


class RecipeManager:
    """Manages cleaning recipes for reuse."""
    
    def __init__(self, recipes_dir: str = 'recipes'):
        self.recipes_dir = recipes_dir
        os.makedirs(recipes_dir, exist_ok=True)
    
    def create_recipe(self, name: str, description: str, expected_columns: list[str],
                       steps: list[dict]) -> CleaningRecipe:
        """Create a new recipe from a list of step definitions.
        Validate all operations are in ALLOWED_OPERATIONS.
        No executable code allowed in recipe."""
        recipe_steps = []
        for i, step in enumerate(steps, start=1):
            operation = step.get('operation', '')
            if operation not in ALLOWED_OPERATIONS:
                raise ValueError(f"Operation '{operation}' not allowed.")
            
            recipe_step = RecipeStep(
                step_number=i,
                operation=operation,
                params=step.get('params', {}),
                description=step.get('description', ''),
                validation_rules=step.get('validation_rules', [])
            )
            recipe_steps.append(recipe_step)
            
        recipe = CleaningRecipe(
            name=name,
            description=description,
            expected_columns=expected_columns,
            steps=recipe_steps
        )
        return recipe
    
    def save_recipe(self, recipe: CleaningRecipe, filepath: str | None = None) -> str:
        """Save recipe as JSON file. Returns filepath.
        Validate recipe before saving."""
        errors = self.validate_recipe(recipe)
        if errors:
            raise ValueError(f"Recipe validation failed: {errors}")
            
        if not filepath:
            filename = f"{recipe.recipe_id}_{recipe.name.replace(' ', '_').lower()}.json"
            filepath = os.path.join(self.recipes_dir, filename)
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(recipe), f, indent=4)
            
        return filepath
    
    def load_recipe(self, filepath: str) -> CleaningRecipe:
        """Load recipe from JSON file.
        Validate structure, check for forbidden content (code, exec, eval, import).
        Validate all operations are allowed."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        data_str = json.dumps(data)
        forbidden = ['__', 'import', 'eval', 'exec', 'lambda', 'os.']
        for word in forbidden:
            if word in data_str:
                raise ValueError(f"Forbidden content detected: {word}")
                
        steps_data = data.pop('steps', [])
        steps = []
        for s_data in steps_data:
            steps.append(RecipeStep(**s_data))
            
        recipe = CleaningRecipe(steps=steps, **data)
        errors = self.validate_recipe(recipe)
        if errors:
            raise ValueError(f"Recipe validation failed: {errors}")
            
        return recipe
    
    def list_recipes(self) -> list[dict]:
        """List all saved recipes with name, description, date, version, step count."""
        recipes = []
        if not os.path.exists(self.recipes_dir):
            return recipes
        for filename in os.listdir(self.recipes_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.recipes_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        recipes.append({
                            'name': data.get('name', ''),
                            'description': data.get('description', ''),
                            'creation_date': data.get('creation_date', ''),
                            'version': data.get('version', ''),
                            'step_count': len(data.get('steps', [])),
                            'filepath': filepath
                        })
                except Exception:
                    pass
        return recipes
    
    def delete_recipe(self, filepath: str) -> bool:
        """Delete a recipe file."""
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
    
    def validate_recipe(self, recipe: CleaningRecipe) -> list[str]:
        """Validate recipe structure and content. Return list of validation errors.
        Check: name not empty, steps not empty, all operations allowed,
        no executable code in any field, params are dicts."""
        errors = []
        if not recipe.name:
            errors.append("Recipe name cannot be empty.")
        if not recipe.steps:
            errors.append("Recipe must have at least one step.")
            
        forbidden = ['__', 'import', 'eval', 'exec', 'lambda', 'os.']
        
        for step in recipe.steps:
            if step.operation not in ALLOWED_OPERATIONS:
                errors.append(f"Step {step.step_number}: operation '{step.operation}' is not allowed.")
            if not isinstance(step.params, dict):
                errors.append(f"Step {step.step_number}: params must be a dictionary.")
                
            step_str = str(step)
            for word in forbidden:
                if word in step_str:
                    errors.append(f"Step {step.step_number}: Forbidden content '{word}' detected.")
                    
        return errors
    
    def check_compatibility(self, recipe: CleaningRecipe, df_columns: list[str]) -> dict:
        """Check recipe compatibility with a dataset.
        Returns: {compatible: bool, missing_columns: list, extra_columns: list,
        compatible_steps: list[int], incompatible_steps: list[int], warnings: list[str]}."""
        expected = set(recipe.expected_columns)
        actual = set(df_columns)
        
        missing = list(expected - actual)
        extra = list(actual - expected)
        
        compatible = len(missing) == 0
        
        warnings = []
        if missing:
            warnings.append(f"Missing expected columns: {missing}")
            
        compatible_steps = []
        incompatible_steps = []
        
        for step in recipe.steps:
            col = step.params.get('column')
            if col and col in missing:
                incompatible_steps.append(step.step_number)
            else:
                compatible_steps.append(step.step_number)
                
        return {
            'compatible': compatible,
            'missing_columns': missing,
            'extra_columns': extra,
            'compatible_steps': compatible_steps,
            'incompatible_steps': incompatible_steps,
            'warnings': warnings
        }
    
    def create_recipe_from_history(self, change_history: list, df_columns: list[str],
                                     name: str, description: str) -> CleaningRecipe:
        """Create a recipe from cleaning change history."""
        steps = []
        for change in change_history:
            if hasattr(change, 'operation_name'):
                operation = change.operation_name
                params = getattr(change, 'parameters', {})
                desc = f"Applied {operation}"
            elif isinstance(change, dict):
                operation = change.get('operation') or change.get('operation_name', '')
                params = change.get('params') or change.get('parameters', {})
                desc = change.get('description', f"Applied {operation}")
            else:
                continue

            if operation in ALLOWED_OPERATIONS:
                steps.append({
                    'operation': operation,
                    'params': params,
                    'description': desc
                })
        return self.create_recipe(name, description, df_columns, steps)
    
    def recipe_to_json(self, recipe: CleaningRecipe) -> str:
        """Serialize recipe to JSON string."""
        return json.dumps(asdict(recipe), indent=4)
    
    def get_recipe_preview(self, recipe: CleaningRecipe) -> str:
        """Generate human-readable preview of recipe steps."""
        preview = f"Recipe: {recipe.name}\nDescription: {recipe.description}\n\nSteps:\n"
        for step in recipe.steps:
            preview += f"{step.step_number}. {step.operation} ({step.description})\n"
            preview += f"   Params: {step.params}\n"
        return preview
