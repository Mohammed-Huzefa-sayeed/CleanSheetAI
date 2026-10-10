import os
import re
from typing import Any
from dotenv import load_dotenv

# App constants
APP_NAME = 'CleanSheet AI'
TAGLINE = 'From messy data to decision-ready data.'
VERSION = '1.0.0'

MAX_FILE_SIZE_MB = 200
ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls']
DEFAULT_SIMILARITY_THRESHOLD = 85

QUALITY_SCORE_WEIGHTS = {
    'completeness': 0.3,
    'uniqueness': 0.25,
    'consistency': 0.25,
    'validity': 0.2
}

# AI config
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4')
OPENAI_MAX_TOKENS = int(os.environ.get('OPENAI_MAX_TOKENS', '2000'))
AI_ENABLED = os.environ.get('AI_ENABLED', 'true').lower() == 'true'

DANGEROUS_PATTERNS = [
    r'^[=+\-@]',  # Starts with =, +, -, @
    r'^\s',      # Starts with whitespace
    r'\t|\r'     # Contains tab or carriage return
]

def load_env() -> None:
    """Load environment variables from .env file."""
    load_dotenv()

def get_openai_api_key() -> str | None:
    """Read OpenAI API key from environment, never expose it."""
    return os.environ.get('OPENAI_API_KEY')

def sanitize_cell_value(value: Any) -> Any:
    """Sanitize cell value to prevent formula injection."""
    if not isinstance(value, str):
        return value
        
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, value):
            return f"'{value}"
            
    return value
