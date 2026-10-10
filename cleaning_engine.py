import pandas as pd
import numpy as np
from typing import Optional
import re
from datetime import datetime

# === Missing Value Operations ===

def fill_missing_with_strategy(df: pd.DataFrame, column: str, strategy: str, fill_value=None) -> pd.DataFrame:
    """Fill missing values in a column using the specified strategy.
    Strategies: 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'value' (uses fill_value), 'empty_string', 'zero'.
    For mean/median only works on numeric columns. Mode works on any type.
    Pre-processes the column with convert_blanks_to_nan to catch text tokens like 'None', 'null', 'NA', etc.
    Returns new DataFrame."""
    if column not in df.columns:
        return df.copy()
    
    new_df = df.copy()
    # Normalize common missing-value tokens to NaN first
    new_df = convert_blanks_to_nan(new_df, columns=[column])
    
    if strategy == 'mean':
        new_df[column] = new_df[column].fillna(new_df[column].mean())
    elif strategy == 'median':
        new_df[column] = new_df[column].fillna(new_df[column].median())
    elif strategy == 'mode':
        mode_vals = new_df[column].mode()
        if not mode_vals.empty:
            new_df[column] = new_df[column].fillna(mode_vals[0])
    elif strategy == 'forward_fill':
        new_df[column] = new_df[column].ffill()
    elif strategy == 'backward_fill':
        new_df[column] = new_df[column].bfill()
    elif strategy == 'value':
        new_df[column] = new_df[column].fillna(fill_value)
    elif strategy == 'empty_string':
        new_df[column] = new_df[column].fillna("")
    elif strategy == 'zero':
        new_df[column] = new_df[column].fillna(0)
    
    return new_df

def drop_rows_with_missing(df: pd.DataFrame, columns: list[str] | None = None, how: str = 'any') -> pd.DataFrame:
    """Drop rows with missing values. If columns specified, only check those columns.
    how='any' drops if any column is missing, 'all' drops only if all specified columns are missing."""
    return df.dropna(subset=columns, how=how)

def detect_blank_strings(df: pd.DataFrame) -> dict[str, int]:
    """Find columns with blank strings or placeholder missing tokens that aren't NaN.
    Returns {column_name: count_of_blank_strings}."""
    _MISSING_TOKENS = frozenset({
        '', 'none', 'null', 'nan', 'na', 'n/a', 'missing', '-', '--', '#n/a', '#na', '#null',
    })
    result = {}
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            stripped = df[col].astype(str).str.strip().str.lower()
            mask = stripped.isin(_MISSING_TOKENS) & df[col].notna()
            count = mask.sum()
            if count > 0:
                result[col] = int(count)
    return result

def convert_blanks_to_nan(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert blank strings and common missing-value tokens to NaN in specified columns (or all string columns).
    Recognized tokens: empty, whitespace-only, 'None', 'null', 'nan', 'NA', 'N/A', 'n/a', 'missing', '-', '--'."""
    _MISSING_TOKENS = frozenset({
        '', 'none', 'null', 'nan', 'na', 'n/a', 'missing', '-', '--', '#n/a', '#na', '#null',
    })
    new_df = df.copy()
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
    
    for col in columns:
        if col in new_df.columns:
            stripped = new_df[col].astype(str).str.strip()
            mask = stripped.str.lower().isin(_MISSING_TOKENS)
            new_df.loc[mask, col] = np.nan
    return new_df

# === Normalization Operations ===

def trim_whitespace(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Strip leading/trailing whitespace and normalize internal multiple spaces to single space.
    If columns is None, apply to all string/object columns."""
    new_df = df.copy()
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
        
    for col in columns:
        if col in new_df.columns and (pd.api.types.is_string_dtype(new_df[col]) or pd.api.types.is_object_dtype(new_df[col])):
            new_df[col] = new_df[col].apply(lambda x: re.sub(r'\s+', ' ', str(x).strip()) if pd.notna(x) and isinstance(x, str) else x)
    return new_df

def normalize_case(df: pd.DataFrame, column: str, case: str = 'lower') -> pd.DataFrame:
    """Normalize text case. Options: 'lower', 'upper', 'title', 'sentence'."""
    if column not in df.columns:
        return df.copy()
    
    new_df = df.copy()
    
    def to_sentence_case(text):
        if not isinstance(text, str) or not text:
            return text
        return text[0].upper() + text[1:].lower()
        
    if case == 'lower':
        new_df[column] = new_df[column].str.lower()
    elif case == 'upper':
        new_df[column] = new_df[column].str.upper()
    elif case == 'title':
        new_df[column] = new_df[column].str.title()
    elif case == 'sentence':
        new_df[column] = new_df[column].apply(lambda x: to_sentence_case(x) if pd.notna(x) else x)
        
    return new_df

def normalize_categories(df: pd.DataFrame, column: str, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    """Normalize category labels. If mapping provided, use it. Otherwise, auto-detect similar categories
    using case-insensitive comparison and whitespace normalization, and map them to the most common variant."""
    if column not in df.columns:
        return df.copy()
        
    new_df = df.copy()
    
    if mapping:
        new_df[column] = new_df[column].map(lambda x: mapping.get(x, x))
    else:
        # Auto-detect similar categories
        value_counts = new_df[column].value_counts()
        normalized_map = {}
        
        for val in value_counts.index:
            if pd.isna(val):
                continue
            # Normalize key: lowercase, strip whitespace
            norm_key = re.sub(r'\s+', ' ', str(val).strip().lower())
            
            if norm_key not in normalized_map:
                # Find the most frequent original value that normalizes to this key
                most_frequent_original = None
                max_count = -1
                for orig_val, count in value_counts.items():
                    if pd.isna(orig_val):
                        continue
                    if re.sub(r'\s+', ' ', str(orig_val).strip().lower()) == norm_key:
                        if count > max_count:
                            most_frequent_original = orig_val
                            max_count = count
                
                normalized_map[norm_key] = most_frequent_original
        
        def map_val(x):
            if pd.isna(x):
                return x
            norm = re.sub(r'\s+', ' ', str(x).strip().lower())
            return normalized_map.get(norm, x)
            
        new_df[column] = new_df[column].apply(map_val)
        
    return new_df

def normalize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize column names: lowercase, replace spaces/special chars with underscores,
    remove leading/trailing underscores, handle duplicates by appending _1, _2, etc.
    Returns (new_df, {new_name: original_name} mapping)."""
    new_df = df.copy()
    original_cols = new_df.columns.tolist()
    
    new_cols = []
    seen = {}
    
    for col in original_cols:
        # Lowercase, replace non-alphanumeric with underscore
        norm_col = re.sub(r'[^a-zA-Z0-9]+', '_', str(col).lower()).strip('_')
        if not norm_col:
            norm_col = 'column'
            
        if norm_col in seen:
            seen[norm_col] += 1
            norm_col = f"{norm_col}_{seen[norm_col]}"
        else:
            seen[norm_col] = 0
            
        new_cols.append(norm_col)
        
    new_df.columns = new_cols
    mapping = dict(zip(new_cols, original_cols))
    
    return new_df, mapping

def convert_to_numeric(df: pd.DataFrame, column: str, errors: str = 'coerce') -> tuple[pd.DataFrame, int]:
    """Convert column to numeric type. Remove common formatting (commas, currency symbols, %).
    Returns (new_df, count_of_failed_conversions)."""
    if column not in df.columns:
        return df.copy(), 0
        
    new_df = df.copy()
    
    def clean_numeric(val):
        if pd.isna(val):
            return val
        if isinstance(val, (int, float)):
            return val
        # Remove currency, commas, spaces, %
        s = str(val)
        s = re.sub(r'[$,\s€£¥%]', '', s)
        return s
        
    cleaned = new_df[column].apply(clean_numeric)
    
    # Calculate failed conversions before converting to numeric
    original_notna = new_df[column].notna()
    
    new_df[column] = pd.to_numeric(cleaned, errors=errors)
    
    failed_count = 0
    if errors == 'coerce':
        failed_mask = original_notna & new_df[column].isna()
        failed_count = failed_mask.sum()
        
    return new_df, int(failed_count)

def parse_dates(df: pd.DataFrame, column: str, date_format: str | None = None, dayfirst: bool = False) -> tuple[pd.DataFrame, int]:
    """Parse a column as dates. Try common formats if none specified.
    Preserves each valid original date and parses formats safely without cross-row corruption.
    Returns (new_df, count_of_failed_conversions)."""
    if column not in df.columns:
        return df.copy(), 0
        
    new_df = df.copy()
    original_notna = new_df[column].notna()
    
    if date_format:
        new_df[column] = pd.to_datetime(new_df[column], format=date_format, errors='coerce')
    else:
        common_formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%m-%d-%Y',
            '%b %d %Y',
            '%B %d, %Y',
            '%d %b %Y',
            '%Y%m%d'
        ]
        if dayfirst:
            common_formats = ['%d/%m/%Y', '%d-%m-%Y'] + [f for f in common_formats if f not in ('%d/%m/%Y', '%d-%m-%Y')]

        def _safe_parse_val(val):
            if pd.isna(val):
                return pd.NaT
            if isinstance(val, (datetime, pd.Timestamp)):
                return pd.Timestamp(val)
            s = str(val).strip()
            if not s or s.lower() in ('nat', 'nan', 'none', 'null', 'not-a-date', 'invalid', 'unknown', '-'):
                return pd.NaT
            for fmt in common_formats:
                try:
                    return pd.Timestamp(datetime.strptime(s, fmt))
                except (ValueError, TypeError):
                    continue
            try:
                return pd.to_datetime(s, dayfirst=dayfirst, errors='coerce')
            except Exception:
                return pd.NaT

        new_df[column] = new_df[column].apply(_safe_parse_val)
        
    failed_mask = original_notna & new_df[column].isna()
    failed_count = failed_mask.sum()
    
    return new_df, int(failed_count)


def normalize_booleans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Normalize common boolean representations (yes/no, y/n, true/false, 1/0, on/off) to True/False."""
    if column not in df.columns:
        return df.copy()
        
    new_df = df.copy()
    
    true_vals = {'yes', 'y', 'true', 't', '1', 'on', 'active'}
    false_vals = {'no', 'n', 'false', 'f', '0', 'off', 'inactive'}
    
    def to_bool(val):
        if pd.isna(val):
            return val
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        if s in true_vals:
            return True
        if s in false_vals:
            return False
        return val # Keep original if not recognized
        
    new_df[column] = new_df[column].apply(to_bool)
    return new_df

def standardize_text_encoding(text: str) -> str:
    """Fix common encoding artifacts like â€™ -> ', etc."""
    if not isinstance(text, str):
        return text
    
    replacements = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€˜': "'",
        'â€”': '-',
        'â€“': '-',
        'Ã©': 'é',
        'Ã¨': 'è',
        'Ã ': 'à',
        'Ã¢': 'â',
        'Ãª': 'ê',
        'Ã®': 'î',
        'Ã´': 'ô',
        'Ã»': 'û',
        'Ã¼': 'ü',
        'Ã¶': 'ö',
        'Ã¤': 'ä',
        'Ã±': 'ñ',
        'Ã§': 'ç'
    }
    
    result = text
    for bad, good in replacements.items():
        result = result.replace(bad, good)
    return result

# === Restructuring Operations ===

def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
    """Rename a single column."""
    return df.rename(columns={old_name: new_name})

def rename_columns_bulk(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename multiple columns using a mapping."""
    return df.rename(columns=mapping)

def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove specified columns."""
    cols_to_drop = [c for c in columns if c in df.columns]
    return df.drop(columns=cols_to_drop)

def reorder_columns(df: pd.DataFrame, new_order: list[str]) -> pd.DataFrame:
    """Reorder columns. Columns not in new_order are appended at the end."""
    ordered = [c for c in new_order if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]

def split_column(df: pd.DataFrame, column: str, delimiter: str, new_column_names: list[str] | None = None, max_splits: int = -1) -> pd.DataFrame:
    """Split a column by delimiter into multiple columns. 
    If new_column_names provided, use them (must match number of resulting columns).
    Otherwise auto-name as column_1, column_2, etc.
    Original column is kept."""
    if column not in df.columns:
        return df.copy()
        
    new_df = df.copy()
    n_splits = max_splits if max_splits > 0 else -1
    
    split_df = new_df[column].astype(str).str.split(delimiter, n=n_splits, expand=True)
    
    na_mask = new_df[column].isna()
    for col in split_df.columns:
        split_df.loc[na_mask, col] = np.nan
        
    if new_column_names and len(new_column_names) == len(split_df.columns):
        split_df.columns = new_column_names
    else:
        split_df.columns = [f"{column}_{i+1}" for i in range(len(split_df.columns))]
        
    return pd.concat([new_df, split_df], axis=1)

def merge_columns(df: pd.DataFrame, columns: list[str], new_column_name: str, separator: str = ' ', drop_originals: bool = False) -> pd.DataFrame:
    """Merge multiple columns into one, joining with separator. Handle NaN gracefully."""
    new_df = df.copy()
    valid_cols = [c for c in columns if c in new_df.columns]
    
    if not valid_cols:
        return new_df
        
    def join_row(row):
        vals = [str(row[c]) for c in valid_cols if pd.notna(row[c])]
        return separator.join(vals) if vals else np.nan
        
    new_df[new_column_name] = new_df.apply(join_row, axis=1)
    
    if drop_originals:
        new_df = new_df.drop(columns=valid_cols)
        
    return new_df

def melt_wide_to_long(df: pd.DataFrame, id_columns: list[str], value_columns: list[str], var_name: str = 'variable', value_name: str = 'value') -> pd.DataFrame:
    """Convert from wide to long format using pd.melt."""
    valid_ids = [c for c in id_columns if c in df.columns]
    valid_vals = [c for c in value_columns if c in df.columns]
    return pd.melt(df, id_vars=valid_ids, value_vars=valid_vals, var_name=var_name, value_name=value_name)

def pivot_long_to_wide(df: pd.DataFrame, index_col: str, columns_col: str, values_col: str) -> pd.DataFrame:
    """Convert from long to wide format using pd.pivot_table."""
    if not all(c in df.columns for c in [index_col, columns_col, values_col]):
        return df.copy()
    
    try:
        res = df.pivot(index=index_col, columns=columns_col, values=values_col).reset_index()
    except ValueError:
        res = df.pivot_table(index=index_col, columns=columns_col, values=values_col, aggfunc='first').reset_index()
    
    res.columns.name = None
    return res

# === Anomaly Detection ===

def detect_numeric_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """Detect numeric outliers using IQR or z-score method.
    Returns DataFrame of outlier rows with the outlier values.
    method='iqr': values below Q1 - threshold*IQR or above Q3 + threshold*IQR
    method='zscore': absolute z-score > threshold (default 3.0 for zscore)"""
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return pd.DataFrame(columns=df.columns)
        
    series = df[column].dropna()
    if series.empty:
        return pd.DataFrame(columns=df.columns)
        
    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'zscore':
        if threshold == 1.5:
            threshold = 3.0
        mean = series.mean()
        std = series.std()
        if std == 0:
            outliers_mask = pd.Series(False, index=df.index)
        else:
            z_scores = np.abs((df[column] - mean) / std)
            outliers_mask = z_scores > threshold
    else:
        outliers_mask = pd.Series(False, index=df.index)
        
    return df[outliers_mask].copy()

def detect_invalid_dates(df: pd.DataFrame, column: str) -> pd.Series:
    """Return boolean mask of rows where the column value cannot be parsed as a date."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
        
    def is_invalid(val):
        if pd.isna(val):
            return False
        if isinstance(val, (datetime, pd.Timestamp)):
            return False
        try:
            pd.to_datetime(val)
            return False
        except (ValueError, TypeError):
            return True
            
    return df[column].apply(is_invalid)

def detect_type_mismatches(df: pd.DataFrame, column: str, expected_type: str) -> pd.Series:
    """Detect values that don't match expected type ('numeric', 'date', 'boolean', 'email', 'phone').
    Returns boolean mask of mismatched rows."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
        
    def check_type(val, exp_type):
        if pd.isna(val):
            return False
            
        s = str(val).strip()
        if exp_type == 'numeric':
            try:
                float(re.sub(r'[$,\s€£¥%]', '', s))
                return False
            except ValueError:
                return True
        elif exp_type == 'date':
            if isinstance(val, (datetime, pd.Timestamp)):
                return False
            try:
                pd.to_datetime(val)
                return False
            except (ValueError, TypeError):
                return True
        elif exp_type == 'boolean':
            if isinstance(val, bool):
                return False
            if s.lower() in {'yes', 'no', 'y', 'n', 'true', 'false', 't', 'f', '1', '0', 'on', 'off'}:
                return False
            return True
        elif exp_type == 'email':
            return not bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", s))
        elif exp_type == 'phone':
            clean_s = re.sub(r'[\+\(\)\-\s]', '', s)
            return not clean_s.isdigit() or len(clean_s) < 7
        return False
        
    return df[column].apply(lambda x: check_type(x, expected_type))

def detect_suspicious_values(df: pd.DataFrame, column: str) -> list[dict]:
    """Detect suspicious values in a column:
    - Values that appear only once in a categorical column with many repeated values
    - Extreme string lengths
    - Embedded special characters
    Returns list of {value, reason, count, row_indices}."""
    if column not in df.columns:
        return []
        
    suspicious = []
    series = df[column].dropna()
    
    if series.empty:
        return suspicious
        
    value_counts = series.value_counts()
    
    # 1. Singletons in categorical
    if len(value_counts) > 1 and len(value_counts) < len(series) * 0.5:
        singletons = value_counts[value_counts == 1].index
        for val in singletons:
            indices = series[series == val].index.tolist()
            suspicious.append({
                'value': val,
                'reason': 'Singleton in categorical column',
                'count': 1,
                'row_indices': indices
            })
            
    # 2. Extreme string lengths
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        str_series = series.astype(str)
        lengths = str_series.str.len()
        if not lengths.empty:
            mean_len = lengths.mean()
            std_len = lengths.std()
            if std_len > 0:
                extreme_mask = np.abs(lengths - mean_len) > 3 * std_len
                extreme_vals = str_series[extreme_mask].unique()
                for val in extreme_vals:
                    if not any(s['value'] == val for s in suspicious):
                        indices = series[str_series == val].index.tolist()
                        suspicious.append({
                            'value': val,
                            'reason': 'Extreme string length',
                            'count': len(indices),
                            'row_indices': indices
                        })
                        
    # 3. Embedded special characters
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        str_series = series.astype(str)
        def has_special(s):
            return bool(re.search(r'[^\x00-\x7F]+', s))
            
        special_mask = str_series.apply(has_special)
        special_vals = str_series[special_mask].unique()
        for val in special_vals:
            if not any(s['value'] == val for s in suspicious):
                indices = series[str_series == val].index.tolist()
                suspicious.append({
                    'value': val,
                    'reason': 'Contains special/non-ASCII characters',
                    'count': len(indices),
                    'row_indices': indices
                })
                
    return suspicious

# === Utility Functions ===

def get_operation_impact(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """Compare two DataFrames and return impact metrics:
    rows_before, rows_after, rows_removed, rows_added, cols_before, cols_after,
    missing_before, missing_after, missing_change, duplicate_rows_before, duplicate_rows_after."""
    
    rows_b = len(df_before)
    rows_a = len(df_after)
    
    cols_b = len(df_before.columns)
    cols_a = len(df_after.columns)
    
    miss_b = df_before.isna().sum().sum()
    miss_a = df_after.isna().sum().sum()
    
    dup_b = df_before.duplicated().sum()
    dup_a = df_after.duplicated().sum()
    
    return {
        'rows_before': int(rows_b),
        'rows_after': int(rows_a),
        'rows_removed': int(max(0, rows_b - rows_a)),
        'rows_added': int(max(0, rows_a - rows_b)),
        'cols_before': int(cols_b),
        'cols_after': int(cols_a),
        'missing_before': int(miss_b),
        'missing_after': int(miss_a),
        'missing_change': int(miss_a - miss_b),
        'duplicate_rows_before': int(dup_b),
        'duplicate_rows_after': int(dup_a)
    }

def preview_operation(df: pd.DataFrame, operation: str, params: dict, max_preview_rows: int = 20) -> dict:
    """Preview the result of an operation without modifying the original DataFrame.
    Returns: {preview_df, impact_metrics, warnings}."""
    func = globals().get(operation)
    warnings = []
    
    if not func:
        return {
            'preview_df': pd.DataFrame(),
            'impact_metrics': {},
            'warnings': [f"Operation '{operation}' not found."]
        }
        
    try:
        result = func(df, **params)
        
        if isinstance(result, tuple):
            df_after = result[0]
            warnings.append(f"Operation returned additional info: {result[1:]}")
        else:
            df_after = result
            
        impact = get_operation_impact(df, df_after)
        
        preview = df_after.head(max_preview_rows).copy()
        
        return {
            'preview_df': preview,
            'impact_metrics': impact,
            'warnings': warnings
        }
    except Exception as e:
        return {
            'preview_df': pd.DataFrame(),
            'impact_metrics': {},
            'warnings': [f"Error during operation '{operation}': {str(e)}"]
        }
