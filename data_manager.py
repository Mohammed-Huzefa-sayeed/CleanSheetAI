import pandas as pd
import numpy as np
import chardet
import io
import os
import uuid
import datetime
from typing import Optional, Dict, Any, List, Tuple

class ChangeRecord:
    """Records a single cleaning operation for undo/history."""
    def __init__(self, operation_name: str, parameters: dict, affected_columns: list, 
                 affected_rows_count: int, before_metrics: dict, after_metrics: dict, 
                 validation_warnings: list, accepted: bool, df_before_snapshot_hash: str, 
                 reversible: bool = True):
        self.operation_id = str(uuid.uuid4())
        self.operation_name = operation_name
        self.parameters = parameters
        self.affected_columns = affected_columns
        self.affected_rows_count = affected_rows_count
        self.timestamp = datetime.datetime.now().isoformat()
        self.before_metrics = before_metrics
        self.after_metrics = after_metrics
        self.validation_warnings = validation_warnings
        self.accepted = accepted
        self.df_before_snapshot_hash = df_before_snapshot_hash
        self.reversible = reversible

class DatasetInfo:
    """Stores metadata about a loaded dataset."""
    def __init__(self, filename: str, original_df: pd.DataFrame, sheet_name: Optional[str] = None):
        self.filename = filename
        self.original_df = original_df.copy(deep=True)
        self.current_df = original_df.copy(deep=True)
        self.sheet_name = sheet_name
        self.row_count = len(original_df)
        self.col_count = len(original_df.columns)
        self.memory_usage = original_df.memory_usage(deep=True).sum()
        self.column_types = original_df.dtypes.astype(str).to_dict()
        self.missing_stats = original_df.isnull().sum().to_dict()
        self.load_timestamp = datetime.datetime.now().isoformat()
        self.change_history: List[ChangeRecord] = []
        # Store snapshots for undo
        self.snapshots: Dict[str, pd.DataFrame] = {}

def get_sheet_names(uploaded_file, filename: str) -> list[str]:
    """Return list of sheet names for Excel files."""
    if filename.endswith(('.xlsx', '.xls')):
        xls = pd.ExcelFile(uploaded_file)
        return xls.sheet_names
    return []

def load_file(uploaded_file, filename: str, sheet_name: Optional[str] = None) -> DatasetInfo:
    """Load CSV or XLSX file."""
    if filename.endswith('.csv'):
        raw_data = uploaded_file.read()
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
        
        # Try UTF-8 first
        try:
            df = pd.read_csv(io.BytesIO(raw_data), encoding='utf-8', dtype=str)
        except UnicodeDecodeError:
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            try:
                df = pd.read_csv(io.BytesIO(raw_data), encoding=encoding, dtype=str)
            except Exception:
                # Try multiple separators if needed
                df = pd.read_csv(io.BytesIO(raw_data), encoding=encoding, sep=None, engine='python', dtype=str)
    elif filename.endswith(('.xlsx', '.xls')):
        if sheet_name:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name, dtype=str)
        else:
            df = pd.read_excel(uploaded_file, dtype=str)
    else:
        raise ValueError("Unsupported file extension.")
    
    # Validation checks
    if df.empty:
        raise ValueError("The uploaded file is empty.")
    
    if len(df.columns) == 0:
        raise ValueError("No columns found in the file.")
        
    if df.columns.duplicated().any():
        df.columns = [f"{c}_{i}" if c in df.columns[:i] else c for i, c in enumerate(df.columns)]
        
    return DatasetInfo(filename=filename, original_df=df, sheet_name=sheet_name)


def get_column_profile(df: pd.DataFrame, column: str) -> dict:
    """Profile a single column."""
    col = df[column]
    null_count = int(col.isnull().sum())
    total_count = len(col)
    
    profile = {
        'dtype': str(col.dtype),
        'unique_count': int(col.nunique(dropna=True)),
        'null_count': null_count,
        'null_pct': (null_count / total_count * 100) if total_count > 0 else 0.0,
        'is_constant': col.nunique(dropna=True) <= 1,
        'sample_values': col.dropna().head(5).tolist()
    }
    
    if pd.api.types.is_numeric_dtype(col):
        profile.update({
            'min': col.min() if not col.empty else None,
            'max': col.max() if not col.empty else None,
            'mean': col.mean() if not col.empty else None,
            'median': col.median() if not col.empty else None,
            'looks_like_numeric': True
        })
    else:
        profile['looks_like_numeric'] = pd.to_numeric(col.dropna(), errors='coerce').notnull().all() if not col.dropna().empty else False
        profile['looks_like_date'] = False 
        try:
            if not col.dropna().empty and pd.to_datetime(col.dropna().head(10), errors='coerce').notnull().any():
                profile['looks_like_date'] = True
        except Exception:
            pass
            
        profile['has_leading_zeros'] = col.dropna().astype(str).str.match(r'^0\d+').any() if not col.empty else False

    return profile

def get_dataset_profile(df: pd.DataFrame) -> dict:
    """Full dataset profile."""
    total_cells = df.size
    total_missing = int(df.isnull().sum().sum())
    
    return {
        'row_count': len(df),
        'col_count': len(df.columns),
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'total_missing': total_missing,
        'total_missing_pct': (total_missing / total_cells * 100) if total_cells > 0 else 0.0,
        'duplicate_rows': int(df.duplicated().sum()),
        'constant_columns': [col for col in df.columns if df[col].nunique(dropna=True) <= 1],
        'empty_rows': int(df.isnull().all(axis=1).sum()),
        'empty_columns': df.columns[df.isnull().all()].tolist(),
        'column_profiles': {col: get_column_profile(df, col) for col in df.columns}
    }

def _is_string_col(series: pd.Series) -> bool:
    """Return True if the series holds text data.
    Handles both legacy object dtype and the pandas StringDtype extension type
    that is created when files are loaded with dtype=str.
    """
    return (
        series.dtype == 'object'
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.StringDtype)
    )


def detect_issues(df: pd.DataFrame) -> list[dict]:
    """Detect all data quality issues."""
    issues = []
    
    # missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        total_missing = int(df[missing_cols].isnull().sum().sum())
        issues.append({
            'issue_id': 'missing_values',
            'issue_type': 'missing_values',
            'severity': 'high',
            'affected_columns': missing_cols,
            'affected_rows': int(df.isnull().any(axis=1).sum()),
            'description': f"Missing values found in {len(missing_cols)} column(s) ({total_missing} total missing cells).",
            'suggested_action': 'fill_missing'
        })
        
    # duplicate rows
    dups = int(df.duplicated().sum())
    if dups > 0:
        issues.append({
            'issue_id': 'duplicate_rows',
            'issue_type': 'duplicate_rows',
            'severity': 'medium',
            'affected_columns': df.columns.tolist(),
            'affected_rows': dups,
            'description': f"Found {dups} exact duplicate row(s).",
            'suggested_action': 'remove_duplicates'
        })
        
    # empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        issues.append({
            'issue_id': 'empty_column',
            'issue_type': 'empty_column',
            'severity': 'low',
            'affected_columns': empty_cols,
            'affected_rows': len(df),
            'description': f"Found {len(empty_cols)} completely empty column(s): {', '.join(empty_cols)}.",
            'suggested_action': 'drop_column'
        })

    # constant columns
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1 and len(df[col].dropna()) > 0]
    if constant_cols:
        issues.append({
            'issue_id': 'constant_column',
            'issue_type': 'constant_column',
            'severity': 'low',
            'affected_columns': constant_cols,
            'affected_rows': len(df),
            'description': f"Found {len(constant_cols)} column(s) with constant values: {', '.join(constant_cols)}.",
            'suggested_action': 'drop_column'
        })

    # empty rows
    empty_rows_count = int(df.isnull().all(axis=1).sum())
    if empty_rows_count > 0:
        issues.append({
            'issue_id': 'empty_rows',
            'issue_type': 'empty_rows',
            'severity': 'medium',
            'affected_columns': df.columns.tolist(),
            'affected_rows': empty_rows_count,
            'description': f"Found {empty_rows_count} completely empty row(s).",
            'suggested_action': 'drop_rows_with_missing'
        })

    # whitespace issues
    whitespace_cols = []
    for col in df.columns:
        if _is_string_col(df[col]):
            has_ws = df[col].dropna().astype(str).str.contains(r'^\s+|\s+$|\s{2,}', regex=True)
            if has_ws.any():
                whitespace_cols.append(col)
    if whitespace_cols:
        issues.append({
            'issue_id': 'whitespace_issues',
            'issue_type': 'whitespace_issues',
            'severity': 'low',
            'affected_columns': whitespace_cols,
            'affected_rows': 0,
            'description': f"Leading/trailing whitespace or multiple spaces found in {len(whitespace_cols)} column(s).",
            'suggested_action': 'trim_whitespace'
        })

    # inconsistent case
    for col in df.columns:
        if _is_string_col(df[col]):
            vals = df[col].dropna().astype(str)
            if len(vals) > 0:
                unique_vals = vals.unique()
                lower_vals = set(v.lower().strip() for v in unique_vals)
                if len(lower_vals) < len(unique_vals) and len(unique_vals) <= 50:
                    issues.append({
                        'issue_id': f'inconsistent_case_{col}',
                        'issue_type': 'inconsistent_case',
                        'severity': 'low',
                        'affected_columns': [col],
                        'affected_rows': len(vals),
                        'description': f"Column '{col}' has inconsistent casing (e.g., mixed upper/lower).",
                        'suggested_action': 'normalize_case'
                    })

    # inconsistent categories
    for col in df.columns:
        if _is_string_col(df[col]):
            vals = df[col].dropna().astype(str)
            if len(vals) > 0 and vals.nunique() <= 30:
                unique_vals = vals.unique()
                normalized = [v.lower().strip() for v in unique_vals]
                if len(set(normalized)) < len(unique_vals):
                    issues.append({
                        'issue_id': f'inconsistent_categories_{col}',
                        'issue_type': 'inconsistent_categories',
                        'severity': 'medium',
                        'affected_columns': [col],
                        'affected_rows': len(vals),
                        'description': f"Column '{col}' has inconsistent category labels that likely represent the same value.",
                        'suggested_action': 'normalize_categories'
                    })

    # leading zeros risk
    for col in df.columns:
        if _is_string_col(df[col]):
            vals = df[col].dropna().astype(str)
            has_leading_zeros = vals.str.match(r'^0\d+').any()
            col_lower = col.lower()
            is_id_col = any(kw in col_lower for kw in ['id', 'code', 'sku', 'zip', 'postal', 'phone', 'fax'])
            if has_leading_zeros and is_id_col:
                issues.append({
                    'issue_id': f'leading_zeros_risk_{col}',
                    'issue_type': 'leading_zeros_risk',
                    'severity': 'high',
                    'affected_columns': [col],
                    'affected_rows': int(vals.str.match(r'^0\d+').sum()),
                    'description': f"Column '{col}' contains values with leading zeros (possible identifiers). Converting to numeric would lose these.",
                    'suggested_action': 'preserve_as_text'
                })

    # mixed types / invalid non-numeric values in numeric-like columns
    # A column is considered "numeric-like" when it is stored as object dtype but most
    # values can be parsed as numbers.  We no longer gate on len(vals) > 5 so that
    # small datasets (like the 7-row test file) are also inspected.
    for col in df.columns:
        if _is_string_col(df[col]):
            vals = df[col].dropna().astype(str)
            if len(vals) == 0:
                continue
            numeric_mask = pd.to_numeric(vals, errors='coerce').notna()
            pct_numeric = numeric_mask.sum() / len(vals)

            # Case 1: mostly numeric column with some text "poison" values
            # (e.g. 'unknown', 'N/A', 'none' in an Age column)
            if pct_numeric >= 0.5 and pct_numeric < 1.0:
                invalid_vals = vals[~numeric_mask].unique().tolist()
                issues.append({
                    'issue_id': f'invalid_values_{col}',
                    'issue_type': 'invalid_values',
                    'severity': 'high',
                    'affected_columns': [col],
                    'affected_rows': int((~numeric_mask).sum()),
                    'description': (
                        f"Column '{col}' appears numeric but contains {int((~numeric_mask).sum())} "
                        f"non-numeric value(s): {invalid_vals}. These will become NaN if converted."
                    ),
                    'suggested_action': 'review_and_convert_numeric',
                    'invalid_values': invalid_vals,
                })

            # Case 2: mixed types with significant non-numeric portion (classic mixed_types)
            elif 0.1 < pct_numeric < 0.5:
                issues.append({
                    'issue_id': f'mixed_types_{col}',
                    'issue_type': 'mixed_types',
                    'severity': 'medium',
                    'affected_columns': [col],
                    'affected_rows': int((~numeric_mask).sum()),
                    'description': f"Column '{col}' has mixed types: {pct_numeric:.0%} numeric and {1-pct_numeric:.0%} text values.",
                    'suggested_action': 'convert_to_numeric',
                })

    # suspicious / out-of-range outliers in numeric-like columns
    # Detect statistical and domain-knowledge outliers (e.g. Age > 120 or Age < 0).
    DOMAIN_BOUNDS: dict[str, tuple[float, float]] = {
        'age': (0, 120),
        'quantity': (0, 1e9),
    }
    for col in df.columns:
        col_series = df[col]
        # Work on the numeric-parseable values of object columns too
        if _is_string_col(col_series):
            col_series = pd.to_numeric(col_series, errors='coerce')
        if not pd.api.types.is_numeric_dtype(col_series):
            continue
        numeric_vals = col_series.dropna()
        if len(numeric_vals) < 2:
            continue

        outlier_rows = []
        reasons = []

        # Domain bounds check
        col_lower = col.lower()
        for key, (lo, hi) in DOMAIN_BOUNDS.items():
            if key in col_lower:
                out_of_range = numeric_vals[(numeric_vals < lo) | (numeric_vals > hi)]
                if not out_of_range.empty:
                    outlier_rows.extend(out_of_range.index.tolist())
                    for idx, val in out_of_range.items():
                        reasons.append(f"Row {idx}: {col}={val} (expected {lo}–{hi})")

        # IQR-based statistical outliers (only flag if domain bounds didn't already catch them)
        if not outlier_rows:
            q1, q3 = numeric_vals.quantile(0.25), numeric_vals.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lo_fence = q1 - 3.0 * iqr
                hi_fence = q3 + 3.0 * iqr
                stat_outliers = numeric_vals[(numeric_vals < lo_fence) | (numeric_vals > hi_fence)]
                if not stat_outliers.empty:
                    outlier_rows.extend(stat_outliers.index.tolist())
                    for idx, val in stat_outliers.items():
                        reasons.append(f"Row {idx}: {col}={val} (stat outlier)")

        if outlier_rows:
            issues.append({
                'issue_id': f'suspicious_outliers_{col}',
                'issue_type': 'suspicious_outliers',
                'severity': 'medium',
                'affected_columns': [col],
                'affected_rows': len(set(outlier_rows)),
                'description': (
                    f"Column '{col}' has {len(set(outlier_rows))} suspicious/out-of-range value(s). "
                    + "; ".join(reasons[:5])
                ),
                'suggested_action': 'review_outliers',
                'outlier_indices': list(set(outlier_rows)),
            })

    # mixed date formats in date-like columns
    DATE_PATTERNS = [
        (r'^\d{4}-\d{2}-\d{2}$', 'ISO (YYYY-MM-DD)'),
        (r'^\d{2}/\d{2}/\d{4}$', 'DD/MM/YYYY or MM/DD/YYYY'),
        (r'^\d{2}-\d{2}-\d{4}$', 'DD-MM-YYYY'),
        (r'^\d{2}\.\d{2}\.\d{4}$', 'DD.MM.YYYY'),
    ]
    for col in df.columns:
        if not _is_string_col(df[col]):
            continue
        vals = df[col].dropna().astype(str)
        if len(vals) == 0:
            continue
        formats_found = set()
        for pattern, label in DATE_PATTERNS:
            if vals.str.match(pattern).any():
                formats_found.add(label)
        if len(formats_found) > 1:
            issues.append({
                'issue_id': f'mixed_date_formats_{col}',
                'issue_type': 'mixed_date_formats',
                'severity': 'medium',
                'affected_columns': [col],
                'affected_rows': int(len(vals)),
                'description': (
                    f"Column '{col}' has mixed date formats: {', '.join(sorted(formats_found))}. "
                    f"Parsing may produce incorrect results."
                ),
                'suggested_action': 'parse_dates',
                'formats_found': list(formats_found),
            })

    return issues

def apply_operation(dataset_info: DatasetInfo, operation_name: str, params: dict) -> tuple[pd.DataFrame, ChangeRecord]:
    """Apply a cleaning operation."""
    df = dataset_info.current_df.copy(deep=True)
    affected_cols = []
    affected_rows = 0
    
    if operation_name == 'fill_missing':
        col = params.get('column')
        strategy = params.get('strategy')
        val = params.get('value', '')
        cols = params.get('columns', [col] if col else [])
        
        # Guard against filling sensitive / unique / transactional columns with mode or mean
        sensitive_keywords = (
            'email', 'phone', 'mobile', 'id', 'identifier', 'uid', 'uuid',
            'customer', 'ssn', 'username', 'login', 'total', 'amount', 'price',
            'revenue', 'cost', 'city', 'state', 'zip', 'postal', 'address', 'date'
        )
        for c in cols:
            if c and c in df.columns:
                c_lower = str(c).lower()
                is_sensitive = any(kw in c_lower for kw in sensitive_keywords)
                if is_sensitive and strategy in ('mode', 'mean'):
                    # Refuse to blindly copy another record's value into a sensitive column
                    continue
                affected_rows += int(df[c].isnull().sum())
                if strategy:
                    from cleaning_engine import fill_missing_with_strategy
                    df = fill_missing_with_strategy(df, c, strategy, fill_value=val)
                else:
                    df[c] = df[c].fillna(val)
                affected_cols.append(c)

    elif operation_name == 'remove_duplicates':
        keep = params.get('keep', 'first')
        cols = params.get('columns')
        affected_rows = int(df.duplicated(subset=cols).sum())
        df = df.drop_duplicates(subset=cols, keep=keep)
        affected_cols = cols if cols else df.columns.tolist()
    elif operation_name == 'drop_column':
        cols = params.get('columns', [])
        affected_rows = len(df)
        df = df.drop(columns=cols, errors='ignore')
        affected_cols = cols
    elif operation_name == 'trim_whitespace':
        from cleaning_engine import trim_whitespace as engine_trim
        cols = params.get('columns')
        affected_rows = len(df)
        df = engine_trim(df, columns=cols)
        affected_cols = cols if cols else df.columns.tolist()
    elif operation_name == 'convert_blanks_to_nan':
        from cleaning_engine import convert_blanks_to_nan as engine_blanks_to_nan
        cols = params.get('columns')
        df = engine_blanks_to_nan(df, columns=cols)
        affected_cols = cols if cols else df.columns.tolist()
        affected_rows = len(df)
    elif operation_name == 'drop_rows_with_missing':
        before = len(df)
        cols = params.get('columns')
        how = params.get('how', 'any')
        df = df.dropna(subset=cols, how=how)
        affected_rows = before - len(df)
        affected_cols = cols if cols else df.columns.tolist()
    elif operation_name == 'normalize_case':
        col = params.get('column')
        case = params.get('case', 'lower')
        if col and col in df.columns and _is_string_col(df[col]):
            if case == 'lower':
                df[col] = df[col].astype(str).str.lower()
            elif case == 'upper':
                df[col] = df[col].astype(str).str.upper()
            elif case == 'title':
                df[col] = df[col].astype(str).str.title()
            elif case == 'sentence':
                df[col] = df[col].astype(str).str.capitalize()
            affected_cols = [col]
            affected_rows = len(df)
    elif operation_name in ('convert_numeric', 'convert_to_numeric'):
        col = params.get('column')
        if col and col in df.columns:
            before_nulls = int(df[col].isnull().sum())
            df[col] = pd.to_numeric(df[col], errors='coerce')
            after_nulls = int(df[col].isnull().sum())
            affected_cols = [col]
            affected_rows = after_nulls - before_nulls
    elif operation_name == 'parse_dates':
        col = params.get('column')
        if col and col in df.columns:
            from cleaning_engine import parse_dates as engine_parse_dates
            dayfirst = params.get('dayfirst', False)
            date_fmt = params.get('format')
            df, failed_count = engine_parse_dates(df, col, date_format=date_fmt, dayfirst=dayfirst)
            affected_cols = [col]
            affected_rows = len(df) - int(failed_count)

    elif operation_name == 'rename_column':
        mapping = params.get('mapping', {})
        df = df.rename(columns=mapping)
        affected_cols = list(mapping.keys())
        affected_rows = 0
    elif operation_name == 'reorder_columns':
        cols = params.get('columns', df.columns.tolist())
        df = df[cols]
        affected_cols = cols
        affected_rows = 0
    elif operation_name == 'split_column':
        col = params.get('column')
        sep = params.get('separator', ',')
        if col and col in df.columns:
            new_cols = df[col].astype(str).str.split(sep, expand=True)
            new_col_names = [f"{col}_{i}" for i in range(new_cols.shape[1])]
            new_cols.columns = new_col_names
            df = pd.concat([df, new_cols], axis=1)
            affected_cols = [col]
            affected_rows = len(df)
    elif operation_name == 'merge_columns':
        cols = params.get('columns', [])
        new_col = params.get('new_column_name', 'merged')
        sep = params.get('separator', '_')
        if cols and all(c in df.columns for c in cols):
            df[new_col] = df[cols].astype(str).agg(sep.join, axis=1)
            affected_cols = cols
            affected_rows = len(df)
    elif operation_name == 'normalize_categories':
        col = params.get('column')
        mapping = params.get('mapping', {})
        if col and col in df.columns:
            if mapping:
                df[col] = df[col].replace(mapping)
            else:
                from cleaning_engine import normalize_categories as _eng_norm_cats
                df = _eng_norm_cats(df, col, mapping=None)
            affected_cols = [col]
            affected_rows = len(df)
    elif operation_name == 'fill_with_value':
        col = params.get('column')
        val = params.get('value')
        if col and col in df.columns:
            affected_rows = int(df[col].isnull().sum())
            df[col] = df[col].fillna(val)
            affected_cols = [col]
    elif operation_name == 'normalize_column_names':
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True)
        affected_cols = df.columns.tolist()
        affected_rows = 0
    elif operation_name in ('leave_unchanged', 'flag_for_review'):
        col = params.get('column', '')
        affected_cols = [col] if col else []
        affected_rows = 0
    hash_val = str(uuid.uuid4())
    record = ChangeRecord(
        operation_name=operation_name,
        parameters=params,
        affected_columns=affected_cols,
        affected_rows_count=affected_rows,
        before_metrics={},
        after_metrics={},
        validation_warnings=[],
        accepted=False,
        df_before_snapshot_hash=hash_val
    )
    dataset_info.snapshots[hash_val] = dataset_info.current_df.copy(deep=True)
    return df, record

def accept_change(dataset_info: DatasetInfo, new_df: pd.DataFrame, change_record: ChangeRecord) -> None:
    """Accept a change."""
    dataset_info.current_df = new_df.copy(deep=True)
    change_record.accepted = True
    dataset_info.change_history.append(change_record)
    dataset_info.row_count = len(new_df)
    dataset_info.col_count = len(new_df.columns)

def undo_last_change(dataset_info: DatasetInfo) -> bool:
    """Undo the most recent accepted change."""
    if not dataset_info.change_history:
        return False
    last_change = dataset_info.change_history.pop()
    if last_change.df_before_snapshot_hash in dataset_info.snapshots:
        dataset_info.current_df = dataset_info.snapshots[last_change.df_before_snapshot_hash].copy(deep=True)
        dataset_info.row_count = len(dataset_info.current_df)
        dataset_info.col_count = len(dataset_info.current_df.columns)
        return True
    return False

def compute_quality_score(df: pd.DataFrame, original_df: pd.DataFrame | None = None) -> dict:
    """Compute data quality score with real component calculations.

    Components
    ----------
    completeness  : fraction of non-missing cells
    uniqueness    : fraction of non-duplicate rows
    validity      : penalises invalid (non-parseable) values in numeric-like columns
                    and values outside known domain bounds (e.g. Age 145)
    consistency   : penalises mixed date formats and casing inconsistencies
    """
    total_cells = df.size
    if total_cells == 0:
        return {'overall_score': 0, 'component_scores': {}, 'improvements': {}}

    # --- completeness ---
    missing_ratio = df.isnull().sum().sum() / total_cells
    completeness = max(0.0, 1.0 - missing_ratio)

    # --- uniqueness ---
    dup_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
    uniqueness = max(0.0, 1.0 - dup_ratio)

    # --- validity ---
    # Count cells that contain invalid values (non-numeric in a numeric-like column,
    # or values outside known domain bounds).
    total_invalid_cells = 0
    DOMAIN_BOUNDS = {'age': (0, 120), 'quantity': (0, 1e9), 'total': (0, 1000000)}
    for col in df.columns:

        col_series = df[col]
        col_lower = col.lower()

        if _is_string_col(col_series):
            vals = col_series.dropna().astype(str)
            if len(vals) == 0:
                continue
            numeric_parsed = pd.to_numeric(vals, errors='coerce')
            pct_numeric = numeric_parsed.notna().sum() / len(vals)
            # Column is "should-be numeric" if ≥50% of values parse as numbers
            if pct_numeric >= 0.5:
                invalid_count = int(numeric_parsed.isna().sum())
                total_invalid_cells += invalid_count
                # Domain bounds on successfully-parsed values
                for key, (lo, hi) in DOMAIN_BOUNDS.items():
                    if key in col_lower:
                        out_of_range = numeric_parsed.dropna()
                        out_of_range = out_of_range[(out_of_range < lo) | (out_of_range > hi)]
                        total_invalid_cells += len(out_of_range)
        elif pd.api.types.is_numeric_dtype(col_series):
            for key, (lo, hi) in DOMAIN_BOUNDS.items():
                if key in col_lower:
                    vals_num = col_series.dropna()
                    total_invalid_cells += int(((vals_num < lo) | (vals_num > hi)).sum())

    invalid_ratio = total_invalid_cells / total_cells if total_cells > 0 else 0
    validity = max(0.0, 1.0 - invalid_ratio * 2)  # multiply by 2 to amplify penalty

    # --- consistency ---
    # Penalise mixed date formats and casing inconsistencies in text columns.
    consistency_penalties = 0
    DATE_PATTERNS = [
        r'^\d{4}-\d{2}-\d{2}$',
        r'^\d{2}/\d{2}/\d{4}$',
        r'^\d{2}-\d{2}-\d{4}$',
        r'^\d{2}\.\d{2}\.\d{4}$',
    ]
    for col in df.columns:
        if not _is_string_col(df[col]):
            continue
        vals = df[col].dropna().astype(str)
        if len(vals) == 0:
            continue
        # Mixed date formats
        formats_found = sum(1 for p in DATE_PATTERNS if vals.str.match(p).any())
        if formats_found > 1:
            consistency_penalties += 1
        # Casing inconsistency (same base-value exists in multiple cases)
        unique_vals = vals.unique()
        lower_vals = set(v.lower().strip() for v in unique_vals)
        if len(lower_vals) < len(unique_vals):
            consistency_penalties += 1

    # Normalise: each column can contribute at most 2 penalties; penalise proportionally
    max_penalties = len(df.columns) * 2
    consistency = max(0.0, 1.0 - consistency_penalties / max_penalties) if max_penalties > 0 else 1.0

    overall_score = (completeness * 0.3 + uniqueness * 0.25 + consistency * 0.25 + validity * 0.2) * 100

    result = {
        'overall_score': round(float(overall_score), 1),
        'component_scores': {
            'completeness': round(float(completeness), 4),
            'uniqueness': round(float(uniqueness), 4),
            'consistency': round(float(consistency), 4),
            'validity': round(float(validity), 4),
        },
        'improvements': {},
    }

    if original_df is not None:
        orig = compute_quality_score(original_df)
        result['improvements'] = {
            k: round(result['component_scores'].get(k, 0) - orig['component_scores'].get(k, 0), 4)
            for k in result['component_scores']
        }
        result['original_score'] = orig['overall_score']

    return result
