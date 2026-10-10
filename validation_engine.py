import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
import uuid

@dataclass
class ValidationWarning:
    warning_id: str
    severity: str  # 'info', 'warning', 'critical'
    category: str  # 'row_loss', 'column_loss', 'missing_increase', 'type_change', 'identifier_damage', 'total_change', 'duplicate_keys', 'schema_change'
    message: str
    details: dict = field(default_factory=dict)
    affected_columns: list = field(default_factory=list)
    affected_rows: int = 0

class ValidationEngine:
    """Validates cleaning operations to prevent data damage."""
    
    def __init__(self):
        self.warnings: list[ValidationWarning] = []
    
    def validate_transformation(self, df_before: pd.DataFrame, df_after: pd.DataFrame,
                                  operation_name: str, params: dict) -> list[ValidationWarning]:
        """Run ALL validation checks and return warnings.
        Calls each specific check method below."""
        self.warnings = []
        
        self.warnings.extend(self.check_row_loss(df_before, df_after, operation_name))
        self.warnings.extend(self.check_column_loss(df_before, df_after))
        self.warnings.extend(self.check_missing_increase(df_before, df_after))
        self.warnings.extend(self.check_identifier_damage(df_before, df_after))
        self.warnings.extend(self.check_numeric_totals(df_before, df_after))
        
        if 'key_columns' in params:
             self.warnings.extend(self.check_duplicate_keys(df_before, df_after, params['key_columns']))
             
        self.warnings.extend(self.check_type_changes(df_before, df_after))
        
        return self.warnings
    
    def check_row_loss(self, df_before: pd.DataFrame, df_after: pd.DataFrame, 
                        operation_name: str) -> list[ValidationWarning]:
        """Warn if >10% rows lost (warning), >50% rows lost (critical)."""
        warnings = []
        rows_before = len(df_before)
        rows_after = len(df_after)
        if rows_before == 0:
            return warnings
            
        loss_ratio = (rows_before - rows_after) / rows_before
        
        if loss_ratio > 0.5:
            warnings.append(ValidationWarning(
                warning_id=str(uuid.uuid4()),
                severity='critical',
                category='row_loss',
                message=f"Critical row loss: {loss_ratio*100:.1f}% of rows were removed by {operation_name}.",
                details={'rows_before': rows_before, 'rows_after': rows_after, 'loss_ratio': loss_ratio},
                affected_rows=rows_before - rows_after
            ))
        elif loss_ratio > 0.1:
            warnings.append(ValidationWarning(
                warning_id=str(uuid.uuid4()),
                severity='warning',
                category='row_loss',
                message=f"Significant row loss: {loss_ratio*100:.1f}% of rows were removed by {operation_name}.",
                details={'rows_before': rows_before, 'rows_after': rows_after, 'loss_ratio': loss_ratio},
                affected_rows=rows_before - rows_after
            ))
            
        return warnings
    
    def check_column_loss(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> list[ValidationWarning]:
        """Warn about any dropped columns."""
        warnings = []
        cols_before = set(df_before.columns)
        cols_after = set(df_after.columns)
        
        dropped_cols = list(cols_before - cols_after)
        if dropped_cols:
            warnings.append(ValidationWarning(
                warning_id=str(uuid.uuid4()),
                severity='warning',
                category='column_loss',
                message=f"Columns were dropped: {', '.join(dropped_cols)}",
                details={'dropped_columns': dropped_cols},
                affected_columns=dropped_cols
            ))
        return warnings
    
    def check_missing_increase(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> list[ValidationWarning]:
        """Warn if missing values increased in any column after transformation."""
        warnings = []
        common_cols = list(set(df_before.columns) & set(df_after.columns))
        
        missing_before = df_before[common_cols].isna().sum()
        missing_after = df_after[common_cols].isna().sum()
        
        increased_missing_cols = []
        total_missing_increase = 0
        
        for col in common_cols:
            if missing_after[col] > missing_before[col]:
                increased_missing_cols.append(col)
                total_missing_increase += (missing_after[col] - missing_before[col])
                
        if increased_missing_cols:
            warnings.append(ValidationWarning(
                warning_id=str(uuid.uuid4()),
                severity='warning',
                category='missing_increase',
                message=f"Missing values increased in columns: {', '.join(increased_missing_cols)}",
                details={'columns': increased_missing_cols},
                affected_columns=increased_missing_cols,
                affected_rows=int(total_missing_increase)
            ))
            
        return warnings
    
    def check_identifier_damage(self, df_before: pd.DataFrame, df_after: pd.DataFrame,
                                  columns: list[str] | None = None) -> list[ValidationWarning]:
        """Check if leading zeros were stripped from potential ID columns.
        Auto-detect ID columns by name pattern (id, code, sku, zip, postal, phone)."""
        warnings = []
        if columns is None:
            id_patterns = ['id', 'code', 'sku', 'zip', 'postal', 'phone']
            columns = [c for c in df_before.columns if any(p in str(c).lower() for p in id_patterns)]
            
        common_cols = list(set(columns) & set(df_after.columns))
        damaged_cols = []
        
        for col in common_cols:
            before_is_text = pd.api.types.is_object_dtype(df_before[col]) or pd.api.types.is_string_dtype(df_before[col]) or not pd.api.types.is_numeric_dtype(df_before[col])
            after_is_num = pd.api.types.is_numeric_dtype(df_after[col])
            if before_is_text and after_is_num:
                has_leading_zeros = df_before[col].astype(str).str.startswith('0').any()
                if has_leading_zeros:
                    damaged_cols.append(col)
                    
        if damaged_cols:
            warnings.append(ValidationWarning(
                warning_id=str(uuid.uuid4()),
                severity='critical',
                category='identifier_damage',
                message=f"Possible identifier damage (leading zeros stripped) in columns: {', '.join(damaged_cols)}",
                details={'columns': damaged_cols},
                affected_columns=damaged_cols
            ))
            
        return warnings
    
    def check_numeric_totals(self, df_before: pd.DataFrame, df_after: pd.DataFrame,
                              tolerance: float = 0.01) -> list[ValidationWarning]:
        """Warn if sum of any numeric column changed by more than tolerance fraction."""
        warnings = []
        num_cols_before = df_before.select_dtypes(include=[np.number]).columns
        num_cols_after = df_after.select_dtypes(include=[np.number]).columns
        common_num_cols = list(set(num_cols_before) & set(num_cols_after))
        
        changed_cols = []
        for col in common_num_cols:
            sum_before = df_before[col].sum()
            sum_after = df_after[col].sum()
            
            if pd.isna(sum_before) or pd.isna(sum_after):
                continue
                
            if sum_before != 0:
                change_ratio = abs((sum_after - sum_before) / sum_before)
                if change_ratio > tolerance:
                    changed_cols.append(col)
            elif sum_after != 0:
                changed_cols.append(col)
                
        if changed_cols:
            warnings.append(ValidationWarning(
                warning_id=str(uuid.uuid4()),
                severity='warning',
                category='total_change',
                message=f"Numeric totals changed significantly in columns: {', '.join(changed_cols)}",
                details={'columns': changed_cols, 'tolerance': tolerance},
                affected_columns=changed_cols
            ))
            
        return warnings
    
    def check_duplicate_keys(self, df_before: pd.DataFrame, df_after: pd.DataFrame,
                              key_columns: list[str] | None = None) -> list[ValidationWarning]:
        """Check if transformation introduced duplicate values in key columns."""
        warnings = []
        if not key_columns:
            return warnings
            
        missing_keys = [k for k in key_columns if k not in df_after.columns]
        if missing_keys:
            warnings.append(ValidationWarning(
                warning_id=str(uuid.uuid4()),
                severity='critical',
                category='column_loss',
                message=f"Key columns missing after transformation: {', '.join(missing_keys)}",
                affected_columns=missing_keys
            ))
            return warnings
            
        dupes_before = df_before.duplicated(subset=key_columns).sum()
        dupes_after = df_after.duplicated(subset=key_columns).sum()
        
        if dupes_after > dupes_before:
            warnings.append(ValidationWarning(
                warning_id=str(uuid.uuid4()),
                severity='critical',
                category='duplicate_keys',
                message=f"Transformation introduced {dupes_after - dupes_before} new duplicate keys in columns: {', '.join(key_columns)}",
                details={'dupes_before': int(dupes_before), 'dupes_after': int(dupes_after)},
                affected_columns=key_columns,
                affected_rows=int(dupes_after - dupes_before)
            ))
            
        return warnings
    
    def check_type_changes(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> list[ValidationWarning]:
        """Warn about unexpected dtype changes."""
        warnings = []
        common_cols = list(set(df_before.columns) & set(df_after.columns))
        changed_types = {}
        
        for col in common_cols:
            type_before = str(df_before[col].dtype)
            type_after = str(df_after[col].dtype)
            if type_before != type_after:
                changed_types[col] = f"{type_before} -> {type_after}"
                
        if changed_types:
            warnings.append(ValidationWarning(
                warning_id=str(uuid.uuid4()),
                severity='info',
                category='type_change',
                message=f"Data types changed for columns: {', '.join(changed_types.keys())}",
                details={'type_changes': changed_types},
                affected_columns=list(changed_types.keys())
            ))
            
        return warnings
    
    def check_schema_compatibility(self, df: pd.DataFrame, expected_schema: dict) -> list[ValidationWarning]:
        """Check if DataFrame matches expected schema (column names, types)."""
        warnings = []
        for col, expected_type in expected_schema.items():
            if col not in df.columns:
                warnings.append(ValidationWarning(
                    warning_id=str(uuid.uuid4()),
                    severity='critical',
                    category='schema_change',
                    message=f"Missing expected column: {col}",
                    affected_columns=[col]
                ))
            else:
                actual_type = str(df[col].dtype)
                if expected_type != actual_type:
                    warnings.append(ValidationWarning(
                        warning_id=str(uuid.uuid4()),
                        severity='warning',
                        category='schema_change',
                        message=f"Type mismatch for {col}: expected {expected_type}, got {actual_type}",
                        affected_columns=[col]
                    ))
        return warnings
    
    def get_validation_summary(self) -> dict:
        """Return summary: total_warnings, by_severity, by_category, critical_count."""
        by_severity = {}
        by_category = {}
        critical_count = 0
        
        for w in self.warnings:
            by_severity[w.severity] = by_severity.get(w.severity, 0) + 1
            by_category[w.category] = by_category.get(w.category, 0) + 1
            if w.severity == 'critical':
                critical_count += 1
                
        return {
            'total_warnings': len(self.warnings),
            'by_severity': by_severity,
            'by_category': by_category,
            'critical_count': critical_count
        }
    
    def is_safe_to_apply(self) -> bool:
        """Returns False if any critical warnings exist."""
        return not any(w.severity == 'critical' for w in self.warnings)
