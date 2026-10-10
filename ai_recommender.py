import pandas as pd
import json
import os
import uuid
from typing import Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

try:
    import openai
except ImportError:
    openai = None

@dataclass
class CleaningRecommendation:
    issue_id: str
    issue_type: str
    affected_columns: list[str]
    affected_rows: int
    description: str
    suggested_operation: str
    operation_params: dict
    explanation: str
    confidence: float  # 0.0-1.0
    risk_level: str  # 'low', 'medium', 'high'
    requires_confirmation: bool
    source: str  # 'rule_based' or 'ai'
    priority: int  # 1 (highest) to 5 (lowest)

ALLOWED_OPERATIONS = [
    'trim_whitespace', 'normalize_case', 'fill_missing', 'drop_rows_with_missing',
    'remove_duplicates', 'convert_to_numeric', 'parse_dates', 'normalize_booleans',
    'normalize_categories', 'rename_column', 'drop_column', 'split_column',
    'merge_columns', 'normalize_column_names', 'convert_blanks_to_nan',
    'fill_with_value', 'detect_outliers', 'leave_unchanged', 'flag_for_review'
]


class AIRecommender:
    """AI-powered data cleaning recommendation engine."""
    
    def __init__(self, use_ai: bool = False, api_key: str | None = None):
        self.use_ai = use_ai and api_key is not None and openai is not None
        self.api_key = api_key
        if self.use_ai and self.api_key:
            openai.api_key = self.api_key
        self.recommendations: list[CleaningRecommendation] = []
    
    def analyze_dataset(self, df: pd.DataFrame, issues: list[dict]) -> list[CleaningRecommendation]:
        """Analyze dataset and generate cleaning recommendations.
        Uses AI if enabled and available, otherwise falls back to rule-based."""
        if self.use_ai:
            try:
                self.recommendations = self._generate_ai_recommendations(df, issues)
            except Exception as e:
                print(f"AI recommendation failed: {e}. Falling back to rule-based.")
                self.recommendations = self._generate_rule_based_recommendations(df, issues)
        else:
            self.recommendations = self._generate_rule_based_recommendations(df, issues)
            
        return self.recommendations
    
    # Keywords that identify columns that hold unique, transactional, geographic, or identifier values
    # that must NEVER have values copied from another record via mode/mean imputation.
    _UNIQUE_COL_KEYWORDS = (
        'email', 'e-mail', 'phone', 'mobile', 'tel', 'fax',
        'id', 'identifier', 'uid', 'uuid', 'customer_id', 'user_id', 'account',
        'ssn', 'sin', 'passport', 'licence', 'license', 'registration',
        'username', 'login', 'iban', 'credit_card', 'card_number',
        'url', 'website', 'ip', 'mac_address',
        'city', 'state', 'zip', 'postal', 'address', 'location',
        'total', 'amount', 'price', 'revenue', 'cost', 'balance', 'salary'
    )

    def _is_unique_identifier_column(self, col: str) -> bool:
        """Return True if the column name suggests it holds unique per-record,
        transactional, or geographic values (emails, phone numbers, IDs, totals, cities, etc.)
        that must NEVER be auto-filled from another row without explicit approval."""
        col_lower = col.lower()
        return any(kw in col_lower for kw in self._UNIQUE_COL_KEYWORDS)


    def _generate_rule_based_recommendations(self, df: pd.DataFrame, issues: list[dict]) -> list[CleaningRecommendation]:
        """Generate recommendations using rule-based logic.
        Map each detected issue type to specific cleaning operations with explanations.
        Provide confidence scores based on issue severity and clarity."""
        recs = []
        for issue in issues:
            issue_type = issue.get('issue_type') or issue.get('type', 'unknown')
            affected_cols = issue.get('affected_columns', [])
            if not affected_cols and 'column' in issue:
                affected_cols = [issue['column']]
            primary_col = affected_cols[0] if affected_cols else ''
            rows = issue.get('affected_rows', 0)
            desc = issue.get('description', '')
            
            if issue_type == 'missing_values':
                for col in (affected_cols or [primary_col]):
                    if not col:
                        continue
                    missing_cnt = int(df[col].isnull().sum()) if col in df.columns else rows

                    # ------------------------------------------------------------------
                    # CRITICAL: unique/contact columns (email, phone, ID, etc.) must
                    # NEVER be filled using mode or mean from another record.  Doing so
                    # would copy one customer's email/phone into a different customer's
                    # blank field, which is a serious data-integrity violation.
                    # ------------------------------------------------------------------
                    if self._is_unique_identifier_column(col):
                        recs.append(CleaningRecommendation(
                            issue_id=str(uuid.uuid4()),
                            issue_type=issue_type,
                            affected_columns=[col],
                            affected_rows=missing_cnt,
                            description=(
                                f"Missing value(s) in '{col}' — this column holds unique "
                                f"per-record values (e.g. email, phone, or ID)."
                            ),
                            suggested_operation='leave_unchanged',
                            operation_params={'column': col, 'reason': 'unique_identifier'},
                            explanation=(
                                f"Column '{col}' looks like a unique identifier or contact field. "
                                f"Auto-filling it with another record's value (mode) would assign the "
                                f"wrong email/phone to a different person — a data-integrity violation. "
                                f"Leave blank and let the user correct it manually, or drop the row."
                            ),
                            confidence=0.99,
                            risk_level='high',
                            requires_confirmation=True,
                            source='rule_based',
                            priority=1,
                        ))
                        continue  # Do NOT generate any fill recommendation for this column

                    is_heavy_loss = (missing_cnt / len(df) > 0.5) if len(df) > 0 else False
                    if is_heavy_loss:
                        op = 'drop_column'
                        params = {'columns': [col]}
                        exp = f"Column '{col}' has >50% missing values. Consider dropping it."
                        risk = 'medium'
                    else:
                        is_numeric = col in df.columns and pd.api.types.is_numeric_dtype(df[col])
                        op = 'fill_missing'
                        strategy = 'mean' if is_numeric else 'mode'
                        params = {'column': col, 'strategy': strategy, 'value': ''}
                        exp = f"Fill missing values in '{col}' using {strategy} strategy."
                        risk = 'low'
                    recs.append(CleaningRecommendation(
                        issue_id=str(uuid.uuid4()), issue_type=issue_type, affected_columns=[col], affected_rows=missing_cnt,
                        description=f"Missing values in '{col}'", suggested_operation=op, operation_params=params, explanation=exp,
                        confidence=0.85, risk_level=risk, requires_confirmation=True, source='rule_based', priority=2
                    ))
            elif issue_type == 'duplicate_rows':
                recs.append(CleaningRecommendation(
                    issue_id=str(uuid.uuid4()), issue_type=issue_type, affected_columns=affected_cols, affected_rows=rows,
                    description=desc, suggested_operation='remove_duplicates', operation_params={'keep': 'first'}, 
                    explanation="Remove exact duplicate rows, keeping the first occurrence.",
                    confidence=0.95, risk_level='low', requires_confirmation=False, source='rule_based', priority=1
                ))
            elif issue_type == 'whitespace_issues':
                cols = affected_cols if affected_cols else ([primary_col] if primary_col else df.columns.tolist())
                recs.append(CleaningRecommendation(
                    issue_id=str(uuid.uuid4()), issue_type=issue_type, affected_columns=cols, affected_rows=rows,
                    description=desc, suggested_operation='trim_whitespace', operation_params={'columns': cols}, 
                    explanation=f"Trim leading, trailing, and redundant internal whitespace in: {', '.join(cols)}.",
                    confidence=0.95, risk_level='low', requires_confirmation=False, source='rule_based', priority=3
                ))
            elif issue_type == 'inconsistent_case':
                for col in (affected_cols or [primary_col]):
                    if not col:
                        continue
                    recs.append(CleaningRecommendation(
                        issue_id=str(uuid.uuid4()), issue_type=issue_type, affected_columns=[col], affected_rows=rows,
                        description=f"Standardize casing in '{col}'", suggested_operation='normalize_case', 
                        operation_params={'column': col, 'case': 'lower'}, 
                        explanation=f"Convert text in column '{col}' to lowercase for category consistency.",
                        confidence=0.85, risk_level='low', requires_confirmation=False, source='rule_based', priority=3
                    ))
            elif issue_type == 'inconsistent_categories':
                for col in (affected_cols or [primary_col]):
                    if not col:
                        continue
                    recs.append(CleaningRecommendation(
                        issue_id=str(uuid.uuid4()), issue_type=issue_type, affected_columns=[col], affected_rows=rows,
                        description=f"Normalize categories in '{col}'", suggested_operation='normalize_categories',
                        operation_params={'column': col, 'mapping': {}},
                        explanation=f"Standardize near-duplicate category variations in '{col}'.",
                        confidence=0.80, risk_level='medium', requires_confirmation=True, source='rule_based', priority=3
                    ))
            elif issue_type == 'mixed_types':
                for col in (affected_cols or [primary_col]):
                    if not col:
                        continue
                    recs.append(CleaningRecommendation(
                        issue_id=str(uuid.uuid4()), issue_type=issue_type, affected_columns=[col], affected_rows=rows,
                        description=f"Convert mixed types in '{col}'", suggested_operation='convert_to_numeric', 
                        operation_params={'column': col, 'errors': 'coerce'}, 
                        explanation=f"Convert mixed types in '{col}' to numeric, setting invalid strings to NaN.",
                        confidence=0.75, risk_level='medium', requires_confirmation=True, source='rule_based', priority=2
                    ))
            elif issue_type == 'empty_column':
                recs.append(CleaningRecommendation(
                    issue_id=str(uuid.uuid4()), issue_type=issue_type, affected_columns=affected_cols, affected_rows=rows,
                    description=desc, suggested_operation='drop_column', operation_params={'columns': affected_cols},
                    explanation=f"Drop empty column(s) {', '.join(affected_cols)} as they contain no data.",
                    confidence=0.99, risk_level='low', requires_confirmation=True, source='rule_based', priority=1
                ))
            elif issue_type == 'constant_column':
                recs.append(CleaningRecommendation(
                    issue_id=str(uuid.uuid4()), issue_type=issue_type, affected_columns=affected_cols, affected_rows=rows,
                    description=desc, suggested_operation='drop_column', operation_params={'columns': affected_cols},
                    explanation=f"Consider dropping constant column(s) {', '.join(affected_cols)} providing zero variance.",
                    confidence=0.70, risk_level='low', requires_confirmation=True, source='rule_based', priority=4
                ))
            elif issue_type == 'empty_rows':
                recs.append(CleaningRecommendation(
                    issue_id=str(uuid.uuid4()), issue_type=issue_type, affected_columns=affected_cols, affected_rows=rows,
                    description=desc, suggested_operation='drop_rows_with_missing', operation_params={'how': 'all'},
                    explanation="Drop rows that are completely empty across all columns.",
                    confidence=0.98, risk_level='low', requires_confirmation=False, source='rule_based', priority=1
                ))
            elif issue_type == 'leading_zeros_risk':
                for col in (affected_cols or [primary_col]):
                    if not col:
                        continue
                    recs.append(CleaningRecommendation(
                        issue_id=str(uuid.uuid4()), issue_type=issue_type, affected_columns=[col], affected_rows=rows,
                        description=f"Preserve leading zeros in identifier column '{col}'",
                        suggested_operation='leave_unchanged',
                        operation_params={'column': col, 'reason': 'leading_zeros_identifier'},
                        explanation=(
                            f"Column '{col}' has leading zeros — it is likely an identifier (e.g. ZIP code, "
                            f"customer ID). Converting to numeric would silently strip those zeros. "
                            f"Keep the column as text and do NOT convert."
                        ),
                        confidence=0.95, risk_level='high', requires_confirmation=True, source='rule_based', priority=1
                    ))

            elif issue_type == 'invalid_values':
                # e.g. 'unknown' string in an otherwise-numeric Age column, or 'invalid-phone' in Phone column
                for col in (affected_cols or [primary_col]):
                    if not col:
                        continue
                    invalid_vals = issue.get('invalid_values', [])
                    col_lower = col.lower()
                    # If this is a phone or ID column, do NOT coerce to numeric (which strips leading zeros or turns into NaN float)
                    is_phone_or_id = any(k in col_lower for k in ('phone', 'mobile', 'tel', 'fax', 'id', 'code', 'sku', 'zip', 'postal'))
                    if is_phone_or_id:
                        recs.append(CleaningRecommendation(
                            issue_id=str(uuid.uuid4()),
                            issue_type=issue_type,
                            affected_columns=[col],
                            affected_rows=rows,
                            description=f"Column '{col}' contains invalid/non-standard identifier value(s): {invalid_vals}",
                            suggested_operation='flag_for_review',
                            operation_params={'column': col, 'invalid_values': invalid_vals},
                            explanation=(
                                f"Column '{col}' is a contact/identifier field containing invalid value(s): {invalid_vals}. "
                                f"Converting to numeric would lose leading zeros or discard original text. "
                                f"Flagged for manual review to preserve the original information."
                            ),
                            confidence=0.90,
                            risk_level='high',
                            requires_confirmation=True,
                            source='rule_based',
                            priority=2,
                        ))
                    else:
                        recs.append(CleaningRecommendation(
                            issue_id=str(uuid.uuid4()),
                            issue_type=issue_type,
                            affected_columns=[col],
                            affected_rows=rows,
                            description=f"Column '{col}' contains invalid non-numeric value(s): {invalid_vals}",
                            suggested_operation='convert_to_numeric',
                            operation_params={'column': col, 'errors': 'coerce'},
                            explanation=(
                                f"Column '{col}' is mostly numeric but contains {len(invalid_vals)} invalid "
                                f"string value(s): {invalid_vals}. Converting with errors='coerce' will turn "
                                f"those values into NaN (missing), which is safe and auditable. "
                                f"Review the NaN rows after conversion."
                            ),
                            confidence=0.85,
                            risk_level='medium',
                            requires_confirmation=True,
                            source='rule_based',
                            priority=2,
                        ))


            elif issue_type == 'suspicious_outliers':
                # e.g. Age 145 — flag for human review; NEVER auto-delete or auto-correct
                for col in (affected_cols or [primary_col]):
                    if not col:
                        continue
                    outlier_indices = issue.get('outlier_indices', [])
                    recs.append(CleaningRecommendation(
                        issue_id=str(uuid.uuid4()),
                        issue_type=issue_type,
                        affected_columns=[col],
                        affected_rows=rows,
                        description=f"Column '{col}' has {rows} suspicious/out-of-range value(s)",
                        suggested_operation='flag_for_review',
                        operation_params={'column': col, 'outlier_indices': outlier_indices},
                        explanation=(
                            f"Column '{col}' contains statistically extreme or domain-invalid values "
                            f"(e.g. Age=145). These may be data entry errors but could also be legitimate. "
                            f"Flagging for review — NO automatic change will be made. "
                            f"Manually correct or delete specific rows if appropriate."
                        ),
                        confidence=0.75,
                        risk_level='high',
                        requires_confirmation=True,
                        source='rule_based',
                        priority=2,
                    ))

            elif issue_type == 'mixed_date_formats':
                for col in (affected_cols or [primary_col]):
                    if not col:
                        continue
                    formats_found = issue.get('formats_found', [])
                    recs.append(CleaningRecommendation(
                        issue_id=str(uuid.uuid4()),
                        issue_type=issue_type,
                        affected_columns=[col],
                        affected_rows=rows,
                        description=f"Column '{col}' has mixed date formats: {', '.join(formats_found)}",
                        suggested_operation='parse_dates',
                        operation_params={'column': col, 'dayfirst': False},
                        explanation=(
                            f"Column '{col}' contains dates in multiple formats ({', '.join(formats_found)}). "
                            f"Parsing to a uniform datetime type ensures consistent comparisons. "
                            f"Review dayfirst setting if your data uses DD/MM/YYYY format."
                        ),
                        confidence=0.80,
                        risk_level='medium',
                        requires_confirmation=True,
                        source='rule_based',
                        priority=3,
                    ))

            else:
                # Unknown issue type — show it for review but do NOT map to trim_whitespace
                recs.append(CleaningRecommendation(
                    issue_id=str(uuid.uuid4()),
                    issue_type=issue_type,
                    affected_columns=affected_cols,
                    affected_rows=rows,
                    description=desc,
                    suggested_operation='flag_for_review',
                    operation_params={'reason': desc},
                    explanation=(
                        f"Issue of type '{issue_type}' detected. No automatic fix is available for this "
                        f"issue type. Please review and handle manually."
                    ),
                    confidence=0.5,
                    risk_level='medium',
                    requires_confirmation=True,
                    source='rule_based',
                    priority=5,
                ))
        return recs
    
    def _generate_ai_recommendations(self, df: pd.DataFrame, issues: list[dict]) -> list[CleaningRecommendation]:
        """Generate recommendations using OpenAI API.
        Only send aggregate statistics and column info, NOT raw data.
        Parse structured response into CleaningRecommendation objects.
        Validate that suggested operations are in ALLOWED_OPERATIONS.
        Fall back to rule-based if API call fails."""
        if not openai:
            return self._generate_rule_based_recommendations(df, issues)
            
        context = self._prepare_ai_context(df, issues)
        prompt = f"""
        You are a data cleaning expert. Analyze the following dataset metadata and detected issues:
        {context}
        
        Provide cleaning recommendations. Return a JSON array of objects with the following keys:
        - issue_type: string
        - affected_columns: list of strings
        - affected_rows: integer
        - description: string
        - suggested_operation: string (must be one of: {', '.join(ALLOWED_OPERATIONS)})
        - operation_params: dictionary of parameters for the operation
        - explanation: string
        - confidence: float between 0.0 and 1.0
        - risk_level: string ('low', 'medium', 'high')
        - requires_confirmation: boolean
        - priority: integer (1 to 5)
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful data engineering assistant. Only output valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            content = response.choices[0].message['content']
            
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end != 0:
                json_data = json.loads(content[start:end])
                recs = []
                for item in json_data:
                    if item.get('suggested_operation') in ALLOWED_OPERATIONS:
                        recs.append(CleaningRecommendation(
                            issue_id=str(uuid.uuid4()),
                            issue_type=item.get('issue_type', 'unknown'),
                            affected_columns=item.get('affected_columns', []),
                            affected_rows=item.get('affected_rows', 0),
                            description=item.get('description', ''),
                            suggested_operation=item.get('suggested_operation'),
                            operation_params=item.get('operation_params', {}),
                            explanation=item.get('explanation', ''),
                            confidence=item.get('confidence', 0.5),
                            risk_level=item.get('risk_level', 'medium'),
                            requires_confirmation=item.get('requires_confirmation', True),
                            source='ai',
                            priority=item.get('priority', 3)
                        ))
                return recs
            else:
                return self._generate_rule_based_recommendations(df, issues)
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return self._generate_rule_based_recommendations(df, issues)
    
    def _prepare_ai_context(self, df: pd.DataFrame, issues: list[dict]) -> str:
        """Prepare context for AI without sending raw data.
        Include: column names, dtypes, shape, missing counts, sample statistics,
        detected issues. Do NOT include actual data values."""
        context = {
            "dataset_shape": df.shape,
            "columns": {},
            "issues": issues
        }
        for col in df.columns:
            context["columns"][col] = {
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique())
            }
        return json.dumps(context, indent=2)
    
    def get_recommendations_summary(self) -> str:
        """Generate a human-readable summary of all recommendations."""
        if not self.recommendations:
            return "No cleaning recommendations available."
            
        summary = ["### Data Cleaning Recommendations\n"]
        for i, rec in enumerate(sorted(self.recommendations, key=lambda x: x.priority)):
            summary.append(f"{i+1}. **{rec.issue_type.replace('_', ' ').title()}** (Priority: {rec.priority}, Risk: {rec.risk_level.title()})")
            summary.append(f"   - **Issue**: {rec.description}")
            summary.append(f"   - **Suggestion**: {rec.suggested_operation} on {rec.affected_columns}")
            summary.append(f"   - **Explanation**: {rec.explanation}")
            summary.append(f"   - **Confidence**: {rec.confidence*100:.1f}%\n")
            
        return "\n".join(summary)
    
    def answer_dataset_question(self, df: pd.DataFrame, question: str) -> str:
        """Answer questions about the dataset using metadata and aggregate stats.
        Works both with and without AI. For rule-based: parse common question patterns
        and return relevant statistics."""
        q_lower = question.lower()
        if self.use_ai:
            try:
                context = self._prepare_ai_context(df, [])
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful data assistant. Answer the user's question based on the dataset metadata."},
                        {"role": "user", "content": f"Dataset Metadata:\n{context}\n\nQuestion: {question}"}
                    ]
                )
                return response.choices[0].message['content']
            except:
                pass # Fallback to rule-based
                
        # Rule-based fallback
        if "rows" in q_lower or "size" in q_lower or "how many" in q_lower:
            return f"The dataset has {len(df)} rows and {len(df.columns)} columns."
        elif "missing" in q_lower or "null" in q_lower:
            missing = df.isna().sum()
            missing = missing[missing > 0]
            if missing.empty:
                return "There are no missing values in the dataset."
            else:
                return "Missing values by column:\n" + "\n".join(f"- {k}: {v}" for k, v in missing.items())
        elif "columns" in q_lower:
            return "Columns: " + ", ".join(df.columns)
        else:
            return "I am a simple rule-based assistant and can only answer questions about rows, columns, and missing values right now. Enable AI for more complex answers."
    
    def get_cleaning_summary(self, operations_applied: list[dict]) -> str:
        """Generate a human-readable summary of all cleaning operations performed."""
        if not operations_applied:
            return "No cleaning operations were applied."
            
        summary = ["### Cleaning Operations Summary\n"]
        for i, op in enumerate(operations_applied):
            op_name = op.get('operation', 'Unknown Operation')
            cols = op.get('columns', 'Multiple/Unknown')
            summary.append(f"{i+1}. Applied `{op_name}` on {cols}.")
            if 'details' in op:
                summary.append(f"   Details: {op['details']}")
                
        return "\n".join(summary)
