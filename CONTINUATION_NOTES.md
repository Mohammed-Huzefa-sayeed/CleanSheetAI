# CleanSheet AI — Continuation Notes

**Last updated:** 2026-10-10

---

## 1. Project Overview & Status

CleanSheet AI is an AI-powered data cleaning platform built with Python, Streamlit, Pandas, RapidFuzz, Plotly, and ReportLab.
Locked Problem Statement: *"Upload messy Excel or CSV files and clean, normalize, deduplicate and restructure them automatically."*

All modules are fully functional with a complete 70-test passing regression suite across customer and sales datasets. The application runs headlessly and interactively via Streamlit without requiring an OpenAI API key.

---

## 2. Completed Bug Fixes & Improvements

### A. Sales Dataset Integrity & Imputation Guardrails
- **Date parsing:**
  - `cleaning_engine.parse_dates` updated with element-wise parsing preserving each distinct date format (ISO, DD/MM/YYYY, etc.) without cross-row contamination or replacing with `2026-01-05`.
  - `data_manager.py` apply_operation updated to invoke `cleaning_engine.parse_dates` with proper parameters.
- **Numeric conversion:**
  - Unparseable values like `'two'` in `Quantity` are converted to `NaN` under `errors='coerce'` or flagged for review; never silently imputed with statistical mean/mode (preventing `1.428571`).
- **Missing Total & City values:**
  - Added strict keyword protection (`total`, `amount`, `price`, `city`, `address`, `location`, `date`, `phone`, `id`) in `ai_recommender.py` and `data_manager.py` preventing mode/mean auto-imputation. Dev Malhotra's Total and Ananya Iyer's City remain properly preserved.
- **Order IDs and Identifiers:**
  - Identifier columns (`Order ID`, etc.) are formatted explicitly with text format (`number_format = '@'`) in `reporting.export_clean_excel` so leading zeros like `00101` are never truncated.
- **Suspicious Totals:**
  - Added `'total': (0, 1000000)` to domain bounds in `data_manager.py`, flagging Kabir Joshi's `9999999` for review.
- **Invalid Phone Numbers:**
  - Phone and identifier columns containing invalid text like `'invalid-phone'` are flagged for review (`flag_for_review`), preserving the original data.

### B. AI Cleaning Assistant Apply Buttons & Execution Flow
- Low-risk operations (`requires_confirmation=False`) execute and commit immediately with `apply_operation()` + `accept_change()`.
- Medium- and High-risk operations (`requires_confirmation=True`) correctly transition into `st.session_state.pending_operation` with full preview and validation warnings before explicit user acceptance or rejection.
- Buttons use stable unique keys based on `rec.issue_id`.
- Informational operations (`leave_unchanged`, `flag_for_review`) display clean explanatory cards without mutating data, with an option to audit-log the review.

### C. Missing Token & Blanks Handling
- `cleaning_engine.detect_blank_strings` and `cleaning_engine.convert_blanks_to_nan` detect and normalize empty strings, whitespace, and placeholder missing tokens (`'None'`, `'null'`, `'nan'`, `'NA'`, `'N/A'`, `'n/a'`, `'missing'`, `'-'`, `'--'`).
- `cleaning_engine.fill_missing_with_strategy` pre-processes target columns with `convert_blanks_to_nan` so text placeholder tokens loaded under `dtype=str` are treated as missing values prior to strategy application.
- Added `convert_blanks_to_nan` to `data_manager.apply_operation` and wired the "Convert all blanks to NaN" button in `app.py` directly to `convert_blanks_to_nan` rather than `trim_whitespace`.

### D. Pandas Dtype Consistency Across UI Tabs
- Replaced all legacy `dtype == 'object'` column filtering checks in `app.py` with `_is_string_col` from `data_manager`, ensuring consistent behavior across both legacy `object` and pandas 2.0+ `StringDtype`.

### E. Duplicate Detection on Non-Consecutive / Filtered Indices
- `get_comparison_view()` handles label-based `.loc` matching with positional `.iloc` fallback.
- `get_field_match_details()` and `resolve_group()` safely validate `df.index` before accessing or dropping rows.

---

## 3. Test Suite Verification

Run command:
```powershell
python -m pytest tests -v
```

**Results:**
```
======================= 70 passed, 4 warnings in 1.90s =======================
```

### Coverage includes:
- Customer dataset regression suite (`tests/test_customer_regression.py` - 28 tests)
- Sales dataset regression suite (`tests/test_sales_regression.py` - 7 tests)
- AI recommendation accuracy & safety filters (14 tests)
- Data manager & profiling (6 tests)
- Cleaning engine core operations (9 tests)
- Duplicate detector exact & fuzzy blocking (5 tests)
- Recipe manager validation & compatibility (3 tests)
- Reporting & exports sanitization (5 tests)
- Validation engine integrity checks (5 tests)

---

## 4. How to Run CleanSheet AI

```powershell
cd C:\Users\khada\OneDrive\Dokumen\Project\cleansheet-ai
python -m streamlit run app.py
```
*(Streamlit runs locally at `http://localhost:8501` or `http://localhost:8502`)*
