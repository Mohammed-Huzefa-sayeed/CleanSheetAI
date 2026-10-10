# CleanSheet AI — Advanced AI-Powered Data Cleaning Platform

> **"From messy data to decision-ready data."**

CleanSheet AI is a professional, privacy-first, and explainable data-cleaning platform engineered to automatically clean, normalize, deduplicate, validate, and restructure messy CSV and Excel spreadsheets.

---

## 🌟 Key Differentiators & Features

1. **Explainable AI & Rule-Based Recommendations**:
   - Analyzes detected data anomalies and provides actionable, human-readable recommendations with confidence metrics and risk levels (Low, Medium, High).
   - Functions 100% offline out-of-the-box using built-in deterministic rule heuristics.
   - Optional OpenAI API integration with strict privacy protections (sends aggregated schema/metrics only, never raw row-level records).

2. **Exact & Fuzzy Duplicate Matching**:
   - Detects exact row duplicates across full datasets or custom subset columns.
   - Integrates **RapidFuzz** with token-sort ratio algorithms and phonetic blocking (First3, Soundex) to surface potential fuzzy duplicates side-by-side with field-by-field similarity scores.

3. **Data Damage Prevention & Validation Engine**:
   - Automatically inspects proposed transformations before applying them.
   - Detects and issues warnings for:
     - Unexpected row loss (>10% warning, >50% critical)
     - Column drops
     - Increases in missing values
     - Identifier truncation and damage (preserving leading zeros in codes/ZIPs like `00123`)
     - Numeric total distortions
     - Schema incompatibility

4. **Interactive Cleaning Workspace & Undo Engine**:
   - Apply missing value imputation (Mean, Median, Mode, Value, Forward/Backward fill, or drop).
   - Normalize whitespace, standardize text casing (lower, upper, title), parse mixed date formats, and standardize boolean values.
   - Restructure datasets via column reordering, renaming, splitting, merging, and reshaping (Melt wide-to-long, Pivot long-to-wide).
   - Reversible operations with full undo and reapply stack backed by snapshot hashes.

5. **Reusable Cleaning Recipes**:
   - Save transformation sequences as secure, executable-free JSON cleaning recipes.
   - Test recipe schema compatibility against new datasets with automated warnings for missing/incompatible columns.

6. **Before/After Comparison & Heuristic Quality Score**:
   - Interactive comparison dashboard showing metric shifts (row count, columns, missing cells, duplicates, cells changed).
   - Multidimensional Quality Score evaluating Completeness (30%), Uniqueness (25%), Consistency (25%), and Validity (20%).

7. **Reports & Exports with Injection Protection**:
   - Download cleaned datasets as CSV or styled Excel workbooks (`.xlsx`) with bold headers and auto-adjusted column widths.
   - Built-in Formula Injection Sanitization (neutralizing cells starting with `=`, `+`, `-`, `@`, `\t`, `\r`).
   - Professional PDF Data Quality Reports generated via **ReportLab**.
   - Downloadable JSON audit history.

---

## 🏗️ Architecture & Project Structure

```
cleansheet-ai/
├── app.py                     # Streamlit frontend with 10 integrated pages
├── config.py                  # Global configurations, env management, security constants
├── data_manager.py            # Core dataset state, profiling, issue detection, undo/redo
├── cleaning_engine.py         # Pure transformation and normalization operators
├── duplicate_detector.py      # RapidFuzz-powered exact & fuzzy duplicate matching
├── validation_engine.py       # Data damage prevention & safety checks
├── ai_recommender.py          # AI & Rule-based recommendation engine with privacy boundaries
├── recipe_manager.py          # JSON recipe generation, serialization, and compatibility
├── reporting.py               # ReportLab PDF report generator & openpyxl exports
├── sample_messy_sales.csv     # Representative messy dataset with realistic errors
├── requirements.txt           # Project dependencies
├── .env.example               # Configuration template
├── .gitignore                 # Version control ignores
└── tests/                     # Comprehensive automated unit & integration tests
    ├── test_data_manager.py
    ├── test_cleaning_engine.py
    ├── test_duplicate_detector.py
    ├── test_validation_engine.py
    ├── test_recipe_manager.py
    ├── test_reporting.py
    └── test_ai_recommender.py
```

---

## 🚀 Getting Started

### 1. Requirements & Dependencies
Ensure Python 3.10+ (tested on Python 3.14) is installed. Install the dependencies:

```bash
cd cleansheet-ai
pip install -r requirements.txt
```

### 2. Environment Setup (Optional)
The application is fully functional offline without any API keys. If you wish to enable OpenAI API assistance:

```bash
copy .env.example .env
```
Edit `.env` and set:
```ini
OPENAI_API_KEY=sk-...
AI_ENABLED=true
```

### 3. Running the Application
Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```
Or with python executable directly:
```bash
python -m streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the full automated test suite using `pytest` or `unittest`:

```bash
python -m pytest tests
```
Or:
```bash
python -m unittest discover tests
```

All 36 unit tests pass covering data loading, cleaning operators, date parsing, fuzzy deduplication, formula injection prevention, validation rules, recipes, and PDF report generation.
