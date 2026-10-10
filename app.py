"""
CleanSheet AI — Advanced AI-Powered Data Cleaning Platform
Main Streamlit Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import io
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import APP_NAME, TAGLINE, VERSION, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS, load_env, get_openai_api_key, AI_ENABLED
from data_manager import (
    DatasetInfo, ChangeRecord, load_file, get_sheet_names,
    get_column_profile, get_dataset_profile, detect_issues,
    apply_operation, accept_change, undo_last_change, compute_quality_score,
    _is_string_col
)
from cleaning_engine import (
    fill_missing_with_strategy, drop_rows_with_missing, detect_blank_strings,
    convert_blanks_to_nan, trim_whitespace, normalize_case, normalize_categories,
    normalize_column_names, convert_to_numeric, parse_dates, normalize_booleans,
    rename_column, drop_columns, reorder_columns, split_column, merge_columns,
    detect_numeric_outliers, get_operation_impact
)
from duplicate_detector import DuplicateDetector, DuplicateGroup
from validation_engine import ValidationEngine
from ai_recommender import AIRecommender, CleaningRecommendation
from recipe_manager import RecipeManager, CleaningRecipe, RecipeStep
from reporting import (
    generate_pdf_report, export_clean_csv, export_clean_excel,
    export_cleaning_history, generate_comparison_data
)

# Load environment
load_env()

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0d1b2a 100%);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #132238 100%);
        border-right: 1px solid #1e3a5f;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #c8d6e5 !important;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a2d42 0%, #1e3a54 100%);
        border: 1px solid #2a5478;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h3 {
        color: #00d4aa;
        margin: 0 0 8px 0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .metric-value {
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
    }
    
    /* Section headers */
    .section-header {
        color: #00d4aa;
        font-size: 24px;
        font-weight: 600;
        margin: 20px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #00d4aa40;
    }
    
    /* Issue badges */
    .severity-high { 
        background: #ff4757; color: white; padding: 2px 10px; border-radius: 12px; 
        font-size: 12px; font-weight: 600; 
    }
    .severity-medium { 
        background: #ffa502; color: #1a1a2e; padding: 2px 10px; border-radius: 12px; 
        font-size: 12px; font-weight: 600; 
    }
    .severity-low { 
        background: #2ed573; color: #1a1a2e; padding: 2px 10px; border-radius: 12px; 
        font-size: 12px; font-weight: 600; 
    }
    
    /* Info box */
    .info-box {
        background: #1a2d4250;
        border-left: 4px solid #00d4aa;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        color: #c8d6e5;
    }
    
    /* Buttons */
    .stButton > button {
        border: 1px solid #00d4aa;
        border-radius: 8px;
        color: #ffffff;
        background: linear-gradient(135deg, #00d4aa20, #00d4aa10);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00d4aa40, #00d4aa20);
        border-color: #00ffd5;
        box-shadow: 0 0 15px rgba(0, 212, 170, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1a2d4250;
        border-radius: 8px 8px 0 0;
        border: 1px solid #2a5478;
        color: #c8d6e5;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 1px solid #2a5478;
        border-radius: 8px;
    }
    
    /* Quality score gauge */
    .quality-score {
        text-align: center;
        padding: 20px;
    }
    .quality-score .score-value {
        font-size: 64px;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4aa, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .quality-score .score-label {
        font-size: 14px;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Privacy notice */
    .privacy-notice {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px;
        font-size: 13px;
        color: #8899aa;
    }
    
    /* Logo header */
    .logo-header {
        padding: 10px 0 20px 0;
        text-align: center;
    }
    .logo-header h1 {
        background: linear-gradient(135deg, #00d4aa, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px;
        margin-bottom: 4px;
    }
    .logo-header p {
        color: #8899aa;
        font-size: 13px;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'dataset_info': None,
        'issues': [],
        'recommendations': [],
        'duplicate_groups': [],
        'quality_before': None,
        'quality_after': None,
        'pending_operation': None,
        'pending_df': None,
        'pending_record': None,
        'recipe_manager': RecipeManager(recipes_dir=os.path.join(os.path.dirname(__file__), 'recipes')),
        'ai_recommender': None,
        'page': '🏠 Dashboard',
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-header">
        <h1>🧹 CleanSheet AI</h1>
        <p>From messy data to decision-ready data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    pages = [
        "🏠 Dashboard",
        "📤 Upload Dataset",
        "🔍 Data Profiler",
        "🤖 AI Assistant",
        "🧹 Cleaning Workspace",
        "🔗 Duplicate Detection",
        "🔄 Data Transformation",
        "📊 Before/After Comparison",
        "📋 Cleaning Recipes",
        "📥 Export & Reports",
    ]
    st.session_state.page = st.radio("Navigation", pages, label_visibility="collapsed")
    
    st.markdown("---")
    
    # Dataset status
    if st.session_state.dataset_info:
        di = st.session_state.dataset_info
        st.markdown(f"**📁 Loaded:** {di.filename}")
        st.markdown(f"**Rows:** {len(di.current_df):,} | **Cols:** {len(di.current_df.columns)}")
        st.markdown(f"**Changes:** {len(di.change_history)}")
        if di.change_history:
            if st.button("↩️ Undo Last Change"):
                if undo_last_change(di):
                    st.session_state.issues = detect_issues(di.current_df)
                    st.session_state.quality_after = compute_quality_score(di.current_df, di.original_df)
                    st.rerun()
                else:
                    st.error("Cannot undo.")
    else:
        st.info("No dataset loaded. Upload a file to get started.")
    
    st.markdown("---")
    st.markdown(f"""
    <div class="privacy-notice">
        🔒 <b>Privacy:</b> All data is processed locally. AI features require explicit opt-in
        and only send aggregate statistics, never raw data, unless you approve.
        <br><br>v{VERSION}
    </div>
    """, unsafe_allow_html=True)


# ─── Helper functions ──────────────────────────────────────────────────────────

def severity_badge(severity: str) -> str:
    """Return HTML badge for severity level."""
    return f'<span class="severity-{severity}">{severity.upper()}</span>'

def metric_card(title: str, value, col=None):
    """Render a metric card."""
    target = col if col else st
    target.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def ensure_dataset():
    """Check if dataset is loaded, show message if not."""
    if st.session_state.dataset_info is None:
        st.warning("⚠️ No dataset loaded. Please go to **📤 Upload Dataset** first.")
        return False
    return True

def apply_and_preview(operation_name: str, params: dict, description: str):
    """Apply an operation with preview and validation."""
    di = st.session_state.dataset_info
    try:
        new_df, record = apply_operation(di, operation_name, params)
        
        # Validate
        validator = ValidationEngine()
        warnings = validator.validate_transformation(di.current_df, new_df, operation_name, params)
        record.validation_warnings = [w.message for w in warnings]
        
        # Store pending
        st.session_state.pending_operation = description
        st.session_state.pending_df = new_df
        st.session_state.pending_record = record
        
        return new_df, record, warnings
    except Exception as e:
        st.error(f"❌ Operation failed: {str(e)}")
        return None, None, []

def show_pending_operation():
    """Show pending operation accept/reject buttons."""
    if st.session_state.pending_operation:
        di = st.session_state.dataset_info
        record = st.session_state.pending_record
        new_df = st.session_state.pending_df
        
        st.markdown("---")
        st.markdown(f"### 🔄 Pending: {st.session_state.pending_operation}")
        
        if record and record.validation_warnings:
            for w in record.validation_warnings:
                st.warning(f"⚠️ {w}")
        
        impact = get_operation_impact(di.current_df, new_df)
        icol1, icol2, icol3 = st.columns(3)
        icol1.metric("Rows", f"{impact['rows_after']:,}", f"{impact['rows_after'] - impact['rows_before']:+,}")
        icol2.metric("Missing Values", f"{impact['missing_after']:,}", f"{impact['missing_after'] - impact['missing_before']:+,}")
        icol3.metric("Duplicates", f"{impact.get('duplicate_rows_after', 0):,}")
        
        st.markdown("**Preview (first 10 rows):**")
        st.dataframe(new_df.head(10), use_container_width=True)
        
        acol1, acol2 = st.columns(2)
        if acol1.button("✅ Accept Change", type="primary", use_container_width=True):
            accept_change(di, new_df, record)
            st.session_state.issues = detect_issues(di.current_df)
            st.session_state.quality_after = compute_quality_score(di.current_df, di.original_df)
            st.session_state.pending_operation = None
            st.session_state.pending_df = None
            st.session_state.pending_record = None
            st.success("✅ Change applied successfully!")
            st.rerun()
        if acol2.button("❌ Reject Change", use_container_width=True):
            st.session_state.pending_operation = None
            st.session_state.pending_df = None
            st.session_state.pending_record = None
            st.info("Change rejected.")
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Dashboard ─────────────────────────────────────────────────────────────────
def page_dashboard():
    st.markdown(f"""
    <div style="text-align:center; padding: 30px 0 20px 0;">
        <h1 style="background: linear-gradient(135deg, #00d4aa, #00b4d8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 48px; margin-bottom: 8px;">🧹 {APP_NAME}</h1>
        <p style="color: #8899aa; font-size: 18px;">{TAGLINE}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.dataset_info is None:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📤 Step 1</h3>
                <div style="color:#c8d6e5; font-size:15px;">Upload your CSV or Excel file</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🔍 Step 2</h3>
                <div style="color:#c8d6e5; font-size:15px;">Review detected issues & AI recommendations</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📥 Step 3</h3>
                <div style="color:#c8d6e5; font-size:15px;">Clean, validate, and export your data</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### ✨ Key Features")
            features = [
                "🤖 **AI-powered** cleaning recommendations (works offline too)",
                "🔍 **Exact & fuzzy** duplicate detection with RapidFuzz",
                "🛡️ **Data damage prevention** with validation checks",
                "📋 **Reusable cleaning recipes** for repeatable workflows",
                "📊 **Before/After comparison** with quality scores",
                "↩️ **Full undo history** — never lose your original data",
                "🔒 **Privacy-first** — all processing happens locally",
            ]
            for f in features:
                st.markdown(f)
        with col_b:
            st.markdown("### 🚀 Getting Started")
            st.info("Navigate to **📤 Upload Dataset** in the sidebar to begin cleaning your data.")
            st.markdown("""
            **Supported formats:**
            - CSV files (auto-detects separator and encoding)
            - Excel files (.xlsx, .xls) with multi-sheet support
            
            **Sample data included:** Use the built-in sample dataset to explore features.
            """)
    else:
        di = st.session_state.dataset_info
        qs = st.session_state.quality_after or compute_quality_score(di.current_df, di.original_df)
        
        # Quality score
        score = qs.get('overall_score', 0)
        st.markdown("---")
        c1, c2, c3, c4, c5 = st.columns(5)
        metric_card("Quality Score", f"{score:.0f}%", c1)
        metric_card("Rows", f"{len(di.current_df):,}", c2)
        metric_card("Columns", f"{len(di.current_df.columns)}", c3)
        metric_card("Issues", f"{len(st.session_state.issues)}", c4)
        metric_card("Changes", f"{len(di.change_history)}", c5)
        
        st.markdown("---")
        col_l, col_r = st.columns(2)
        
        with col_l:
            st.markdown("### 📋 Detected Issues")
            if st.session_state.issues:
                for issue in st.session_state.issues[:8]:
                    sev = issue.get('severity', 'low')
                    st.markdown(f"{severity_badge(sev)} {issue['description']}", unsafe_allow_html=True)
            else:
                st.success("✅ No issues detected!")
        
        with col_r:
            st.markdown("### 📊 Quality Breakdown")
            components = qs.get('component_scores', {})
            if components:
                fig = go.Figure(go.Bar(
                    x=list(components.values()),
                    y=[k.title() for k in components.keys()],
                    orientation='h',
                    marker_color=['#00d4aa', '#00b4d8', '#3742fa', '#ffa502'],
                    text=[f"{v:.0%}" for v in components.values()],
                    textposition='auto',
                ))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#c8d6e5',
                    height=250,
                    margin=dict(l=0, r=0, t=10, b=10),
                    xaxis=dict(range=[0, 1], showgrid=False),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True)
            st.caption("⚠️ This is a heuristic score, not an absolute measure of correctness.")


# ─── Upload Dataset ────────────────────────────────────────────────────────────
def page_upload():
    st.markdown('<div class="section-header">📤 Upload Dataset</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help=f"Max file size: {MAX_FILE_SIZE_MB}MB"
        )
        
        use_sample = st.checkbox("📂 Use sample messy dataset instead")
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <b>Supported formats:</b><br>
            • CSV (auto-detect separator & encoding)<br>
            • Excel (.xlsx, .xls)<br><br>
            <b>Max size:</b> 200 MB<br>
            <b>Privacy:</b> Files are processed locally only.
        </div>
        """, unsafe_allow_html=True)
    
    sheet_name = None
    
    if uploaded_file and uploaded_file.name.endswith(('.xlsx', '.xls')):
        try:
            sheets = get_sheet_names(uploaded_file, uploaded_file.name)
            uploaded_file.seek(0)
            if len(sheets) > 1:
                sheet_name = st.selectbox("Select worksheet:", sheets)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
    
    if st.button("🚀 Load Dataset", type="primary") or use_sample:
        try:
            if use_sample:
                sample_path = os.path.join(os.path.dirname(__file__), 'sample_messy_sales.csv')
                if not os.path.exists(sample_path):
                    st.error("Sample file not found.")
                    return
                with open(sample_path, 'rb') as f:
                    dataset_info = load_file(f, 'sample_messy_sales.csv')
            elif uploaded_file:
                # File size check
                uploaded_file.seek(0, 2)
                size_mb = uploaded_file.tell() / (1024 * 1024)
                uploaded_file.seek(0)
                if size_mb > MAX_FILE_SIZE_MB:
                    st.error(f"File too large ({size_mb:.1f}MB). Maximum is {MAX_FILE_SIZE_MB}MB.")
                    return
                dataset_info = load_file(uploaded_file, uploaded_file.name, sheet_name)
            else:
                st.warning("Please upload a file or select the sample dataset.")
                return
            
            st.session_state.dataset_info = dataset_info
            st.session_state.issues = detect_issues(dataset_info.current_df)
            st.session_state.quality_before = compute_quality_score(dataset_info.original_df)
            st.session_state.quality_after = compute_quality_score(dataset_info.current_df, dataset_info.original_df)
            
            # Initialize AI recommender
            api_key = get_openai_api_key()
            st.session_state.ai_recommender = AIRecommender(
                use_ai=AI_ENABLED and api_key is not None,
                api_key=api_key
            )
            
            st.success(f"✅ **{dataset_info.filename}** loaded successfully!")
            st.rerun()
            
        except ValueError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    # Show preview if loaded
    if st.session_state.dataset_info:
        di = st.session_state.dataset_info
        st.markdown("---")
        st.markdown("### 📋 Dataset Preview")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(di.current_df):,}")
        c2.metric("Columns", f"{len(di.current_df.columns)}")
        c3.metric("Missing Cells", f"{di.current_df.isnull().sum().sum():,}")
        mem = di.current_df.memory_usage(deep=True).sum()
        c4.metric("Memory", f"{mem / 1024:.1f} KB" if mem < 1048576 else f"{mem / 1048576:.1f} MB")
        
        st.dataframe(di.current_df.head(20), use_container_width=True, height=400)
        
        if st.session_state.issues:
            st.markdown(f"### ⚠️ {len(st.session_state.issues)} Issue(s) Detected")
            for issue in st.session_state.issues:
                sev = issue.get('severity', 'low')
                st.markdown(f"{severity_badge(sev)} {issue['description']}", unsafe_allow_html=True)


# ─── Data Profiler ─────────────────────────────────────────────────────────────
def page_profiler():
    st.markdown('<div class="section-header">🔍 Data Profiler</div>', unsafe_allow_html=True)
    if not ensure_dataset():
        return
    
    di = st.session_state.dataset_info
    df = di.current_df
    profile = get_dataset_profile(df)
    
    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    metric_card("Rows", f"{profile['row_count']:,}", c1)
    metric_card("Columns", f"{profile['col_count']}", c2)
    metric_card("Missing Cells", f"{profile['total_missing']:,}", c3)
    metric_card("Duplicates", f"{profile['duplicate_rows']}", c4)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Column Details", "📈 Visualizations", "🔎 Column Inspector"])
    
    with tab1:
        col_data = []
        for col_name, col_prof in profile['column_profiles'].items():
            col_data.append({
                'Column': col_name,
                'Type': col_prof['dtype'],
                'Unique': col_prof['unique_count'],
                'Missing': col_prof['null_count'],
                'Missing %': f"{col_prof['null_pct']:.1f}%",
                'Constant': '⚠️' if col_prof['is_constant'] else '',
                'Samples': str(col_prof['sample_values'][:3]),
            })
        st.dataframe(pd.DataFrame(col_data), use_container_width=True, height=400)
    
    with tab2:
        # Missing values chart
        missing_data = {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0}
        if missing_data:
            st.markdown("#### Missing Values by Column")
            fig = px.bar(
                x=list(missing_data.keys()), y=list(missing_data.values()),
                labels={'x': 'Column', 'y': 'Missing Count'},
                color_discrete_sequence=['#ff4757']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#c8d6e5', height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Column type distribution
        type_counts = df.dtypes.astype(str).value_counts()
        st.markdown("#### Column Type Distribution")
        fig2 = px.pie(names=type_counts.index, values=type_counts.values,
                      color_discrete_sequence=['#00d4aa', '#00b4d8', '#3742fa', '#ffa502'])
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#c8d6e5', height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        selected_col = st.selectbox("Select a column to inspect:", df.columns.tolist())
        if selected_col:
            col_prof = get_column_profile(df, selected_col)
            
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Type", col_prof['dtype'])
            mc2.metric("Unique Values", col_prof['unique_count'])
            mc3.metric("Missing", f"{col_prof['null_count']} ({col_prof['null_pct']:.1f}%)")
            mc4.metric("Constant?", "Yes ⚠️" if col_prof['is_constant'] else "No")
            
            st.markdown("**Sample Values:**")
            st.code(str(col_prof['sample_values']))
            
            if col_prof.get('looks_like_numeric'):
                st.info("💡 This column looks like it could be numeric.")
            if col_prof.get('looks_like_date'):
                st.info("📅 This column looks like it could contain dates.")
            if col_prof.get('has_leading_zeros'):
                st.warning("⚠️ This column has leading zeros — may be an identifier. Be careful with numeric conversion.")
            
            # Value distribution for low-cardinality columns
            if col_prof['unique_count'] <= 30 and col_prof['unique_count'] > 0:
                st.markdown("**Value Distribution:**")
                vc = df[selected_col].value_counts().head(20)
                fig = px.bar(x=vc.index.astype(str), y=vc.values,
                             labels={'x': selected_col, 'y': 'Count'},
                             color_discrete_sequence=['#00d4aa'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#c8d6e5', height=300
                )
                st.plotly_chart(fig, use_container_width=True)


# ─── AI Assistant ──────────────────────────────────────────────────────────────
def page_ai_assistant():
    st.markdown('<div class="section-header">🤖 AI Cleaning Assistant</div>', unsafe_allow_html=True)
    if not ensure_dataset():
        return
    
    di = st.session_state.dataset_info
    recommender = st.session_state.ai_recommender
    if recommender is None:
        api_key = get_openai_api_key()
        recommender = AIRecommender(use_ai=AI_ENABLED and api_key is not None, api_key=api_key)
        st.session_state.ai_recommender = recommender
    
    source_label = "AI-powered" if recommender.use_ai else "Rule-based"
    st.info(f"🔧 Using **{source_label}** recommendation engine. "
            + ("Configure `OPENAI_API_KEY` in `.env` for AI-powered recommendations." if not recommender.use_ai else ""))
    
    if st.button("🔍 Analyze Dataset & Generate Recommendations", type="primary"):
        with st.spinner("Analyzing your dataset..."):
            issues = st.session_state.issues or detect_issues(di.current_df)
            recommendations = recommender.analyze_dataset(di.current_df, issues)
            st.session_state.recommendations = recommendations
    
    recommendations = st.session_state.recommendations
    if recommendations:
        st.markdown(f"### 💡 {len(recommendations)} Recommendation(s)")

        # Show any pending confirmation FIRST so it appears right after a rerun
        if st.session_state.pending_operation:
            show_pending_operation()
            st.markdown("---")

        for i, rec in enumerate(recommendations):
            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(rec.risk_level, "⚪")
            # Stable unique key: use recommendation's issue_id (not index i which shifts)
            safe_key = rec.issue_id.replace("-", "")[:20]
            op = rec.suggested_operation

            with st.expander(f"{risk_emoji} {rec.description} — Confidence: {rec.confidence:.0%}", expanded=(i < 3)):
                st.markdown(f"**Type:** {rec.issue_type}")
                st.markdown(f"**Affected columns:** {', '.join(rec.affected_columns)}")
                st.markdown(f"**Rows affected:** {rec.affected_rows:,}")
                st.markdown(f"**Explanation:** {rec.explanation}")
                st.markdown(f"**Suggested operation:** `{op}`")
                st.markdown(f"**Risk level:** {risk_emoji} {rec.risk_level.upper()}")
                st.markdown(f"**Source:** {rec.source}")

                if rec.requires_confirmation:
                    st.warning("⚠️ This operation requires your confirmation before applying.")

                # ── Apply button logic per operation type ──────────────────────
                if op in ('leave_unchanged', 'flag_for_review'):
                    # Informational only — no data mutation
                    col_label = ', '.join(f'`{c}`' for c in rec.affected_columns) if rec.affected_columns else 'N/A'
                    st.info(
                        f"ℹ️ **No automatic change** will be made for this issue. "
                        f"Column(s) {col_label} will remain exactly as-is. "
                        f"Review and correct manually if needed."
                    )
                    if st.button("📋 Log as Reviewed (no data change)", key=f"log_{safe_key}"):
                        di = st.session_state.dataset_info
                        try:
                            new_df, record = apply_operation(di, 'flag_for_review', rec.operation_params)
                            accept_change(di, new_df, record)
                            st.session_state.quality_after = compute_quality_score(di.current_df, di.original_df)
                            st.success("✅ Logged as reviewed. No data was changed.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Logging failed: {e}")

                elif not rec.requires_confirmation:
                    # Low-risk: apply & accept immediately, no confirm step needed
                    if st.button(f"▶️ Apply: {op}", key=f"apply_{safe_key}"):
                        di = st.session_state.dataset_info
                        try:
                            new_df, record = apply_operation(di, op, rec.operation_params)
                            validator = ValidationEngine()
                            warns = validator.validate_transformation(di.current_df, new_df, op, rec.operation_params)
                            record.validation_warnings = [w.message for w in warns]
                            accept_change(di, new_df, record)
                            st.session_state.issues = detect_issues(di.current_df)
                            st.session_state.quality_after = compute_quality_score(di.current_df, di.original_df)
                            st.success(f"✅ `{op}` applied. {record.affected_rows_count:,} row(s) affected.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to apply `{op}`: {e}")

                else:
                    # Medium/high-risk: show preview + accept/reject flow
                    if st.button(f"▶️ Preview & Apply: {op}", key=f"apply_{safe_key}"):
                        di = st.session_state.dataset_info
                        try:
                            new_df, record = apply_operation(di, op, rec.operation_params)
                            validator = ValidationEngine()
                            warns = validator.validate_transformation(di.current_df, new_df, op, rec.operation_params)
                            record.validation_warnings = [w.message for w in warns]
                            st.session_state.pending_operation = f"{op} — {rec.description}"
                            st.session_state.pending_df = new_df
                            st.session_state.pending_record = record
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to prepare `{op}`: {e}")

        st.markdown("---")
        st.markdown("### 📝 Cleaning Summary")
        st.text(recommender.get_recommendations_summary())

    # If there's a pending op but no recommendations loaded, show confirm UI here
    if not recommendations and st.session_state.pending_operation:
        show_pending_operation()

    # Dataset Q&A
    st.markdown("---")
    st.markdown("### ❓ Ask About Your Dataset")
    question = st.text_input("Ask a question about your data:", placeholder="How many missing values are there?")
    if question:
        answer = recommender.answer_dataset_question(di.current_df, question)
        st.markdown(f"**Answer:** {answer}")



# ─── Cleaning Workspace ───────────────────────────────────────────────────────
def page_cleaning_workspace():
    st.markdown('<div class="section-header">🧹 Cleaning Workspace</div>', unsafe_allow_html=True)
    if not ensure_dataset():
        return
    
    di = st.session_state.dataset_info
    df = di.current_df
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 Missing Values", "✂️ Normalization", "🏷️ Column Operations", 
        "📊 Anomaly Detection", "📜 Change History"
    ])
    
    # ── Missing Values Tab ──
    with tab1:
        st.markdown("### Handle Missing Values")
        missing_cols = [col for col in df.columns if df[col].isnull().any()]
        
        if not missing_cols:
            st.success("✅ No missing values in the current dataset!")
        else:
            st.markdown(f"**{len(missing_cols)} column(s)** with missing values:")
            
            selected_col = st.selectbox("Select column:", missing_cols, key="missing_col")
            miss_count = int(df[selected_col].isnull().sum())
            st.metric(f"Missing in '{selected_col}'", f"{miss_count} ({miss_count/len(df)*100:.1f}%)")
            
            strategy = st.selectbox("Fill strategy:", [
                "Leave unchanged", "Drop rows with missing", "Fill with specific value",
                "Fill with mean", "Fill with median", "Fill with mode",
                "Forward fill", "Backward fill"
            ], key="fill_strategy")
            
            fill_value = None
            if strategy == "Fill with specific value":
                fill_value = st.text_input("Enter fill value:", key="fill_value")
            
            strategy_map = {
                "Fill with mean": "mean", "Fill with median": "median",
                "Fill with mode": "mode", "Forward fill": "forward_fill",
                "Backward fill": "backward_fill", "Fill with specific value": "value",
            }
            
            if strategy != "Leave unchanged":
                impact_text = ""
                if strategy == "Drop rows with missing":
                    rows_to_drop = int(df[selected_col].isnull().sum())
                    impact_text = f"⚠️ This will remove up to **{rows_to_drop}** rows."
                elif strategy in ("Fill with mean", "Fill with median"):
                    if not pd.api.types.is_numeric_dtype(df[selected_col]):
                        st.warning("⚠️ Mean/Median only work on numeric columns.")
                
                if impact_text:
                    st.markdown(impact_text)
                
                if st.button("🔄 Apply", key="apply_missing"):
                    if strategy == "Drop rows with missing":
                        result = apply_and_preview('drop_rows_with_missing', 
                                                   {'columns': [selected_col]},
                                                   f"Drop rows missing '{selected_col}'")
                    else:
                        op_params = {'column': selected_col, 'value': fill_value or ''}
                        if strategy in strategy_map:
                            op_params['strategy'] = strategy_map[strategy]
                        result = apply_and_preview('fill_missing', op_params,
                                                   f"Fill missing in '{selected_col}' with {strategy}")
                    if result[0] is not None:
                        st.rerun()
        
        # Blank strings
        st.markdown("---")
        st.markdown("### Convert Blank Strings to Missing")
        blanks = detect_blank_strings(df)
        if blanks:
            st.markdown(f"Found blank strings in **{len(blanks)} column(s)**:")
            for col, count in blanks.items():
                st.markdown(f"- `{col}`: {count} blank string(s)")
            if st.button("Convert all blanks to NaN", key="convert_blanks"):
                result = apply_and_preview('convert_blanks_to_nan', {'columns': list(blanks.keys())},
                                           "Convert blank strings to NaN")
                if result[0] is not None:
                    st.rerun()
        else:
            st.success("✅ No blank strings detected.")
    
    # ── Normalization Tab ──
    with tab2:
        st.markdown("### Text Normalization")
        
        norm_op = st.selectbox("Operation:", [
            "Trim whitespace (all columns)",
            "Normalize case",
            "Normalize column names",
            "Convert to numeric",
            "Parse dates",
            "Normalize boolean values",
            "Normalize categories",
        ], key="norm_operation")
        
        if norm_op == "Trim whitespace (all columns)":
            st.info("Removes leading/trailing whitespace and collapses multiple spaces in all text columns.")
            if st.button("🔄 Apply Trim", key="apply_trim"):
                result = apply_and_preview('trim_whitespace', {}, "Trim whitespace (all columns)")
                if result[0] is not None:
                    st.rerun()
        
        elif norm_op == "Normalize case":
            str_cols = [c for c in df.columns if _is_string_col(df[c])]
            col = st.selectbox("Column:", str_cols or df.columns.tolist(), key="case_col")
            case = st.selectbox("Case:", ["lower", "upper", "title"], key="case_type")
            if st.button("🔄 Apply Case Normalization", key="apply_case"):
                result = apply_and_preview('normalize_case', {'column': col, 'case': case},
                                           f"Normalize '{col}' to {case}")
                if result[0] is not None:
                    st.rerun()

        
        elif norm_op == "Normalize column names":
            st.info("Converts column names to lowercase with underscores. Original names are preserved in history.")
            if st.button("🔄 Apply", key="apply_norm_cols"):
                result = apply_and_preview('normalize_column_names', {}, "Normalize column names")
                if result[0] is not None:
                    st.rerun()
        
        elif norm_op == "Convert to numeric":
            text_cols = [c for c in df.columns if _is_string_col(df[c])]
            col = st.selectbox("Column:", text_cols, key="num_col")
            if col:
                profile = get_column_profile(df, col)
                if profile.get('has_leading_zeros'):
                    st.warning("⚠️ This column has leading zeros. Converting to numeric will remove them!")
                if st.button("🔄 Convert to Numeric", key="apply_numeric"):
                    result = apply_and_preview('convert_numeric', {'column': col},
                                               f"Convert '{col}' to numeric")
                    if result[0] is not None:
                        st.rerun()
        
        elif norm_op == "Parse dates":
            col = st.selectbox("Column:", df.columns.tolist(), key="date_col")
            dayfirst = st.checkbox("Day comes first in dates (dd/mm/yyyy)", key="dayfirst")
            if st.button("🔄 Parse Dates", key="apply_dates"):
                result = apply_and_preview('parse_dates', {'column': col, 'dayfirst': dayfirst},
                                           f"Parse '{col}' as dates")
                if result[0] is not None:
                    st.rerun()
        
        elif norm_op == "Normalize boolean values":
            col = st.selectbox("Column:", [c for c in df.columns if _is_string_col(df[c])], key="bool_col")
            st.info("Normalizes yes/no, y/n, true/false, 1/0, on/off → True/False")
            if st.button("🔄 Normalize Booleans", key="apply_bools"):
                try:
                    new_df = normalize_booleans(df, col)
                    record = ChangeRecord(
                        operation_name='normalize_booleans', parameters={'column': col},
                        affected_columns=[col], affected_rows_count=len(df),
                        before_metrics={}, after_metrics={}, validation_warnings=[],
                        accepted=False, df_before_snapshot_hash=''
                    )
                    st.session_state.pending_operation = f"Normalize booleans in '{col}'"
                    st.session_state.pending_df = new_df
                    st.session_state.pending_record = record
                    di.snapshots[record.df_before_snapshot_hash] = di.current_df.copy()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif norm_op == "Normalize categories":
            cat_cols = [c for c in df.columns if _is_string_col(df[c]) and df[c].nunique() <= 50]
            if cat_cols:
                col = st.selectbox("Column:", cat_cols, key="cat_col")
                st.markdown("**Current unique values:**")
                vc = df[col].value_counts()
                st.dataframe(vc.reset_index().rename(columns={'index': 'Value', col: 'Count'}), height=200)
                
                if st.button("🔄 Auto-normalize Categories", key="apply_cats"):
                    result = apply_and_preview('normalize_categories', 
                                               {'column': col, 'mapping': {}},
                                               f"Normalize categories in '{col}'")
                    if result[0] is not None:
                        st.rerun()
            else:
                st.info("No categorical columns found (columns with ≤50 unique values).")
    
    # ── Column Operations Tab ──
    with tab3:
        st.markdown("### Column Operations")
        
        col_op = st.selectbox("Operation:", [
            "Rename column", "Drop columns", "Reorder columns",
            "Split column", "Merge columns"
        ], key="col_operation")
        
        if col_op == "Rename column":
            old_name = st.selectbox("Column to rename:", df.columns.tolist(), key="rename_old")
            new_name = st.text_input("New name:", key="rename_new")
            if st.button("🔄 Rename", key="apply_rename") and new_name:
                result = apply_and_preview('rename_column', {'mapping': {old_name: new_name}},
                                           f"Rename '{old_name}' → '{new_name}'")
                if result[0] is not None:
                    st.rerun()
        
        elif col_op == "Drop columns":
            cols_to_drop = st.multiselect("Select columns to drop:", df.columns.tolist(), key="drop_cols")
            if st.button("🔄 Drop Selected", key="apply_drop") and cols_to_drop:
                result = apply_and_preview('drop_column', {'columns': cols_to_drop},
                                           f"Drop columns: {', '.join(cols_to_drop)}")
                if result[0] is not None:
                    st.rerun()
        
        elif col_op == "Reorder columns":
            st.info("Drag columns in the list below to reorder. Columns not listed will be appended at the end.")
            current_order = df.columns.tolist()
            new_order = st.multiselect("Column order (select in desired order):", current_order, 
                                        default=current_order, key="reorder_cols")
            if st.button("🔄 Apply Order", key="apply_reorder") and new_order:
                result = apply_and_preview('reorder_columns', {'columns': new_order},
                                           "Reorder columns")
                if result[0] is not None:
                    st.rerun()
        
        elif col_op == "Split column":
            col = st.selectbox("Column to split:", df.columns.tolist(), key="split_col")
            delimiter = st.text_input("Delimiter:", value=",", key="split_delim")
            if st.button("🔄 Split", key="apply_split"):
                result = apply_and_preview('split_column', {'column': col, 'separator': delimiter},
                                           f"Split '{col}' by '{delimiter}'")
                if result[0] is not None:
                    st.rerun()
        
        elif col_op == "Merge columns":
            cols_to_merge = st.multiselect("Columns to merge:", df.columns.tolist(), key="merge_cols")
            new_col_name = st.text_input("New column name:", value="merged_column", key="merge_name")
            separator = st.text_input("Separator:", value=" ", key="merge_sep")
            if st.button("🔄 Merge", key="apply_merge") and len(cols_to_merge) >= 2:
                result = apply_and_preview('merge_columns', 
                                           {'columns': cols_to_merge, 'new_column_name': new_col_name, 'separator': separator},
                                           f"Merge {', '.join(cols_to_merge)} → '{new_col_name}'")
                if result[0] is not None:
                    st.rerun()
    
    # ── Anomaly Detection Tab ──
    with tab4:
        st.markdown("### Anomaly Detection")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.info("No numeric columns found. Convert text columns to numeric first.")
        else:
            col = st.selectbox("Column:", numeric_cols, key="anomaly_col")
            method = st.selectbox("Detection method:", ["IQR", "Z-Score"], key="anomaly_method")
            threshold = st.slider("Threshold:", 1.0, 5.0, 1.5 if method == "IQR" else 3.0, 0.1, key="anomaly_thresh")
            
            if st.button("🔍 Detect Outliers", key="detect_outliers"):
                method_key = 'iqr' if method == 'IQR' else 'zscore'
                outliers = detect_numeric_outliers(df, col, method=method_key, threshold=threshold)
                
                if len(outliers) > 0:
                    st.warning(f"Found **{len(outliers)}** potential outlier(s) in '{col}'.")
                    st.dataframe(outliers.head(50), use_container_width=True)
                    st.caption("⚠️ These are statistical outliers, not necessarily errors. Review before removing.")
                else:
                    st.success(f"✅ No outliers detected in '{col}' with {method} method (threshold={threshold}).")
            
            # Distribution plot
            if col:
                fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=['#00d4aa'],
                                   title=f"Distribution of {col}")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#c8d6e5', height=300
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ── Change History Tab ──
    with tab5:
        st.markdown("### 📜 Change History")
        if not di.change_history:
            st.info("No changes applied yet. Use the cleaning tools to make changes.")
        else:
            for i, change in enumerate(reversed(di.change_history)):
                idx = len(di.change_history) - i
                with st.expander(f"#{idx} — {change.operation_name} ({change.timestamp[:19]})", expanded=(i == 0)):
                    st.markdown(f"**Operation:** {change.operation_name}")
                    st.markdown(f"**Parameters:** `{json.dumps(change.parameters)}`")
                    st.markdown(f"**Affected columns:** {', '.join(change.affected_columns)}")
                    st.markdown(f"**Rows affected:** {change.affected_rows_count:,}")
                    if change.validation_warnings:
                        for w in change.validation_warnings:
                            st.warning(f"⚠️ {w}")
            
            if st.button("↩️ Undo Last Change", key="undo_history"):
                if undo_last_change(di):
                    st.session_state.issues = detect_issues(di.current_df)
                    st.session_state.quality_after = compute_quality_score(di.current_df, di.original_df)
                    st.success("✅ Last change undone.")
                    st.rerun()
                else:
                    st.error("Cannot undo.")
    
    show_pending_operation()


# ─── Duplicate Detection ──────────────────────────────────────────────────────
def page_duplicates():
    st.markdown('<div class="section-header">🔗 Duplicate Detection</div>', unsafe_allow_html=True)
    if not ensure_dataset():
        return
    
    di = st.session_state.dataset_info
    df = di.current_df
    
    tab1, tab2 = st.tabs(["🎯 Exact Duplicates", "🔍 Fuzzy Duplicates"])
    
    with tab1:
        st.markdown("### Exact Duplicate Detection")
        
        use_subset = st.checkbox("Use specific columns for comparison", key="exact_subset")
        subset_cols = None
        if use_subset:
            subset_cols = st.multiselect("Select columns:", df.columns.tolist(), key="exact_cols")
        
        if st.button("🔍 Find Exact Duplicates", key="find_exact"):
            detector = DuplicateDetector()
            groups = detector.find_exact_duplicates(df, columns=subset_cols if subset_cols else None)
            
            if groups:
                st.warning(f"Found **{len(groups)}** duplicate group(s).")
                for i, group in enumerate(groups[:10]):
                    with st.expander(f"Group {i+1} — {len(group.indices)} records"):
                        comparison = detector.get_comparison_view(df, group)
                        st.dataframe(comparison, use_container_width=True)
                
                keep = st.selectbox("Keep:", ["First occurrence", "Last occurrence"], key="exact_keep")
                if st.button("🗑️ Remove Duplicates", key="remove_exact"):
                    keep_val = 'first' if keep == "First occurrence" else 'last'
                    result = apply_and_preview('remove_duplicates', {'keep': keep_val},
                                               f"Remove exact duplicates (keep {keep_val})")
                    if result[0] is not None:
                        st.rerun()
            else:
                st.success("✅ No exact duplicates found!")
    
    with tab2:
        st.markdown("### Fuzzy Duplicate Detection")
        st.info("Uses RapidFuzz similarity matching to find likely duplicates that aren't exact matches.")
        
        fuzzy_cols = st.multiselect("Columns to compare:", df.columns.tolist(), 
                                     default=df.columns.tolist()[:3] if len(df.columns) >= 3 else df.columns.tolist(),
                                     key="fuzzy_cols")
        threshold = st.slider("Similarity threshold (%):", 50, 100, 85, key="fuzzy_threshold")
        
        if st.button("🔍 Find Fuzzy Duplicates", key="find_fuzzy") and fuzzy_cols:
            with st.spinner("Analyzing potential duplicates..."):
                detector = DuplicateDetector(similarity_threshold=threshold)
                groups = detector.find_fuzzy_duplicates(df, columns=fuzzy_cols, threshold=threshold)
                st.session_state.duplicate_groups = groups
            
            if groups:
                summary = detector.get_duplicate_summary(df, groups)
                st.warning(f"Found **{summary['total_groups']}** potential duplicate group(s) "
                          f"({summary['total_duplicate_rows']} rows involved).")
            else:
                st.success(f"✅ No fuzzy duplicates found above {threshold}% similarity!")
        
        # Display fuzzy duplicate groups
        if st.session_state.duplicate_groups:
            detector = DuplicateDetector()
            for i, group in enumerate(st.session_state.duplicate_groups[:15]):
                with st.expander(f"Potential Match Group {i+1} — Similarity: {group.match_type}", expanded=(i < 3)):
                    comparison = detector.get_comparison_view(df, group)
                    st.dataframe(comparison, use_container_width=True)
                    
                    # Field-by-field comparison
                    if len(group.indices) >= 2:
                        idx1, idx2 = group.indices[0], group.indices[1]
                        if idx1 in df.index and idx2 in df.index:
                            details = detector.get_field_match_details(df, idx1, idx2, fuzzy_cols if fuzzy_cols else df.columns.tolist())
                            detail_df = pd.DataFrame(details)
                            st.markdown("**Field-by-field comparison:**")
                            st.dataframe(detail_df, use_container_width=True)
                    
                    resolve_col1, resolve_col2 = st.columns(2)
                    with resolve_col1:
                        if st.button(f"Keep first record", key=f"keep_first_{i}"):
                            try:
                                indices_to_drop = group.indices[1:]
                                valid_indices = [idx for idx in indices_to_drop if idx in df.index]
                                if valid_indices:
                                    new_df = df.drop(index=valid_indices)
                                    record = ChangeRecord(
                                        operation_name='resolve_duplicate', parameters={'group': i, 'action': 'keep_first'},
                                        affected_columns=df.columns.tolist(), affected_rows_count=len(valid_indices),
                                        before_metrics={}, after_metrics={}, validation_warnings=[],
                                        accepted=False, df_before_snapshot_hash=''
                                    )
                                    di.snapshots[record.df_before_snapshot_hash] = di.current_df.copy()
                                    st.session_state.pending_operation = f"Keep first record from group {i+1}"
                                    st.session_state.pending_df = new_df
                                    st.session_state.pending_record = record
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    with resolve_col2:
                        if st.button(f"Keep all (ignore)", key=f"keep_all_{i}"):
                            st.info("Marked as not a duplicate.")
    
    show_pending_operation()


# ─── Data Transformation ──────────────────────────────────────────────────────
def page_transformation():
    st.markdown('<div class="section-header">🔄 Data Transformation</div>', unsafe_allow_html=True)
    if not ensure_dataset():
        return
    
    di = st.session_state.dataset_info
    df = di.current_df
    
    st.markdown("### Current Schema")
    schema_data = [{"Column": col, "Type": str(df[col].dtype), "Non-Null": int(df[col].notna().sum())} 
                   for col in df.columns]
    st.dataframe(pd.DataFrame(schema_data), use_container_width=True, height=250)
    
    st.markdown("---")
    
    transform_op = st.selectbox("Transformation:", [
        "Wide to Long (Melt)", "Long to Wide (Pivot)",
        "Schema Standardization"
    ], key="transform_op")
    
    if transform_op == "Wide to Long (Melt)":
        id_cols = st.multiselect("ID columns (to keep):", df.columns.tolist(), key="melt_id")
        val_cols = st.multiselect("Value columns (to melt):", 
                                   [c for c in df.columns if c not in (id_cols or [])], key="melt_val")
        var_name = st.text_input("Variable column name:", value="variable", key="melt_var")
        value_name = st.text_input("Value column name:", value="value", key="melt_value")
        
        if st.button("🔄 Apply Melt", key="apply_melt") and id_cols and val_cols:
            try:
                from cleaning_engine import melt_wide_to_long
                new_df = melt_wide_to_long(df, id_cols, val_cols, var_name, value_name)
                record = ChangeRecord(
                    operation_name='melt_wide_to_long', 
                    parameters={'id_columns': id_cols, 'value_columns': val_cols},
                    affected_columns=df.columns.tolist(), affected_rows_count=len(df),
                    before_metrics={}, after_metrics={}, validation_warnings=[],
                    accepted=False, df_before_snapshot_hash=''
                )
                di.snapshots[record.df_before_snapshot_hash] = di.current_df.copy()
                st.session_state.pending_operation = "Wide to Long transformation"
                st.session_state.pending_df = new_df
                st.session_state.pending_record = record
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif transform_op == "Long to Wide (Pivot)":
        index_col = st.selectbox("Index column:", df.columns.tolist(), key="pivot_idx")
        columns_col = st.selectbox("Columns column:", [c for c in df.columns if c != index_col], key="pivot_col")
        values_col = st.selectbox("Values column:", [c for c in df.columns if c not in [index_col, columns_col]], key="pivot_val")
        
        if st.button("🔄 Apply Pivot", key="apply_pivot"):
            try:
                from cleaning_engine import pivot_long_to_wide
                new_df = pivot_long_to_wide(df, index_col, columns_col, values_col)
                record = ChangeRecord(
                    operation_name='pivot_long_to_wide',
                    parameters={'index': index_col, 'columns': columns_col, 'values': values_col},
                    affected_columns=df.columns.tolist(), affected_rows_count=len(df),
                    before_metrics={}, after_metrics={}, validation_warnings=[],
                    accepted=False, df_before_snapshot_hash=''
                )
                di.snapshots[record.df_before_snapshot_hash] = di.current_df.copy()
                st.session_state.pending_operation = "Long to Wide transformation"
                st.session_state.pending_df = new_df
                st.session_state.pending_record = record
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif transform_op == "Schema Standardization":
        st.markdown("Define your target column structure:")
        target_schema = st.text_area("Target columns (one per line):", 
                                      value="\n".join(df.columns.tolist()), height=200, key="target_schema")
        
        if st.button("🔄 Apply Schema", key="apply_schema") and target_schema:
            target_cols = [c.strip() for c in target_schema.strip().split('\n') if c.strip()]
            existing = [c for c in target_cols if c in df.columns]
            missing = [c for c in target_cols if c not in df.columns]
            
            if missing:
                st.warning(f"Columns not found (will be added as empty): {', '.join(missing)}")
            
            new_df = df.copy()
            for col in missing:
                new_df[col] = np.nan
            new_df = new_df[[c for c in target_cols if c in new_df.columns]]
            
            st.markdown("**Preview:**")
            st.dataframe(new_df.head(5), use_container_width=True)
            
            if st.button("✅ Confirm Schema", key="confirm_schema"):
                record = ChangeRecord(
                    operation_name='schema_standardization',
                    parameters={'target_columns': target_cols},
                    affected_columns=df.columns.tolist(), affected_rows_count=0,
                    before_metrics={}, after_metrics={}, validation_warnings=[],
                    accepted=False, df_before_snapshot_hash=''
                )
                di.snapshots[record.df_before_snapshot_hash] = di.current_df.copy()
                st.session_state.pending_operation = "Schema standardization"
                st.session_state.pending_df = new_df
                st.session_state.pending_record = record
                st.rerun()
    
    show_pending_operation()


# ─── Before/After Comparison ──────────────────────────────────────────────────
def page_comparison():
    st.markdown('<div class="section-header">📊 Before/After Comparison</div>', unsafe_allow_html=True)
    if not ensure_dataset():
        return
    
    di = st.session_state.dataset_info
    original = di.original_df
    current = di.current_df
    
    comparison = generate_comparison_data(original, current)
    qs_before = st.session_state.quality_before or compute_quality_score(original)
    qs_after = st.session_state.quality_after or compute_quality_score(current, original)
    
    # Quality Scores side by side
    st.markdown("### 📈 Quality Score")
    scol1, scol2, scol3 = st.columns([1, 1, 1])
    with scol1:
        st.markdown(f"""
        <div class="quality-score">
            <div class="score-label">BEFORE</div>
            <div class="score-value">{qs_before.get('overall_score', 0):.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with scol2:
        change = qs_after.get('overall_score', 0) - qs_before.get('overall_score', 0)
        arrow = "↑" if change > 0 else ("↓" if change < 0 else "→")
        color = "#00d4aa" if change > 0 else ("#ff4757" if change < 0 else "#8899aa")
        st.markdown(f"""
        <div style="text-align:center; padding:40px;">
            <span style="font-size:48px; color:{color};">{arrow}</span>
            <div style="color:{color}; font-size:20px; font-weight:700;">{change:+.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with scol3:
        st.markdown(f"""
        <div class="quality-score">
            <div class="score-label">AFTER</div>
            <div class="score-value">{qs_after.get('overall_score', 0):.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.caption("⚠️ This is a heuristic score based on completeness, uniqueness, consistency, and validity. "
               "It is not an absolute measure of correctness.")
    
    st.markdown("---")
    
    # Metrics comparison
    st.markdown("### 📋 Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{comparison['rows_cleaned']:,}", 
              f"{comparison['rows_cleaned'] - comparison['rows_original']:+,}")
    m2.metric("Columns", f"{comparison['cols_cleaned']}", 
              f"{comparison['cols_cleaned'] - comparison['cols_original']:+,}")
    m3.metric("Missing Values", f"{comparison['missing_cleaned']:,}",
              f"{-comparison['missing_change']:+,}")
    m4.metric("Duplicate Rows", f"{comparison['duplicates_cleaned']:,}",
              f"{comparison['duplicates_cleaned'] - comparison['duplicates_original']:+,}")
    
    st.markdown("---")
    
    # Component scores
    st.markdown("### 🧩 Quality Score Breakdown")
    comp_before = qs_before.get('component_scores', {})
    comp_after = qs_after.get('component_scores', {})
    
    if comp_before and comp_after:
        components = list(comp_before.keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Before', x=[k.title() for k in components],
            y=[comp_before.get(k, 0) * 100 for k in components],
            marker_color='#ff6b6b', opacity=0.8
        ))
        fig.add_trace(go.Bar(
            name='After', x=[k.title() for k in components],
            y=[comp_after.get(k, 0) * 100 for k in components],
            marker_color='#00d4aa', opacity=0.8
        ))
        fig.update_layout(
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#c8d6e5', height=350,
            yaxis_title='Score (%)', yaxis_range=[0, 105],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Change history summary
    st.markdown("### 📜 Operations Applied")
    if di.change_history:
        for i, change in enumerate(di.change_history, 1):
            st.markdown(f"**{i}.** {change.operation_name} — {change.affected_rows_count:,} rows affected "
                       f"({change.timestamp[:19]})")
    else:
        st.info("No cleaning operations applied yet.")
    
    # Remaining issues
    st.markdown("---")
    st.markdown("### ⚠️ Remaining Issues")
    if st.session_state.issues:
        for issue in st.session_state.issues:
            st.markdown(f"- {severity_badge(issue['severity'])} {issue['description']}", unsafe_allow_html=True)
    else:
        st.success("✅ No remaining issues detected!")


# ─── Cleaning Recipes ─────────────────────────────────────────────────────────
def page_recipes():
    st.markdown('<div class="section-header">📋 Cleaning Recipes</div>', unsafe_allow_html=True)
    
    rm = st.session_state.recipe_manager
    
    tab1, tab2, tab3 = st.tabs(["📋 Saved Recipes", "💾 Save Recipe", "📤 Apply Recipe"])
    
    with tab1:
        st.markdown("### Saved Recipes")
        recipes = rm.list_recipes()
        if recipes:
            for recipe in recipes:
                with st.expander(f"📋 {recipe.get('name', 'Unnamed')} (v{recipe.get('version', '?')})"):
                    st.markdown(f"**Description:** {recipe.get('description', 'N/A')}")
                    st.markdown(f"**Steps:** {recipe.get('step_count', '?')}")
                    st.markdown(f"**Created:** {recipe.get('creation_date', 'N/A')[:19]}")
                    
                    filepath = recipe.get('filepath', '')
                    if filepath:
                        if st.button(f"🗑️ Delete", key=f"del_{recipe.get('name', '')}"):
                            rm.delete_recipe(filepath)
                            st.rerun()
                        
                        # Download recipe
                        try:
                            loaded = rm.load_recipe(filepath)
                            recipe_json = rm.recipe_to_json(loaded)
                            st.download_button(
                                "📥 Download Recipe JSON",
                                data=recipe_json,
                                file_name=f"{loaded.name.replace(' ', '_')}_recipe.json",
                                mime="application/json",
                                key=f"download_{recipe.get('name', '')}",
                            )
                        except Exception as e:
                            st.error(f"Error loading recipe: {e}")
        else:
            st.info("No saved recipes. Create one from your cleaning history!")
    
    with tab2:
        st.markdown("### Save Current Cleaning as Recipe")
        if not st.session_state.dataset_info or not st.session_state.dataset_info.change_history:
            st.warning("⚠️ No cleaning history to save. Apply some cleaning operations first.")
        else:
            di = st.session_state.dataset_info
            recipe_name = st.text_input("Recipe name:", key="recipe_name")
            recipe_desc = st.text_area("Description:", key="recipe_desc")
            
            if st.button("💾 Save Recipe", key="save_recipe") and recipe_name:
                try:
                    recipe = rm.create_recipe_from_history(
                        change_history=di.change_history,
                        df_columns=di.original_df.columns.tolist(),
                        name=recipe_name,
                        description=recipe_desc
                    )
                    filepath = rm.save_recipe(recipe)
                    st.success(f"✅ Recipe saved: {filepath}")
                except Exception as e:
                    st.error(f"Error saving recipe: {e}")
    
    with tab3:
        st.markdown("### Apply a Recipe")
        if not ensure_dataset():
            return
        
        di = st.session_state.dataset_info
        
        upload_recipe = st.file_uploader("Upload recipe JSON:", type=['json'], key="upload_recipe")
        
        recipes = rm.list_recipes()
        saved_recipes = [r.get('name', 'Unnamed') for r in recipes]
        
        recipe_source = st.radio("Recipe source:", ["Uploaded file", "Saved recipe"], key="recipe_source")
        
        recipe_to_apply = None
        
        if recipe_source == "Uploaded file" and upload_recipe:
            try:
                recipe_json = upload_recipe.read().decode('utf-8')
                # Save temporarily and load
                temp_path = os.path.join(rm.recipes_dir, '_temp_recipe.json')
                with open(temp_path, 'w') as f:
                    f.write(recipe_json)
                recipe_to_apply = rm.load_recipe(temp_path)
                os.remove(temp_path)
            except Exception as e:
                st.error(f"Error loading recipe: {e}")
        
        elif recipe_source == "Saved recipe" and saved_recipes:
            selected = st.selectbox("Select recipe:", saved_recipes, key="select_recipe")
            if selected:
                for r in recipes:
                    if r.get('name') == selected:
                        try:
                            recipe_to_apply = rm.load_recipe(r['filepath'])
                        except Exception as e:
                            st.error(f"Error: {e}")
                        break
        
        if recipe_to_apply:
            st.markdown("---")
            st.markdown(f"### 📋 {recipe_to_apply.name}")
            st.markdown(f"**Description:** {recipe_to_apply.description}")
            st.markdown(f"**Steps:** {len(recipe_to_apply.steps)}")
            
            # Check compatibility
            compat = rm.check_compatibility(recipe_to_apply, di.current_df.columns.tolist())
            
            if not compat['compatible']:
                st.warning("⚠️ Schema mismatch detected!")
                if compat.get('missing_columns'):
                    st.markdown(f"**Missing columns:** {', '.join(compat['missing_columns'])}")
                if compat.get('incompatible_steps'):
                    st.markdown(f"**Incompatible steps:** {compat['incompatible_steps']}")
            
            # Preview steps
            preview = rm.get_recipe_preview(recipe_to_apply)
            st.code(preview, language="text")
            
            if st.button("▶️ Apply Recipe", type="primary", key="apply_recipe"):
                with st.spinner("Applying recipe..."):
                    success_count = 0
                    skip_count = 0
                    for step in recipe_to_apply.steps:
                        try:
                            if step.operation in [s for s in compat.get('incompatible_steps', [])]:
                                skip_count += 1
                                st.warning(f"⏩ Skipped step {step.step_number}: {step.operation} (incompatible)")
                                continue
                            
                            new_df, record = apply_operation(di, step.operation, step.params)
                            accept_change(di, new_df, record)
                            success_count += 1
                        except Exception as e:
                            skip_count += 1
                            st.warning(f"⏩ Skipped step {step.step_number}: {step.operation} ({str(e)})")
                    
                    st.session_state.issues = detect_issues(di.current_df)
                    st.session_state.quality_after = compute_quality_score(di.current_df, di.original_df)
                    st.success(f"✅ Recipe applied: {success_count} steps successful, {skip_count} skipped.")
                    st.rerun()


# ─── Export & Reports ──────────────────────────────────────────────────────────
def page_export():
    st.markdown('<div class="section-header">📥 Export & Reports</div>', unsafe_allow_html=True)
    if not ensure_dataset():
        return
    
    di = st.session_state.dataset_info
    df = di.current_df
    
    st.markdown("### 📁 Download Cleaned Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### CSV Export")
        sanitize_csv = st.checkbox("Sanitize for formula injection (CSV)", value=True, key="sanitize_csv")
        try:
            csv_bytes = export_clean_csv(df, sanitize=sanitize_csv)
            st.download_button(
                label="📥 Download CSV",
                data=csv_bytes,
                file_name=f"cleaned_{di.filename.rsplit('.', 1)[0]}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating CSV: {e}")
    
    with col2:
        st.markdown("#### Excel Export")
        sanitize_xlsx = st.checkbox("Sanitize for formula injection (Excel)", value=True, key="sanitize_xlsx")
        try:
            xlsx_bytes = export_clean_excel(df, sanitize=sanitize_xlsx)
            st.download_button(
                label="📥 Download Excel",
                data=xlsx_bytes,
                file_name=f"cleaned_{di.filename.rsplit('.', 1)[0]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating Excel: {e}")
    
    st.markdown("---")
    st.markdown("### 📊 PDF Quality Report")
    
    if st.button("📄 Generate PDF Report", type="primary", key="gen_pdf"):
        try:
            qs_before = st.session_state.quality_before or compute_quality_score(di.original_df)
            qs_after = st.session_state.quality_after or compute_quality_score(df, di.original_df)
            
            dataset_meta = {
                'name': di.filename,
                'rows': len(df),
                'columns': len(df.columns),
            }
            
            ops_list = []
            for ch in di.change_history:
                ops_list.append({
                    'operation': ch.operation_name,
                    'description': f"{ch.operation_name} — {ch.affected_rows_count} rows",
                })
            
            warnings_list = []
            for ch in di.change_history:
                warnings_list.extend(ch.validation_warnings)
            
            pdf_bytes = generate_pdf_report(
                dataset_info=dataset_meta,
                quality_before={
                    'missing_cells': di.original_df.isnull().sum().sum(),
                    'duplicate_rows': int(di.original_df.duplicated().sum()),
                    'quality_score': qs_before.get('overall_score', 0),
                },
                quality_after={
                    'missing_cells': int(df.isnull().sum().sum()),
                    'duplicate_rows': int(df.duplicated().sum()),
                    'quality_score': qs_after.get('overall_score', 0),
                },
                issues_detected=st.session_state.issues,
                operations_applied=ops_list,
                validation_warnings=warnings_list,
            )
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"quality_report_{di.filename.rsplit('.', 1)[0]}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.success("✅ PDF report generated!")
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
    
    st.markdown("---")
    st.markdown("### 📜 Cleaning History")
    
    if di.change_history:
        history_data = []
        for ch in di.change_history:
            history_data.append({
                'operation_id': ch.operation_id,
                'operation': ch.operation_name,
                'parameters': ch.parameters,
                'affected_columns': ch.affected_columns,
                'affected_rows': ch.affected_rows_count,
                'timestamp': ch.timestamp,
                'warnings': ch.validation_warnings,
            })
        
        history_json = json.dumps(history_data, indent=2, default=str)
        st.download_button(
            label="📥 Download Cleaning History (JSON)",
            data=history_json,
            file_name=f"cleaning_history_{di.filename.rsplit('.', 1)[0]}.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("No cleaning history to export.")
    
    st.markdown("---")
    st.markdown("### 📋 Save as Recipe")
    if di.change_history:
        st.info("💡 You can also save your cleaning workflow as a reusable recipe in the **📋 Cleaning Recipes** page.")
    
    st.markdown("---")
    st.markdown("""
    <div class="privacy-notice">
        🔒 <b>Data Privacy Notice:</b> All exports are generated locally in your browser.
        No data is sent to external servers during the export process.
        Exported files are sanitized to prevent formula injection attacks by default.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
page = st.session_state.page

if page == "🏠 Dashboard":
    page_dashboard()
elif page == "📤 Upload Dataset":
    page_upload()
elif page == "🔍 Data Profiler":
    page_profiler()
elif page == "🤖 AI Assistant":
    page_ai_assistant()
elif page == "🧹 Cleaning Workspace":
    page_cleaning_workspace()
elif page == "🔗 Duplicate Detection":
    page_duplicates()
elif page == "🔄 Data Transformation":
    page_transformation()
elif page == "📊 Before/After Comparison":
    page_comparison()
elif page == "📋 Cleaning Recipes":
    page_recipes()
elif page == "📥 Export & Reports":
    page_export()
