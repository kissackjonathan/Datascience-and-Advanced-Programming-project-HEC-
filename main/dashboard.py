"""
Political Stability Observatory - Interactive Dashboard
"""
# Resources:
# - YouTube tutorial: https://www.youtube.com/watch?v=6Eu2b34alsE&list=PLWuFHho1zKhWN-Qp5hrR0e9RZIo7QO7z6
#                   : https://www.youtube.com/watch?v=jWoqQ8lb778&list=PLWuFHho1zKhWN-Qp5hrR0e9RZIo7QO7z6&index=4

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# UTILITY: SUPPRESS STDOUT/STDERR FOR STREAMLIT COMPATIBILITY
# ============================================================================

# Redirect console outputs to prevent I/O errors in Streamlit
@contextmanager
def suppress_stdout_stderr():
    """
    Context manager to suppress stdout and stderr.
    Prevents I/O errors in Streamlit when Optuna tries to print.
    """
    # Save original stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        # Redirect to devnull
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        # Restore original stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# Page configuration
st.set_page_config(
    page_title="Political Stability Observatory",  # Browser tab title
    page_icon="",  # No icon in tab
    layout="wide",  # Full-width layout for better visualization
    initial_sidebar_state="expanded",  # Show sidebar by default
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Session state persists data across Streamlit reruns 

# Initialize session state for workflow tracking
if "workflow_step" not in st.session_state:
    st.session_state.workflow_step = 0  # Track workflow progress: 0=not started, 1=data loaded, 2=trained, 3=tested

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False  # Flag: True when train/test data loaded

if "models_trained" not in st.session_state:
    st.session_state.models_trained = False  # Flag: True when all 7 models trained

if "models_tested" not in st.session_state:
    st.session_state.models_tested = False  # Flag: True when models evaluated on test set

# Data storage in session state - avoids reloading heavy datasets on every interaction
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None

# Model results storage
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}  # Dict of trained model objects
if "training_results" not in st.session_state:
    st.session_state.training_results = None  # DataFrame with training metrics
if "test_results" not in st.session_state:
    st.session_state.test_results = None  # DataFrame with test set metrics

# Custom CSS
st.markdown(  # Custom CSS for UI styling
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .action-button {
        width: 100%;
        padding: 1rem;
        font-size: 1.1rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,  # Allow HTML/CSS injection
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Centralize all project paths to avoid duplication
def get_project_paths():
    """Get all project paths - centralizes path definitions to avoid hardcoding and enable easy project structure changes"""
    base = Path(__file__).parent.parent  # Navigate up to project root from dashboard.py
    return {
        "data_raw": base / "data" / "raw",  # Raw CSV/Excel files
        "data_processed": base / "data" / "processed",  # Cleaned train/test CSVs
        "models": base / "models",  # Saved model files (if any)
        "results": base / "results",  # Model results txt files
        "figures": base / "results" / "figures",  # PNG visualizations
    }


# ============================================================================
# DATA LOADING
# ============================================================================

# Cache data to prevent reloading on each interaction
@st.cache_data
def load_data():
    """Load all necessary data"""
    try:
        # Try to load full_data.csv first (generated by Action 1)
        # Use absolute path from project root
        data_path = project_root / "data" / "processed" / "full_data.csv"
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        # If full_data.csv doesn't exist, return None
        # User will need to run Action 1 in Project page first
        return None

# Load data specifically for the world map
@st.cache_data
def load_map_data():
    """Load data for world map - requires running Action 1 first to process raw data"""
    try:
        # Load the processed file (created by Action 1: Run Data Preparation)
        # Use absolute path from project root
        data_path = project_root / "data" / "processed" / "full_data.csv"
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        # File doesn't exist - user needs to run Action 1 first
        return None
    except Exception as e:
        # Other error
        return None


df = load_data()
df_map = load_map_data()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("#  Political Stability")
st.sidebar.markdown("## Observatory")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [" Home", " Explorer", " Project"], index=0)  # Radio buttons for page navigation

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    **Data Science & Advanced Programming**
    Master's Project 2025-2026

    **Dataset**: 1996-2023
    **Countries**: 166
    **Models**: 8 ML algorithms
    """
)

# ============================================================================
# PAGE 1: HOME
# ============================================================================

if page == " Home":
    st.markdown(
        "<h1 class='main-header'> Political Stability Observatory</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Introduction
    st.markdown(
        """
    ### Welcome to the Political Stability Observatory

    This interactive dashboard provides machine learning predictions for political stability across 166 countries,
    spanning from 1996 to 2023. Our analysis leverages 8 advanced ML algorithms to forecast stability using
    economic, governance, and development indicators.
    """
    )

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if df_map is not None and "Country Name" in df_map.columns:
            value = df_map["Country Name"].nunique()
        else:
            value = "N/A"
        st.metric(label=" Countries Analyzed", value=value)

    with col2:
        if df_map is not None and "Year" in df_map.columns:
            value = f"{int(df_map['Year'].min())}-{int(df_map['Year'].max())}"
        else:
            value = "N/A"
        st.metric(label=" Time Period", value=value)

    with col3:
        st.metric(label=" Best Model", value="Random Forest")

    with col4:
        value = f"{len(df_map):,}" if df_map is not None else "N/A"
        st.metric(label=" Total Observations", value=value)

    st.markdown("---")

    # Model Performance Comparison
    st.markdown("###  Model Performance Comparison")

    model_performance = {
        "Model": ["Random Forest", "XGBoost", "MLP", "KNN", "SVR"],
        "Test R^2": [0.7726, 0.7204, 0.6984, 0.6869, 0.6293],
        "Test RMSE": [0.4521, 0.5015, 0.5194, 0.5292, 0.5758],
    }

    perf_df = pd.DataFrame(model_performance)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(
            perf_df,
            x="Model",
            y="Test R^2",
            title="Model Performance (R^2 Score)",
            color="Test R^2",
            color_continuous_scale="Blues",
            text="Test R^2",
        )
        fig.update_traces(texttemplate="%{text:.2%}", textposition="inside")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("###  Top 3 Models")
        st.markdown(
            """
        **1. Random Forest**
        R^2 = 77.26%
        RMSE = 0.45

        **2. XGBoost**
        R^2 = 72.04%
        RMSE = 0.50

        **3. MLP Neural Network**
        R^2 = 69.84%
        RMSE = 0.52
        """
        )

    st.markdown("---")

    # World Map
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("###  Global Political Stability (Latest Year)")
    with col2:
        if st.button(" Reload Data", key="reload_data"):
            st.cache_data.clear()
            st.rerun()

    # Check if data is available with minimum required columns
    required_cols = ["Country Name", "Year", "political_stability"]

    if df_map is not None and all(col in df_map.columns for col in required_cols):
        # Get latest year data
        latest_year = int(df_map["Year"].max())
        df_latest = df_map[df_map["Year"] == latest_year].copy()

        # Year slider
        selected_year = st.slider(
            "Select Year",
            min_value=int(df_map["Year"].min()),  # Earliest year in dataset (1996)
            max_value=int(df_map["Year"].max()),  # Latest year in dataset (2023)
            value=latest_year,  # Default to most recent year
            step=1,  # Increment by 1 year
        )

        df_selected = df_map[df_map["Year"] == selected_year].copy()

        # Build hover_data dynamically based on available columns
        hover_data = {"political_stability": ":.3f"}  # Format: 3 decimal places
        if "Country Code" in df_selected.columns:
            hover_data["Country Code"] = False  # Hide country code in hover
        if "gdp_per_capita" in df_selected.columns:
            hover_data["gdp_per_capita"] = ":.0f"  # Format: no decimals
        if "hdi" in df_selected.columns:
            hover_data["hdi"] = ":.3f"  # Format: 3 decimal places
        if "unemployment" in df_selected.columns:
            hover_data["unemployment"] = ":.2f"  # Format: 2 decimal places

        # Create choropleth map - use Country Code if available, otherwise use Country Name
        if "Country Code" in df_selected.columns:
            fig = px.choropleth(
                df_selected,
                locations="Country Code",
                color="political_stability",
                hover_name="Country Name",
                hover_data=hover_data,
                color_continuous_scale="RdYlGn",
                range_color=[-3, 2],
                title=f"Political Stability Index - {selected_year}",
                labels={
                    "political_stability": "Stability Score",
                    "gdp_per_capita": "GDP per Capita",
                    "hdi": "HDI",
                    "unemployment": "Unemployment",
                },
            )
        else:
            # Fallback: use country names (less accurate but works)
            fig = px.choropleth(
                df_selected,
                locations="Country Name",
                locationmode="country names",
                color="political_stability",
                hover_name="Country Name",
                hover_data=hover_data,
                color_continuous_scale="RdYlGn",
                range_color=[-3, 2],
                title=f"Political Stability Index - {selected_year}",
                labels={
                    "political_stability": "Stability Score",
                    "gdp_per_capita": "GDP per Capita",
                    "hdi": "HDI",
                    "unemployment": "Unemployment",
                },
            )

        fig.update_layout(
            height=600,
            geo=dict(
                showframe=False, showcoastlines=True, projection_type="natural earth"
            ),
        )

        st.plotly_chart(fig, width="stretch")

        # Color scale explanation
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Red**: Low stability (< -1.0)")
        with col2:
            st.markdown("**Yellow**: Moderate stability (-1.0 to 0.5)")
        with col3:
            st.markdown("**Green**: High stability (> 0.5)")
    else:
        st.warning(
            """
        **Map data not available**

        The processed data file `data/processed/full_data.csv` is missing.

        **To display the map:**
        1. Go to the **Project** page (in the sidebar)
        2. Click **Action 1: Run Data Preparation**
        3. Wait for data processing to complete
        4. Come back to this page and refresh
        """
        )

# ============================================================================
# PAGE 2: EXPLORER
# ============================================================================

elif page == " Explorer":
    st.markdown(
        "<h1 class='main-header'> Country Explorer</h1>", unsafe_allow_html=True
    )
    st.markdown("---")

    if df is not None:
        # Country selector
        countries = sorted(df["Country Name"].unique())
        selected_country = st.selectbox(
            "Select a country",
            countries,
            index=countries.index("France") if "France" in countries else 0,
        )

        # Filter data for selected country
        df_country = df[df["Country Name"] == selected_country].sort_values("Year")

        # Header with current stability
        latest_stability = df_country.iloc[-1]["political_stability"]
        prev_stability = (
            df_country.iloc[-2]["political_stability"]
            if len(df_country) > 1
            else latest_stability
        )
        delta = latest_stability - prev_stability

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Current Stability", f"{latest_stability:.3f}", f"{delta:+.3f}")

        with col2:
            st.metric("Latest Year", int(df_country["Year"].max()))

        with col3:
            trend = "Improving" if delta > 0 else "Declining" if delta < 0 else "Stable"
            st.metric("Trend", trend)

        st.markdown("---")

        # Timeline
        st.markdown("###  Stability Timeline")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_country["Year"],
                y=df_country["political_stability"],
                mode="lines+markers",
                name="Political Stability",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8),
            )
        )

        fig.add_hline(
            y=0, line_dash="dash", line_color="gray", annotation_text="Neutral"
        )

        fig.update_layout(
            title=f"{selected_country} - Political Stability Over Time",
            xaxis_title="Year",
            yaxis_title="Stability Score",
            height=400,
            hovermode="x unified",
        )

        st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # Economic Indicators
        st.markdown("###  Economic & Governance Indicators")

        col1, col2 = st.columns(2)

        with col1:
            # GDP per capita
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(
                    x=df_country["Year"],
                    y=df_country["gdp_per_capita"],
                    mode="lines+markers",
                    name="GDP per capita",
                    fill="tozeroy",
                    line=dict(color="#2ecc71"),
                )
            )
            fig1.update_layout(
                title="GDP per Capita",
                xaxis_title="Year",
                yaxis_title="USD",
                height=300,
            )
            st.plotly_chart(fig1, width="stretch")

            # Unemployment
            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter(
                    x=df_country["Year"],
                    y=df_country["unemployment"],
                    mode="lines+markers",
                    name="Unemployment",
                    line=dict(color="#e74c3c"),
                )
            )
            fig3.update_layout(
                title="Unemployment Rate (ILO)",
                xaxis_title="Year",
                yaxis_title="%",
                height=300,
            )
            st.plotly_chart(fig3, width="stretch")

        with col2:
            # Rule of Law
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=df_country["Year"],
                    y=df_country["rule_of_law"],
                    mode="lines+markers",
                    name="Rule of Law",
                    line=dict(color="#9b59b6"),
                )
            )
            fig2.update_layout(
                title="Rule of Law Index",
                xaxis_title="Year",
                yaxis_title="Score",
                height=300,
            )
            st.plotly_chart(fig2, width="stretch")

            # HDI (only if available)
            if "hdi" in df_country.columns:
                fig4 = go.Figure()
                fig4.add_trace(
                    go.Scatter(
                        x=df_country["Year"],
                        y=df_country["hdi"],
                        mode="lines+markers",
                        name="HDI",
                        fill="tozeroy",
                        line=dict(color="#3498db"),
                    )
                )
                fig4.update_layout(
                    title="Human Development Index",
                    xaxis_title="Year",
                    yaxis_title="HDI",
                    height=300,
                )
                st.plotly_chart(fig4, width="stretch")

        st.markdown("---")

        # Data table
        with st.expander(" View Raw Data"):
            # Only include columns that exist
            display_cols = [
                "Year",
                "political_stability",
                "gdp_per_capita",
                "gdp_growth",
                "unemployment",
                "inflation",
                "rule_of_law",
            ]
            if "hdi" in df_country.columns:
                display_cols.append("hdi")
            # Filter to only existing columns
            display_cols = [col for col in display_cols if col in df_country.columns]
            st.dataframe(
                df_country[display_cols].sort_values("Year", ascending=False),
                width="stretch",
                height=400,
            )

# ============================================================================
# PAGE 3: PROJECT
# ============================================================================

elif page == " Project":
    st.markdown(
        "<h1 class='main-header'> Project Workflow</h1>", unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown(
        """
    ### Complete ML Pipeline
    Run the full machine learning workflow from data preparation to model evaluation.
    **Important**: Each step must be completed in order before proceeding to the next.
    """
    )

    # Workflow Progress Indicator
    st.markdown("####  Workflow Progress")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = "Completed" if st.session_state.data_loaded else "Pending"
        st.markdown(f"**Step 1: Load Data**\n\n{status}")

    with col2:
        status = "Completed" if st.session_state.models_trained else "Pending"
        st.markdown(f"**Step 2: Train Models**\n\n{status}")

    with col3:
        status = "Completed" if st.session_state.models_tested else "Pending"
        st.markdown(f"**Step 3: Test Models**\n\n{status}")

    with col4:
        can_visualize = st.session_state.models_tested
        status = "Ready" if can_visualize else "Pending"
        st.markdown(f"**Step 4: Visualize**\n\n{status}")

    # Reset workflow button
    if st.session_state.workflow_step > 0:
        if st.button(" Reset Workflow", key="reset_workflow"):
            st.session_state.workflow_step = 0
            st.session_state.data_loaded = False
            st.session_state.models_trained = False
            st.session_state.models_tested = False
            st.session_state.X_train = None
            st.session_state.X_test = None
            st.session_state.y_train = None
            st.session_state.y_test = None
            st.session_state.feature_names = None
            st.session_state.trained_models = {}
            st.session_state.training_results = None
            st.session_state.test_results = None
            st.success("Workflow reset! You can start from Step 1.")
            st.rerun()

    st.markdown("---")

    # ========================================================================
    # ACTION 0: CHECK ENVIRONMENT - Verify packages, folders, required files
    # ========================================================================

    st.markdown("##  Action 0: Check Environment")
    st.markdown(
        "Verify that all required files, packages, and directories are present."
    )

    if st.button(" Run Environment Check", key="check_env", width="stretch"):
        with st.spinner("Checking environment..."):
            # Import check function from tests
            project_root = Path(__file__).parent.parent
            sys.path.insert(0, str(project_root / "tests"))
            from test_structure import check_environment_test

            # Run environment check
            results = check_environment_test()

            # Display Python version
            st.subheader(" Python Version")
            if results["python"]["ok"]:
                st.success(f"Python {results['python']['message']}")
            else:
                st.error(f"Python {results['python']['message']} - Python 3.8+ required")

            # Display packages
            st.subheader(" Required Packages")
            pkg_data = []
            for pkg in results["packages"]["issues"]:
                pkg_data.append({
                    "Package": pkg["name"],
                    "Version": pkg["version"] if pkg["installed"] else "N/A",
                    "Status": " Installed" if pkg["installed"] else " NOT INSTALLED"
                })
            st.dataframe(pd.DataFrame(pkg_data), width="stretch", hide_index=True)

            # Display directories
            st.subheader(" Project Directories")
            dir_data = []
            for issue in results["directories"]["issues"]:
                dir_data.append({
                    "Directory": issue["name"],
                    "Path": issue["path"],
                    "Status": " EXISTS" if issue["exists"] else " CREATED"
                })
            st.dataframe(pd.DataFrame(dir_data), width="stretch", hide_index=True)

            # Display data files
            st.subheader(" Raw Data Files")
            file_data = []
            for issue in results["data_files"]["issues"]:
                file_data.append({
                    "File": issue["name"],
                    "Found": f"{issue['name']}{issue['extension']}" if issue["found"] else " NOT FOUND",
                    "Size (MB)": f"{issue['size_mb']:.2f}" if issue["found"] else "N/A",
                    "Status": " OK" if issue["found"] else " MISSING"
                })
            st.dataframe(pd.DataFrame(file_data), width="stretch", hide_index=True)

            # Summary
            st.markdown("---")
            if results["all_ok"]:
                st.success(" **Environment Check: ALL OK** - Ready to proceed!")
            else:
                st.error(" **Environment Check: ISSUES FOUND** - Please fix missing items.")
                if not results["packages"]["ok"]:
                    st.warning("Install missing packages: `pip install -r requirements.txt`")
                if not results["data_files"]["ok"]:
                    paths = get_project_paths()
                    st.warning(f"Download required datasets to: `{paths['data_raw']}`")

    st.markdown("---")

    # ========================================================================
    # ACTION 1: RUN DATA PREPARATION - Load train/test from data/processed
    # ========================================================================

    st.markdown("##  Action 1: Load Data")
    st.markdown("Load preprocessed training and test datasets.")

    # Always allow Action 1 (it's the first step)
    if st.button(" Load Data", key="data_prep", width="stretch"):
        with st.spinner("Loading data..."):
            try:
                import time
                from src.data_loader import load_processed_data

                paths = get_project_paths()
                processed_dir = paths["data_processed"]

                # Progressive loading with progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text(" Loading processed data...")
                time.sleep(0.3)  # Brief pause for UI update

                # Call data_loader function to load processed data
                data = load_processed_data(processed_dir, target="political_stability")

                # Extract data from returned dict
                X_train = data["X_train"]
                X_test = data["X_test"]
                y_train = data["y_train"]
                y_test = data["y_test"]
                train_data = data["df_train"]
                test_data = data["df_test"]

                feature_names = X_train.columns.tolist()
                progress_bar.progress(100)

                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.feature_names = feature_names
                st.session_state.data_loaded = True
                st.session_state.workflow_step = 1

                status_text.empty()
                progress_bar.empty()

                # Create full dataframe for EDA
                df_full = pd.concat([train_data, test_data])

                # Clear cache so Home page can reload the new data
                st.cache_data.clear()  # Force refresh of cached load_data() and load_map_data()

                # Display results
                st.success(" Data loaded successfully! Proceed to Action 2.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(" Training Samples", f"{len(X_train):,}")
                with col2:
                    st.metric(" Test Samples", f"{len(X_test):,}")
                with col3:
                    st.metric(" Total Samples", f"{len(df_full):,}")

                # Exploratory Data Analysis
                st.markdown("---")
                st.subheader(" Exploratory Data Analysis")

                # Statistiques descriptives
                with st.expander(" Descriptive Statistics"):
                    st.dataframe(
                        df_full.describe().T.style.background_gradient(
                            cmap="Blues"
                        ),
                        width="stretch",
                    )

                # Correlation matrix
                with st.expander(" Correlation Matrix"):
                    # Calculate correlation matrix
                    corr_matrix = df_full.select_dtypes(include=[np.number]).corr()  # Only numeric columns

                    # Create heatmap with plotly
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0,
                        title="Correlation Matrix of Features",
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, width="stretch")

                    # Highlight strong correlations
                    st.markdown("**Strong Correlations (|r| > 0.7):**")
                    strong_corr = []
                    for i in range(len(corr_matrix.columns)):  # Loop through all features
                        for j in range(i + 1, len(corr_matrix.columns)):  # Avoid duplicate pairs
                            if abs(corr_matrix.iloc[i, j]) > 0.7:  # Threshold for strong correlation
                                strong_corr.append(
                                    {
                                        "Feature 1": corr_matrix.columns[i],
                                        "Feature 2": corr_matrix.columns[j],
                                        "Correlation": corr_matrix.iloc[i, j],
                                    }
                                )
                    if strong_corr:
                        st.dataframe(
                            pd.DataFrame(strong_corr).style.background_gradient(
                                subset=["Correlation"],
                                cmap="RdYlGn",
                                vmin=-1,
                                vmax=1,
                            ),
                            width="stretch",
                            hide_index=True,
                        )
                    else:
                        st.info("No strong correlations found (|r| > 0.7)")

                # Distribution de la variable cible
                with st.expander(" Target Variable Distribution"):
                    col1, col2 = st.columns(2)

                    with col1:
                        # Histogram
                        fig1 = px.histogram(
                            df_full,
                            x="political_stability",
                            nbins=50,
                            title="Distribution of Political Stability",
                            labels={
                                "political_stability": "Political Stability Score"
                            },
                            color_discrete_sequence=["steelblue"],
                        )
                        fig1.add_vline(
                            x=df_full["political_stability"].mean(),
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Mean",
                        )
                        fig1.update_layout(height=400)
                        st.plotly_chart(fig1, width="stretch")

                    with col2:
                        # Box plot
                        fig2 = px.box(
                            df_full,
                            y="political_stability",
                            title="Box Plot of Political Stability",
                            labels={
                                "political_stability": "Political Stability Score"
                            },
                            color_discrete_sequence=["lightcoral"],
                        )
                        fig2.update_layout(height=400)
                        st.plotly_chart(fig2, width="stretch")

                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Mean", f"{df_full['political_stability'].mean():.3f}"
                        )
                    with col2:
                        st.metric(
                            "Std Dev", f"{df_full['political_stability'].std():.3f}"
                        )
                    with col3:
                        st.metric(
                            "Min", f"{df_full['political_stability'].min():.3f}"
                        )
                    with col4:
                        st.metric(
                            "Max", f"{df_full['political_stability'].max():.3f}"
                        )

            except FileNotFoundError as e:
                st.error(" Processed data files not found!")
                st.warning(
                    """
                    **Please run data preparation first:**

                    1. Open a terminal
                    2. Run: `python main/main.py`
                    3. Select option 1: "Run Data Preparation"
                    4. Come back here and click "Load Data" again
                    """
                )
            except Exception as e:
                st.error(f" Error loading data: {str(e)}")
                st.exception(e)

    st.markdown("---")

    # ========================================================================
    # ACTION 2: TRAIN MODELS - Train 7 ML models with Optuna
    # ========================================================================

    st.markdown("## Action 2: Train ML Models")
    st.markdown(
        "Train all 7 machine learning models (Random Forest, XGBoost, Gradient Boosting, Elastic Net, SVR, KNN, MLP)."
    )

    # Check if data is loaded
    if not st.session_state.data_loaded:
        st.warning(" Please complete Action 1 (Data Preparation) first!")
        st.button(
            " Train All Models",
            key="train_models",
            width="stretch",
            disabled=True,
        )
    else:
        if st.button(" Train All Models", key="train_models", width="stretch"):
            with st.spinner("Training 7 ML models with Optuna optimization..."):
                try:
                    # Import train_ml_models function from src/models.py
                    from src.models import train_ml_models

                    # Get data from session state
                    X_train = st.session_state.X_train
                    X_test = st.session_state.X_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test

                    # Get results directory
                    project_root = Path(__file__).parent.parent
                    results_dir = project_root / "results"

                    # Call existing train_ml_models function with console suppression
                    with suppress_stdout_stderr():  # Prevent Optuna print statements
                        model_results = train_ml_models(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            results_dir=results_dir,
                            cv=3,  # 3-fold cross-validation for speed in Streamlit
                            n_jobs=1,  # Sequential execution to avoid BrokenPipeError
                        )

                    # Convert results to dashboard format
                    trained_models = {}
                    results = []

                    for result in model_results:
                        if "error" not in result:  # Successful training
                            model_name = result["model"]
                            trained_models[model_name] = result["model_obj"]

                            results.append({
                                "Model": model_name,
                                "R^2 Score": result["r2"],
                                "RMSE": result["rmse"],
                                "MAE": result["mae"],
                                "Status": " Trained",
                            })
                        else:  # Training failed
                            results.append({
                                "Model": result["model"],
                                "R^2 Score": 0.0,
                                "RMSE": 0.0,
                                "MAE": 0.0,
                                "Status": f" Error: {result['error']}",
                            })

                    # Store models and results in session state
                    st.session_state.trained_models = trained_models
                    st.session_state.training_results = pd.DataFrame(results)
                    st.session_state.models_trained = True
                    st.session_state.workflow_step = 2

                    # Display results
                    results_df = st.session_state.training_results
                    st.success(
                        f" All {len(results)} models trained successfully! Proceed to Action 3."
                    )

                    st.subheader(" Model Performance")
                    st.dataframe(
                        results_df.style.background_gradient(
                            subset=["R^2 Score"], cmap="Greens"
                        ).format(
                            {"R^2 Score": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}"}
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    # Best model
                    best_model = results_df.loc[results_df["R^2 Score"].idxmax()]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(" Best Model", best_model["Model"])
                    with col2:
                        st.metric(" R^2 Score", f"{best_model['R^2 Score']:.4f}")
                    with col3:
                        st.metric(" RMSE", f"{best_model['RMSE']:.4f}")

                    # Interactive charts
                    st.markdown("---")
                    st.subheader(" Performance Comparison")

                    col1, col2 = st.columns(2)

                    with col1:
                        # R^2 Score comparison
                        fig1 = px.bar(
                            results_df,
                            x="Model",
                            y="R^2 Score",
                            title="R^2 Score by Model",
                            color="R^2 Score",
                            color_continuous_scale="Blues",
                            text="R^2 Score",
                        )
                        fig1.update_traces(
                            texttemplate="%{text:.4f}", textposition="inside"
                        )
                        fig1.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig1, width="stretch")

                    with col2:
                        # RMSE comparison
                        fig2 = px.bar(
                            results_df,
                            x="Model",
                            y="RMSE",
                            title="RMSE by Model",
                            color="RMSE",
                            color_continuous_scale="Reds_r",
                            text="RMSE",
                        )
                        fig2.update_traces(
                            texttemplate="%{text:.4f}", textposition="inside"
                        )
                        fig2.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig2, width="stretch")

                except Exception as e:
                    st.error(f" Error during model training: {str(e)}")
                    st.exception(e)

    st.markdown("---")

    # ========================================================================
    # ACTION 3: TEST MODELS - Evaluate trained models on test set
    # ========================================================================

    st.markdown("## Action 3: Test Models")
    st.markdown("Test all trained models on the test set.")

    # Check if models are trained
    if not st.session_state.models_trained:
        st.warning(" Please complete Action 2 (Train Models) first!")
        st.button(
            " Test All Models",
            key="test_models",
            width="stretch",
            disabled=True,
        )
    else:
        if st.button(" Test All Models", key="test_models", width="stretch"):
            with st.spinner("Testing models..."):
                try:
                    import time

                    from sklearn.metrics import (
                        mean_absolute_error,
                        mean_squared_error,
                        r2_score,
                    )

                    # Get data and models from session state
                    status_text = st.empty()
                    status_text.text(" Getting data and models from session...")
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test
                    trained_models = st.session_state.trained_models

                    if not trained_models:
                        st.warning(
                            " No trained models found in session. Please train models first (Action 2)."
                        )
                    else:
                        results = []
                        predictions = {}
                        progress_bar = st.progress(0)

                        for i, (model_name, model) in enumerate(
                            trained_models.items(), 1
                        ):
                            status_text.text(
                                f"[{i}/{len(trained_models)}] Testing {model_name}..."
                            )

                            # Generate predictions and calculate R²/RMSE/MAE metrics
                            y_pred = model.predict(X_test)

                            # Store predictions for visualization
                            predictions[model_name] = y_pred

                            # Calculate metrics
                            r2 = r2_score(y_test, y_pred)  # R² coefficient
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Square root of MSE
                            mae = mean_absolute_error(y_test, y_pred)  # Average absolute error

                            results.append(
                                {
                                    "Model": model_name,
                                    "R^2 Score": r2,
                                    "RMSE": rmse,
                                    "MAE": mae,
                                    "Test Samples": len(y_test),
                                }
                            )

                            progress_bar.progress(i / len(trained_models))
                            time.sleep(0.2)

                        # Store results and predictions in session state
                        st.session_state.test_results = pd.DataFrame(results)  # Test metrics table
                        st.session_state.predictions = predictions  # Dict of y_pred arrays
                        st.session_state.models_tested = True  # Enable Action 5 (visualizations)
                        st.session_state.workflow_step = 3  # Advance to final workflow stage

                        status_text.empty()
                        progress_bar.empty()

                        # Display results
                        results_df = st.session_state.test_results
                        st.success(
                            f" All {len(trained_models)} models tested successfully! Proceed to Action 5."
                        )

                    st.subheader(" Test Results")
                    st.dataframe(
                        results_df.style.background_gradient(
                            subset=["R^2 Score"], cmap="Greens"
                        ).format(
                            {"R^2 Score": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}"}
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    # Interactive comparison charts
                    st.markdown("---")
                    st.subheader(" Performance Comparison")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # R^2 Score
                        fig1 = px.bar(
                            results_df.sort_values("R^2 Score", ascending=False),
                            x="Model",
                            y="R^2 Score",
                            title="R^2 Score Comparison",
                            color="R^2 Score",
                            color_continuous_scale="Greens",
                            text="R^2 Score",
                        )
                        fig1.update_traces(
                            texttemplate="%{text:.4f}", textposition="inside"
                        )
                        fig1.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig1, width="stretch")

                    with col2:
                        # RMSE
                        fig2 = px.bar(
                            results_df.sort_values("RMSE"),
                            x="Model",
                            y="RMSE",
                            title="RMSE Comparison",
                            color="RMSE",
                            color_continuous_scale="Reds_r",
                            text="RMSE",
                        )
                        fig2.update_traces(
                            texttemplate="%{text:.4f}", textposition="inside"
                        )
                        fig2.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig2, width="stretch")

                    with col3:
                        # MAE
                        fig3 = px.bar(
                            results_df.sort_values("MAE"),
                            x="Model",
                            y="MAE",
                            title="MAE Comparison",
                            color="MAE",
                            color_continuous_scale="Oranges_r",
                            text="MAE",
                        )
                        fig3.update_traces(
                            texttemplate="%{text:.4f}", textposition="inside"
                        )
                        fig3.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig3, width="stretch")

                except Exception as e:
                    st.error(f" Error during model testing: {str(e)}")
                    st.exception(e)

    st.markdown("---")

    # ========================================================================
    # ACTION 4: EVALUATE MODELS - Display saved results from results/
    # ========================================================================

    st.markdown("##  Action 4: Evaluate Saved Models")
    st.markdown("View and compare all saved model results.")

    if st.button(
        " Evaluate All Models", key="evaluate_models", width="stretch"
    ):
        with st.spinner("Evaluating models..."):
            try:
                paths = get_project_paths()
                results_dir = paths["results"]

                # Check for result files (fixed filenames)
                benchmark_file = results_dir / "benchmark_results.csv"
                test_file = results_dir / "test_results.csv"

                if not benchmark_file.exists() and not test_file.exists():
                    st.warning(
                        " No result files found. Please train or test models first."
                    )
                else:
                    # Load and combine all results
                    all_results = []

                    if benchmark_file.exists():
                        st.subheader(" Benchmark Results")
                        df = pd.read_csv(benchmark_file)
                        df["Source"] = "Training (Benchmark)"
                        all_results.append(df)
                        st.dataframe(df, width="stretch", hide_index=True)

                    if test_file.exists():
                        st.subheader(" Test Results")
                        df = pd.read_csv(test_file)
                        df["Source"] = "Test Set"
                        all_results.append(df)
                        st.dataframe(df, width="stretch", hide_index=True)

                    # Create comprehensive summary
                    if all_results:
                        combined_df = pd.concat(all_results, ignore_index=True)

                        # Save summary (fixed filename, overwrites previous)
                        summary_file = results_dir / "evaluation_summary.csv"
                        combined_df.to_csv(summary_file, index=False)

                        st.markdown("---")
                        st.subheader(" Comprehensive Summary")
                        st.dataframe(
                            combined_df, width="stretch", hide_index=True
                        )

                        # Download button
                        st.markdown("####  Download Summary")
                        from datetime import datetime

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            " Download Evaluation Summary",
                            combined_df.to_csv(index=False).encode("utf-8"),
                            f"evaluation_summary_{timestamp}.csv",
                            "text/csv",
                            key="download-evaluation",
                        )

            except Exception as e:
                st.error(f" Error during model evaluation: {str(e)}")
                st.exception(e)

    st.markdown("---")

    # ========================================================================
    # ACTION 5: GENERATE VISUALIZATIONS - Interactive performance charts
    # ========================================================================

    st.markdown("##  Action 5: Generate Visualizations")
    st.markdown("View comprehensive model performance comparisons and predictions.")

    # Check if models are tested
    if not st.session_state.models_tested:
        st.warning(" Please complete Action 3 (Test Models) first!")
        st.button(
            " Generate All Visualizations",
            key="visualizations",
            width="stretch",
            disabled=True,
        )
    else:
        if st.button(
            " Generate All Visualizations",
            key="visualizations",
            width="stretch",
        ):
            with st.spinner("Generating visualizations from evaluation module..."):
                try:
                    # Import visualization functions from src.evaluation
                    from src.evaluation import (
                        generate_error_distribution,
                        generate_model_comparison_chart,
                        generate_predictions_plot,
                    )

                    # Get data from session state
                    test_results = st.session_state.test_results
                    trained_models = st.session_state.trained_models
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test

                    # Prepare models_data format expected by evaluation functions
                    models_data = []
                    for _, row in test_results.iterrows():
                        model_name = row["Model"]
                        model_obj = trained_models[model_name]
                        y_pred = model_obj.predict(X_test)

                        models_data.append({
                            "model": model_name,
                            "model_obj": model_obj,
                            "y_pred": y_pred,
                            "y_test": y_test,
                            "X_test": X_test,
                            "r2": row["R^2 Score"],
                            "rmse": row["RMSE"],
                            "mae": row["MAE"],
                        })

                    # Get output directory
                    project_root = Path(__file__).parent.parent
                    figures_dir = project_root / "results" / "figures"
                    figures_dir.mkdir(parents=True, exist_ok=True)

                    # Call visualization functions from evaluation module
                    st.info(" Generating model comparison chart...")
                    comparison_file = generate_model_comparison_chart(models_data, figures_dir)

                    st.info(" Generating predictions vs actual plots...")
                    predictions_file = generate_predictions_plot(models_data, figures_dir)

                    st.info(" Generating error distribution analysis...")
                    error_dist_file = generate_error_distribution(models_data, figures_dir)

                    st.success(" All visualizations generated successfully!")

                    # Display summary metrics
                    st.markdown("---")
                    st.subheader(" Model Performance Summary")

                    col1, col2, col3, col4 = st.columns(4)
                    best_model = test_results.loc[test_results["R^2 Score"].idxmax()]

                    with col1:
                        st.metric(" Best Model", best_model["Model"])
                    with col2:
                        st.metric(" Best R^2", f"{best_model['R^2 Score']:.4f}")
                    with col3:
                        st.metric(" Lowest RMSE", f"{test_results['RMSE'].min():.4f}")
                    with col4:
                        st.metric(" Lowest MAE", f"{test_results['MAE'].min():.4f}")

                    # Display performance table
                    st.dataframe(
                        test_results.style.background_gradient(
                            subset=["R^2 Score"], cmap="Greens"
                        )
                        .background_gradient(subset=["RMSE"], cmap="Reds_r")
                        .background_gradient(subset=["MAE"], cmap="Oranges_r")
                        .format(
                            {"R^2 Score": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}"}
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    # Display generated visualizations
                    st.markdown("---")
                    st.subheader(" Generated Visualizations")

                    if comparison_file and comparison_file.exists():
                        st.markdown("#### Model Comparison (R²/MAE/RMSE)")
                        st.image(str(comparison_file), width="stretch")

                    if predictions_file and predictions_file.exists():
                        st.markdown("#### Predictions vs Actual Values")
                        st.image(str(predictions_file), width="stretch")

                    if error_dist_file and error_dist_file.exists():
                        st.markdown("#### Error Distribution Analysis")
                        st.image(str(error_dist_file), width="stretch")

                    # ZIP download for all figures
                    st.markdown("---")
                    st.markdown("###  Download All Figures")

                    if figures_dir.exists() and any(figures_dir.glob("*.png")):
                        import io
                        import zipfile

                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for fig_file in figures_dir.glob("*.png"):
                                zip_file.write(fig_file, arcname=fig_file.name)

                        zip_buffer.seek(0)

                        st.download_button(
                            label=" Download All Figures (ZIP)",
                            data=zip_buffer,
                            file_name="political_stability_figures.zip",
                            mime="application/zip",
                            width="stretch",
                            type="primary"
                        )

                        st.info(" Click the button above to download all visualizations as a ZIP file")
                    else:
                        st.warning(" No figures found to download")

                except Exception as e:
                    st.error(f" Error generating visualizations: {str(e)}")
                    st.exception(e)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray; padding: 2rem 0;'>
    <p><b>Political Stability Observatory</b> | Master's Project 2025-2026</p>
    <p>Data Science & Advanced Programming | Machine Learning Analysis</p>
    <p>Dataset: World Bank, UNDP (1996-2023) | 166 Countries | 5 ML Models</p>
</div>
""",
    unsafe_allow_html=True,
)
