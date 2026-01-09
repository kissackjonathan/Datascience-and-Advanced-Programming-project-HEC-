"""
Terminal Interface for Political Stability Prediction

Provides CLI workflow for:
- Environment checking
- Data preparation
- Model training (ML + Panel)
- Model testing and evaluation
- Visualization generation
- Dashboard access
- Test coverage

Imports all helper functions from main.py
"""

import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import (
    # Session state
    WORKFLOW_COMPLETION,
    SESSION_MODELS_DATA,
    SESSION_TRAIN_DATA,
    SESSION_TEST_DATA,
    SESSION_FULL_DATA,
    SESSION_PANEL_METRICS,
    TRAINING_COMPLETED_THIS_SESSION,

    # Helpers
    Colors,
    get_project_paths,
    print_header,
    print_section,
    print_success,
    print_error,
    print_warning,
    print_info,
    pause,
    clear_screen,
)

# Professional Table Display

def display_professional_results_table(results: list, panel_metrics: Optional[dict] = None) -> None:
    """Display model results in a professional table format with box-drawing characters."""
    if not results and not panel_metrics:
        return

    all_results = []
    if results:
        all_results.extend(results)
    if panel_metrics:
        all_results.append({
            'model': 'Dynamic Panel (FE)',
            'r2': panel_metrics.get('r2', 0.0),
            'mae': panel_metrics.get('mae', float('nan')),
            'rmse': panel_metrics.get('rmse', 0.0)
        })

    def get_r2(x):
        """Extract R² value from model result dict, handling NaN and None values."""
        r2_val = x.get('r2', -1)
        if r2_val is None or (isinstance(r2_val, float) and np.isnan(r2_val)):
            return -1
        return r2_val

    all_results_sorted = sorted(all_results, key=get_r2, reverse=True)

    table_width = 85
    print(f"\n{Colors.BOLD}{'=' * table_width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'MODEL PERFORMANCE COMPARISON':^{table_width}}{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * table_width}{Colors.RESET}")
    print(f"{Colors.BOLD}┌{'─' * 6}┬{'─' * 25}┬{'─' * 16}┬{'─' * 16}┬{'─' * 16}┐{Colors.RESET}")
    print(f"{Colors.BOLD}│ {'Rank':^4} │ {'Model':^23} │ {'R² Score':^14} │ {'MAE':^14} │ {'RMSE':^14} │{Colors.RESET}")
    print(f"{Colors.BOLD}├{'─' * 6}┼{'─' * 25}┼{'─' * 16}┼{'─' * 16}┼{'─' * 16}┤{Colors.RESET}")

    for rank, result in enumerate(all_results_sorted, 1):
        model_name = result.get('model', 'Unknown')
        r2 = result.get('r2', float('nan'))
        mae = result.get('mae', float('nan'))
        rmse = result.get('rmse', float('nan'))

        r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
        mae_str = f"{mae:.4f}" if not np.isnan(mae) else "N/A"
        rmse_str = f"{rmse:.4f}" if not np.isnan(rmse) else "N/A"

        print(f"│ {rank:^4} │ {model_name:<23} │ {r2_str:^14} │ {mae_str:^14} │ {rmse_str:^14} │")
    print(f"{Colors.BOLD}└{'─' * 6}┴{'─' * 25}┴{'─' * 16}┴{'─' * 16}┴{'─' * 16}┘{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * table_width}{Colors.RESET}\n")

    if all_results_sorted:
        best = all_results_sorted[0]
        best_model = best.get('model', 'Unknown')
        best_r2 = best.get('r2', 0.0)
        best_mae = best.get('mae', float('nan'))
        best_rmse = best.get('rmse', float('nan'))

        print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}Best Model: {best_model}{Colors.RESET}")
        print(f"{Colors.BOLD}R² = {best_r2:.4f}{Colors.RESET}")
        if not np.isnan(best_mae):
            print(f"{Colors.BOLD}MAE = {best_mae:.4f}, RMSE = {best_rmse:.4f}{Colors.RESET}\n")
        else:
            print(f"{Colors.BOLD}RMSE = {best_rmse:.4f}{Colors.RESET}\n")


# ACTION 0: Check Environment

def check_environment() -> None:
    """Verify Python version, packages, directories, and data files."""
    print_section("ACTION 0: CHECK ENVIRONMENT")

    sys.path.insert(0, str(get_project_paths()["root"] / "tests"))
    from test_structure import check_environment_test

    results = check_environment_test()

    print(f"{Colors.BRIGHT_GREEN}Python Version:{Colors.RESET}")
    if results["python"]["ok"]:
        print_success(f"Python {results['python']['message']}")
    else:
        print_error("Python version < 3.8")

    print(f"\n{Colors.BRIGHT_GREEN}Required Packages:{Colors.RESET}")
    for pkg in results["packages"]["issues"]:
        if pkg["installed"]:
            print_success(f"{pkg['name']} ({pkg['version']})")
        else:
            print_error(f"{pkg['name']} - NOT INSTALLED")

    print(f"\n{Colors.BRIGHT_GREEN}Project Structure:{Colors.RESET}")
    if results["directories"]["ok"]:
        print_success("All required directories exist")
    else:
        for issue in results["directories"]["issues"]:
            print_error(issue)

    print(f"\n{Colors.BRIGHT_GREEN}Data Files:{Colors.RESET}")
    if results["data_files"]["ok"]:
        print_success("All required data files present")
    else:
        for issue in results["data_files"]["issues"]:
            print_warning(issue)

    if not results["all_ok"]:
        paths = get_project_paths()
        print(f"\n{Colors.YELLOW}Setup Instructions:{Colors.RESET}")
        print(f"  {Colors.CYAN}pip install -r requirements.txt{Colors.RESET}")
        print(f"  {Colors.DIM}Download required datasets to: {paths['data_raw']}{Colors.RESET}")

    global WORKFLOW_COMPLETION
    WORKFLOW_COMPLETION["check_environment"] = True

    pause()


# ACTION 1: Run Data Preparation

def run_data_preparation() -> None:
    """Load, clean, merge indicators, save train/test/full datasets (1996-2017 train, 2018-2023 test)."""
    print_section("ACTION 1: RUN DATA PREPARATION")

    try:
        from src.data_loader import load_data

        paths = get_project_paths()
        processed_dir = paths["data_processed"]
        processed_dir.mkdir(parents=True, exist_ok=True)

        print_info("Loading and processing data (load -> clean -> merge -> save)...")
        data_dict = load_data(data_path=paths["data_raw"], target="political_stability", train_end_year=2017)

        X_train, X_test = data_dict["X_train"], data_dict["X_test"]
        y_train, y_test = data_dict["y_train"], data_dict["y_test"]

        train_data = X_train.copy()
        train_data["political_stability"] = y_train
        train_data.to_csv(processed_dir / "train_data.csv", index=True)

        test_data = X_test.copy()
        test_data["political_stability"] = y_test
        test_data.to_csv(processed_dir / "test_data.csv", index=True)

        full_data = pd.concat([train_data, test_data])
        full_data.to_csv(processed_dir / "full_data.csv", index=True)

        print_success("Data preparation completed")
        print(f"\n{Colors.CYAN}Datasets:{Colors.RESET} train ({len(train_data)}), test ({len(test_data)}), features ({len(X_train.columns)})")

    except (ImportError, FileNotFoundError, Exception) as e:
        error_type = type(e).__name__
        print_error(f"{error_type}: {str(e)}")
        if isinstance(e, FileNotFoundError):
            print(f"{Colors.CYAN}Run [0] Check Environment to verify data files{Colors.RESET}")
        pause()
        return

    global WORKFLOW_COMPLETION
    WORKFLOW_COMPLETION["data_preparation"] = True
    pause()


# ACTION 2: Train Model

def train_model() -> None:
    """Train all models (7 ML + Panel). Requires data preparation first."""
    print_section("ACTION 2: TRAIN MODEL")

    global WORKFLOW_COMPLETION
    if not WORKFLOW_COMPLETION["data_preparation"]:
        print_error("You must prepare data first before training models!")
        print_info("Run [1] Data Preparation first")
        pause()
        return

    train_all_ml_models()


def train_all_ml_models():
    """Train 7 ML models + Dynamic Panel Analysis."""
    from src.models import train_ml_models

    print_section("TRAINING ALL MODELS")

    paths = get_project_paths()
    processed_dir = paths["data_processed"]
    results_dir = paths["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_data = pd.read_csv(processed_dir / "train_data.csv", index_col=[0, 2])
        test_data = pd.read_csv(processed_dir / "test_data.csv", index_col=[0, 2])

        train_data = train_data.drop("Country Code", axis=1, errors='ignore')
        test_data = test_data.drop("Country Code", axis=1, errors='ignore')

        X_train = train_data.drop("political_stability", axis=1)
        y_train = train_data["political_stability"]
        X_test = test_data.drop("political_stability", axis=1)
        y_test = test_data["political_stability"]
    except Exception as e:
        print_error(f"Failed to load data: {str(e)}")
        print_info("Run [1] Data Preparation first")
        pause()
        return

    print_info("Training 7 ML models (this may take several minutes)...\n")
    results = train_ml_models(X_train, y_train, X_test, y_test, results_dir)

    print_info("Training Dynamic Panel model (econometric model with fixed effects)...\n")
    panel_metrics = train_dynamic_panel(skip_input=True)

    display_professional_results_table(results, panel_metrics)

    if results:
        results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
        results_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results_df["train_samples"] = len(X_train)
        results_df["test_samples"] = len(X_test)
        results_df.to_csv(results_dir / "benchmark_results.csv", index=False)

    global TRAINING_COMPLETED_THIS_SESSION, WORKFLOW_COMPLETION, SESSION_MODELS_DATA, SESSION_TRAIN_DATA, SESSION_TEST_DATA, SESSION_FULL_DATA
    TRAINING_COMPLETED_THIS_SESSION = True
    WORKFLOW_COMPLETION["train_model"] = True
    SESSION_MODELS_DATA = results
    SESSION_TRAIN_DATA = train_data
    SESSION_TEST_DATA = test_data

    processed_dir = paths["data_processed"]
    full_data_file = processed_dir / "full_data.csv"
    if full_data_file.exists():
        SESSION_FULL_DATA = pd.read_csv(full_data_file)
    else:
        SESSION_FULL_DATA = pd.concat([train_data, test_data])

    print_success("\nTraining completed!")
    pause()


def display_nickell_bias(bias_metrics: Dict[str, float]) -> None:
    """Display Nickell (1981) bias analysis for dynamic panel models with fixed effects."""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_WHITE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_YELLOW}NICKELL BIAS ANALYSIS (Econometric Diagnostic){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{'='*80}{Colors.RESET}")

    print(f"\n{Colors.BRIGHT_CYAN}Panel Structure:{Colors.RESET}")
    print(f"  N (countries): {bias_metrics['N']}")
    print(f"  T (time periods): {bias_metrics['T']}")
    print(f"\n{Colors.BRIGHT_CYAN}Autoregressive Coefficient (α):{Colors.RESET}")
    print(f"  α̂_OLS (biased):          {bias_metrics['alpha_ols']:.4f}")
    print(f"  α̂* (bias-corrected):     {bias_metrics['alpha_corrected']:.4f}")

    bias_val = bias_metrics['bias']
    bias_pct = bias_metrics['bias_percentage']

    print(f"\n{Colors.BRIGHT_CYAN}Nickell Bias (1981):{Colors.RESET}")
    print(f"  Formula: Bias(α̂) = -(1+α)/(T-1)")
    print(f"  Bias(α̂) = {bias_val:.4f}")
    print(f"  Relative: {bias_pct:.1f}% of OLS estimate")

    interpretation = bias_metrics['interpretation']
    if interpretation == "NEGLIGIBLE":
        severity_color = Colors.BRIGHT_GREEN
        severity_icon = "[OK]"
    elif interpretation == "MODERATE":
        severity_color = Colors.YELLOW
        severity_icon = "[WARNING]"
    else:  # SEVERE
        severity_color = Colors.RED
        severity_icon = "[ERROR]"

    print(f"\n{Colors.BRIGHT_CYAN}Bias Severity:{Colors.RESET} {severity_color}{severity_icon} {interpretation}{Colors.RESET}")

    if not np.isnan(bias_metrics['half_life_ols']) and not np.isnan(bias_metrics['half_life_corrected']):
        hl_ols = bias_metrics['half_life_ols']
        hl_corr = bias_metrics['half_life_corrected']
        hl_diff = hl_corr - hl_ols
        hl_diff_pct = (hl_diff / hl_ols) * 100

        print(f"\n{Colors.BRIGHT_CYAN}Shock Persistence (Half-life):{Colors.RESET}")
        print(f"  OLS estimate:        {hl_ols:.2f} years")
        print(f"  Corrected estimate:  {hl_corr:.2f} years")
        print(f"  Difference:          {hl_diff:+.2f} years ({hl_diff_pct:+.1f}%)")
        print(f"  {Colors.DIM}> Shocks to political stability decay by 50% in ~{hl_corr:.1f} years{Colors.RESET}")

    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{'='*80}{Colors.RESET}\n")


def display_diagnostic_tests(diagnostics: Dict[str, Dict]) -> None:
    """Display econometric diagnostic tests (Hausman, autocorrelation, heteroscedasticity)."""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_WHITE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_YELLOW}ECONOMETRIC DIAGNOSTIC TESTS{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{'='*80}{Colors.RESET}")

    # Test 1: Hausman Test
    if "hausman" in diagnostics:
        hausman = diagnostics["hausman"]
        print(f"\n{Colors.BRIGHT_CYAN}1. HAUSMAN TEST (Fixed Effects vs Random Effects){Colors.RESET}")
        print(f"{Colors.DIM}{'-'*80}{Colors.RESET}")

        if "error" in hausman:
            print(f"  {Colors.RED}Test failed: {hausman['error']}{Colors.RESET}")
            print(f"  {Colors.DIM}Using Fixed Effects (standard for panel data){Colors.RESET}")
        else:
            print(f"  Chi² statistic: {hausman['statistic']:.4f}")
            print(f"  Degrees of freedom: {hausman['df']}")
            print(f"  P-value: {hausman['p_value']:.6f}")

            conclusion = hausman['conclusion']
            if "FE" in conclusion or "reject" in conclusion.lower():
                color = Colors.BRIGHT_GREEN
                icon = "[OK]"
            else:
                color = Colors.YELLOW
                icon = "[!]"
            print(f"  Conclusion: {color}{icon} {conclusion}{Colors.RESET}")

    # Test 2: Autocorrelation Test
    if "autocorrelation" in diagnostics:
        autocorr = diagnostics["autocorrelation"]
        print(f"\n{Colors.BRIGHT_CYAN}2. AUTOCORRELATION TEST (AR1){Colors.RESET}")
        print(f"{Colors.DIM}{'-'*80}{Colors.RESET}")

        if "error" in autocorr:
            print(f"  {Colors.RED}Test failed: {autocorr['error']}{Colors.RESET}")
            print(f"  {Colors.DIM}Using clustered SE as precaution{Colors.RESET}")
        else:
            print(f"  Rho (AR1 coefficient): {autocorr['rho']:.4f}")
            print(f"  P-value: {autocorr['p_value']:.6f}")

            conclusion = autocorr['conclusion']
            if "No evidence" in conclusion:
                color = Colors.BRIGHT_GREEN
                icon = "[OK]"
            else:
                color = Colors.YELLOW
                icon = "[!]"
            print(f"  Conclusion: {color}{icon} {conclusion}{Colors.RESET}")
            print(f"  {Colors.DIM}Note: Model uses clustered SE to handle potential autocorrelation{Colors.RESET}")

    # Test 3: Heteroscedasticity Test
    if "heteroscedasticity" in diagnostics:
        hetero = diagnostics["heteroscedasticity"]
        print(f"\n{Colors.BRIGHT_CYAN}3. HETEROSCEDASTICITY TEST (Breusch-Pagan){Colors.RESET}")
        print(f"{Colors.DIM}{'-'*80}{Colors.RESET}")

        if "error" in hetero:
            print(f"  {Colors.RED}Test failed: {hetero['error']}{Colors.RESET}")
            print(f"  {Colors.DIM}Using robust SE as precaution{Colors.RESET}")
        else:
            print(f"  LM statistic: {hetero['lm_stat']:.4f}")
            print(f"  LM p-value: {hetero['lm_pvalue']:.6f}")
            print(f"  F statistic: {hetero['f_stat']:.4f}")
            print(f"  F p-value: {hetero['f_pvalue']:.6f}")

            conclusion = hetero['conclusion']
            if "No evidence" in conclusion:
                color = Colors.BRIGHT_GREEN
                icon = "[OK]"
            else:
                color = Colors.YELLOW
                icon = "[!]"
            print(f"  Conclusion: {color}{icon} {conclusion}{Colors.RESET}")
            print(f"  {Colors.DIM}Note: Model uses clustered SE to handle potential heteroscedasticity{Colors.RESET}")

    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{'='*80}{Colors.RESET}\n")


def train_dynamic_panel(skip_input: bool = False) -> Optional[Dict[str, float]]:
    """Train Dynamic Panel Fixed Effects model. Returns {r2, rmse, mae, time} or None."""
    from src.models import PanelAnalyzer
    global SESSION_PANEL_METRICS

    paths = get_project_paths()
    processed_dir = paths["data_processed"]
    results_dir = paths["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.data_loader import load_data

        data_dict = load_data(
            data_path=paths["root"] / "data" / "raw",
            target="political_stability",
            train_end_year=2017,
        )

        df_train = data_dict["df_train"].copy()
        df_test = data_dict["df_test"].copy()
        full_data = pd.concat([df_train, df_test])

        if isinstance(df_train.index, pd.MultiIndex):
            df_train = df_train.reset_index()
        if isinstance(df_test.index, pd.MultiIndex):
            df_test = df_test.reset_index()
        if isinstance(full_data.index, pd.MultiIndex):
            full_data = full_data.reset_index()

        if "Country Name" not in full_data.columns or "Year" not in full_data.columns:
            raise ValueError("Missing required columns: Country Name or Year")

    except Exception as e:
        print_error(f"Failed to load data: {str(e)}")
        if not skip_input:
            pause()
        return None

    temp_csv = paths["data_processed"] / "temp_panel_data.csv"
    full_data.to_csv(temp_csv, index=False)

    start_time = time.time()

    try:
        predictors = [
            col for col in full_data.columns
            if col not in ["political_stability", "Country Name", "Year", "Country Code"]
        ]

        analyzer = PanelAnalyzer(
            data_path=str(temp_csv),
            target="political_stability",
            entity_col="Country Name",
            time_col="Year",
            predictors=predictors,
        )

        analyzer.load_data().prepare_panel_structure().train_test_split(test_years=6)
        analyzer.fit_dynamic_panel()
        elapsed = time.time() - start_time

        results_file = results_dir / "panel_analysis.txt"
        with open(results_file, 'w') as f:
            f.write(str(analyzer.dynamic_results.summary))

        test_metrics = analyzer.evaluate_on_test()

        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        r2 = test_metrics["r2"]
        rmse = test_metrics["rmse"]
        mae = test_metrics["mae"]

        nickell_bias_metrics = analyzer.calculate_nickell_bias()
        diagnostic_tests = analyzer.run_diagnostic_tests()

        if not skip_input:
            display_nickell_bias(nickell_bias_metrics)
            display_diagnostic_tests(diagnostic_tests)

        panel_metrics = {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "time": elapsed,
            "nickell_bias": nickell_bias_metrics,
            "diagnostics": diagnostic_tests,
        }

        SESSION_PANEL_METRICS = panel_metrics
        return panel_metrics

    except Exception as e:
        print_error(f"Panel training failed: {str(e)}")
        if not skip_input:
            pause()
        return None


# ACTION 3: Test Model

def test_model():
    """Display test results from all trained models (in-memory, no .pkl files)."""
    print_section("ACTION 3: TEST MODEL")

    global TRAINING_COMPLETED_THIS_SESSION, SESSION_MODELS_DATA
    if not TRAINING_COMPLETED_THIS_SESSION or SESSION_MODELS_DATA is None:
        print_error("You must TRAIN models in THIS SESSION before viewing test results!")
        print_info("Run [2] Train Model first")
        pause()
        return

    test_all_models()


def display_overfitting_analysis(models_data: list) -> None:
    """Display overfitting analysis for ML models (Train vs Test R²)."""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_WHITE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_YELLOW}OVERFITTING ANALYSIS (ML Models Diagnostic){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{'='*80}{Colors.RESET}")

    print(f"\n{Colors.BOLD}{'Model':<25} {'Train R²':>10} {'Test R²':>10} {'Gap':>10} {'Status':>15}{Colors.RESET}")
    print(f"{Colors.DIM}{'-'*80}{Colors.RESET}")

    for model in models_data:
        if "error" in model:
            continue

        model_name = model.get("model", "Unknown")
        train_r2 = model.get("train_score", np.nan)
        test_r2 = model.get("r2", np.nan)

        if np.isnan(train_r2) or np.isnan(test_r2):
            continue

        gap = train_r2 - test_r2
        gap_pct = (gap / train_r2 * 100) if train_r2 != 0 else 0

        if gap < 0.05:
            status_color = Colors.BRIGHT_GREEN
            status_text = "EXCELLENT"
            status_icon = "[OK]"
        elif gap < 0.10:
            status_color = Colors.YELLOW
            status_text = "ACCEPTABLE"
            status_icon = "[!]"
        else:
            status_color = Colors.RED
            status_text = "OVERFITTING"
            status_icon = "[X]"

        print(f"{model_name:<25} {train_r2:>10.4f} {test_r2:>10.4f} {gap:>9.4f}  {status_color}{status_icon} {status_text}{Colors.RESET}")

    print(f"\n{Colors.DIM}Legend:{Colors.RESET}")
    print(f"  {Colors.BRIGHT_GREEN}[OK] EXCELLENT{Colors.RESET}  : Gap < 0.05  (Model generalizes very well)")
    print(f"  {Colors.YELLOW}[!] ACCEPTABLE{Colors.RESET} : Gap < 0.10  (Slight overfitting, still usable)")
    print(f"  {Colors.RED}[X] OVERFITTING{Colors.RESET}: Gap >= 0.10 (Model memorizes training data)")
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{'='*80}{Colors.RESET}\n")


def test_all_models():
    """Display test results from in-memory trained models."""
    global SESSION_MODELS_DATA, SESSION_TEST_DATA, SESSION_PANEL_METRICS

    print(f"\n{Colors.BOLD}{Colors.BRIGHT_GREEN}TEST RESULTS:{Colors.RESET}\n")

    successful_models = [m for m in SESSION_MODELS_DATA if "error" not in m and "r2" in m]
    failed_models = [m for m in SESSION_MODELS_DATA if "error" in m]

    results = [
        {
            "model": m["model"],
            "r2": m["r2"],
            "adj_r2": m.get("adj_r2", m["r2"]),
            "mae": m["mae"],
            "rmse": m["rmse"],
        }
        for m in successful_models
    ]

    if not results and not SESSION_PANEL_METRICS:
        print_error("No models trained successfully!")
        return

    display_professional_results_table(results, SESSION_PANEL_METRICS)

    if failed_models:
        print(f"\n{Colors.BOLD}{Colors.BRIGHT_RED}FAILED MODELS:{Colors.RESET}")
        for m in failed_models:
            print(f"{Colors.RED}  {m['model']}: {m.get('error', 'Unknown error')}{Colors.RESET}")

    display_overfitting_analysis(SESSION_MODELS_DATA)

    from src.evaluation import display_feature_importance_table
    display_feature_importance_table(SESSION_MODELS_DATA)

    if SESSION_PANEL_METRICS and "nickell_bias" in SESSION_PANEL_METRICS:
        display_nickell_bias(SESSION_PANEL_METRICS["nickell_bias"])

    if SESSION_PANEL_METRICS and "diagnostics" in SESSION_PANEL_METRICS:
        display_diagnostic_tests(SESSION_PANEL_METRICS["diagnostics"])

    paths = get_project_paths()
    results_dir = paths["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results_df.to_csv(results_dir / "test_results.csv", index=False)

    global WORKFLOW_COMPLETION
    WORKFLOW_COMPLETION["test_model"] = True

    pause()


# ACTION 4: Evaluate Saved Models

def evaluate_saved_models():
    """Compare all saved models and show performance metrics."""
    print_section("ACTION 4: EVALUATE SAVED MODELS")

    global WORKFLOW_COMPLETION
    if not WORKFLOW_COMPLETION["test_model"]:
        print_error("You must TEST models in this session before evaluating!")
        print_info("Run [3] Test Model first")
        pause()
        return

    paths = get_project_paths()
    results_dir = paths["results"]
    benchmark_file = results_dir / "benchmark_results.csv"
    test_file = results_dir / "test_results.csv"

    if not benchmark_file.exists() and not test_file.exists():
        print_warning("No results found. Run [2] Train Model and [3] Test Model first")
        pause()
        return

    if benchmark_file.exists():
        try:
            df = pd.read_csv(benchmark_file)
            if "r2" in df.columns and "model" in df.columns:
                print(f"{Colors.BRIGHT_GREEN}Training Results:{Colors.RESET}")
                results_list = df.to_dict('records')
                display_professional_results_table(results_list)
        except:
            pass

    if test_file.exists():
        try:
            df = pd.read_csv(test_file)
            if "r2" in df.columns and "model" in df.columns:
                print(f"{Colors.BRIGHT_GREEN}Test Results:{Colors.RESET}")
                results_list = df.to_dict('records')
                display_professional_results_table(results_list)
        except:
            pass

    all_results = []
    if benchmark_file.exists():
        df = pd.read_csv(benchmark_file)
        df["source"] = "training"
        all_results.append(df)
    if test_file.exists():
        df = pd.read_csv(test_file)
        df["source"] = "test"
        all_results.append(df)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        eval_file = results_dir / "evaluation_summary.csv"
        combined_df.to_csv(eval_file, index=False)
        print_success(f"Saved: {eval_file.name}")

    WORKFLOW_COMPLETION["evaluate_models"] = True
    pause()


# ACTION 5: Run Visualization

def run_visualization():
    """Generate all visualizations and export options."""
    print_section("ACTION 5: RUN VISUALIZATION")

    global SESSION_MODELS_DATA, SESSION_TRAIN_DATA, SESSION_TEST_DATA
    if SESSION_MODELS_DATA is None:
        print_error("You must TRAIN models in THIS SESSION before generating visualizations!")
        print_info("Run [2] Train Model first")
        pause()
        return

    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}Visualization Options:{Colors.RESET}\n")
    print(f"  {Colors.BOLD}{Colors.BRIGHT_GREEN}[1]{Colors.RESET}  Generate All Visualizations")
    print(f"  {Colors.BOLD}{Colors.BRIGHT_MAGENTA}[2]{Colors.RESET}  Copy Figures to Desktop")
    print(f"  {Colors.BOLD}{Colors.BRIGHT_MAGENTA}[3]{Colors.RESET}  Show Hyperparameters")
    print(f"  {Colors.BOLD}{Colors.BRIGHT_RED}[0]{Colors.RESET}  Back to menu\n")

    choice = input(f"{Colors.BOLD}Enter choice [0-3]: {Colors.RESET}").strip()

    if choice == "0":
        return
    elif choice == "1":
        from src.evaluation import (
            generate_correlation_heatmap,
            generate_distribution_plot,
            generate_error_distribution,
            generate_model_comparison_chart,
            generate_political_stability_evolution,
            generate_predictions_plot,
            generate_regional_analysis,
            generate_regional_stability_evolution,
            generate_residual_plots,
            generate_time_series_predictions,
        )

        print_info("Generating visualizations...\n")

        call_viz_function(generate_model_comparison_chart)
        display_feature_importance_table()
        call_viz_function(generate_predictions_plot)
        call_viz_function(generate_distribution_plot)
        call_viz_function(generate_correlation_heatmap)
        call_viz_function(generate_residual_plots)
        call_viz_function(generate_time_series_predictions)
        call_viz_function(generate_error_distribution)
        call_viz_function(generate_regional_analysis)
        call_viz_function(generate_political_stability_evolution)
        call_viz_function(generate_regional_stability_evolution)

        print_success("All visualizations generated!")
        global WORKFLOW_COMPLETION
        WORKFLOW_COMPLETION["visualization"] = True
        pause()
    elif choice == "2":
        from src.evaluation import copy_figures_to_desktop
        copy_figures_to_desktop()
    elif choice == "3":
        from src.evaluation import display_hyperparameters_table
        display_hyperparameters_table()
        WORKFLOW_COMPLETION["visualization"] = True
    else:
        print_error("Invalid choice! Please choose 0-3")
        pause()


def call_viz_function(func: Any) -> None:
    """Call visualization function with session data."""
    global SESSION_MODELS_DATA, SESSION_TRAIN_DATA, SESSION_TEST_DATA, SESSION_FULL_DATA, SESSION_PANEL_METRICS

    paths = get_project_paths()
    figures_dir = paths["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    if SESSION_FULL_DATA is None:
        processed_dir = paths["data_processed"]
        full_data_file = processed_dir / "full_data.csv"
        if full_data_file.exists():
            SESSION_FULL_DATA = pd.read_csv(full_data_file)
        elif SESSION_TRAIN_DATA is not None and SESSION_TEST_DATA is not None:
            SESSION_FULL_DATA = pd.concat([SESSION_TRAIN_DATA, SESSION_TEST_DATA])

    try:
        print(f"{Colors.DIM}  Generating {func.__name__.replace('generate_', '').replace('_', ' ')}...{Colors.RESET}", end=" ")
        if func.__name__ in ["generate_distribution_plot", "generate_correlation_heatmap"]:
            func(SESSION_TEST_DATA, figures_dir)
        elif func.__name__ in ["generate_political_stability_evolution", "generate_regional_stability_evolution"]:
            if SESSION_FULL_DATA is None:
                print(f"{Colors.YELLOW}[SKIP - No full data]{Colors.RESET}")
                return
            func(SESSION_FULL_DATA, figures_dir)
        elif func.__name__ == "generate_learning_curves":
            func(SESSION_MODELS_DATA, SESSION_TRAIN_DATA, figures_dir)
        elif func.__name__ == "generate_regional_analysis":
            func(SESSION_MODELS_DATA, SESSION_TEST_DATA, figures_dir)
        elif func.__name__ == "generate_time_series_predictions":
            func(SESSION_MODELS_DATA, SESSION_TEST_DATA, figures_dir, panel_metrics=SESSION_PANEL_METRICS)
        elif func.__name__ == "generate_model_comparison_chart":
            func(SESSION_MODELS_DATA, figures_dir, panel_metrics=SESSION_PANEL_METRICS)
        else:
            func(SESSION_MODELS_DATA, figures_dir)
        print(f"{Colors.BRIGHT_GREEN}[OK]{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}[ERROR: {str(e)}]{Colors.RESET}")


def display_feature_importance_table() -> None:
    """Display feature importance table using session model data."""
    global SESSION_MODELS_DATA
    from src.evaluation import display_feature_importance_table as display_impl
    display_impl(SESSION_MODELS_DATA)


# ACTION 6: Show Dashboard Link

def show_dashboard_link() -> None:
    """Launch Streamlit dashboard and open in browser."""
    print_section("ACTION 6: SHOW DASHBOARD LINK")

    dashboard_url = "http://localhost:8501"
    paths = get_project_paths()
    dashboard_path = paths["root"] / "main" / "dashboard.py"

    is_running = False
    try:
        response = requests.get(dashboard_url, timeout=1) if requests else None
        is_running = response and response.status_code == 200
    except:
        pass

    if is_running:
        print_success("Dashboard already running!")
        print(f"  {Colors.BRIGHT_CYAN}{dashboard_url}{Colors.RESET}\n")
        print_info("Opening in browser...")
        webbrowser.open(dashboard_url)
    else:
        print_info("Launching dashboard...\n")
        try:
            subprocess.run(["pkill", "-f", "streamlit"], capture_output=True, timeout=5)
            time.sleep(1)
        except:
            pass

        subprocess.Popen(
            ["streamlit", "run", str(dashboard_path)],
            cwd=str(paths["root"]),
        )
        print_success("Dashboard launched!")
        print(f"  {Colors.BRIGHT_CYAN}{dashboard_url}{Colors.RESET}\n")

        print_info("Waiting for dashboard to start...")
        time.sleep(3)
        webbrowser.open(dashboard_url)

    global WORKFLOW_COMPLETION
    WORKFLOW_COMPLETION["dashboard"] = True

    pause()


# ACTION 7: Test Coverage

def test_coverage() -> None:
    """Run pytest with coverage and display results."""
    print_section("ACTION 7: TEST COVERAGE")

    paths = get_project_paths()
    project_root = paths["root"]
    tests_dir = project_root / "tests"

    if not tests_dir.exists():
        print_error("Tests directory not found!")
        print_info(f"Expected location: {tests_dir}")
        pause()
        return

    print_info("Running tests with coverage...\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "--cov=src", "--cov-report=term"],
            cwd=project_root,
            timeout=180,
        )

        global WORKFLOW_COMPLETION
        WORKFLOW_COMPLETION["test_coverage"] = True

    except subprocess.TimeoutExpired:
        print_error("Test execution timed out (>3 minutes)")
    except FileNotFoundError:
        print_error("pytest or coverage not found! Install: pip install pytest pytest-cov")
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")

    pause()


# Main Menu

def show_main_menu() -> None:
    """Display main menu with workflow status."""
    global WORKFLOW_COMPLETION

    clear_screen()
    print_header("POLITICAL STABILITY PREDICTION", "ML & Econometric Analysis ")

    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}PROJECT WORKFLOW{Colors.RESET}")
    print(f"{Colors.DIM}{'' * 80}{Colors.RESET}\n")

    def menu_item(number, key, title, description):
        """Format menu item with completion status (green if completed, cyan if pending)."""
        completed = WORKFLOW_COMPLETION.get(key, False)
        color = Colors.BRIGHT_GREEN if completed else Colors.BRIGHT_CYAN
        return f"  {Colors.BOLD}{color}[{number}]{Colors.RESET}  {title:24s}{Colors.DIM}->{Colors.RESET}  {description}"

    print(menu_item("0", "check_environment", "Check Environment", "Verify dependencies and data"))
    print(menu_item("1", "data_preparation", "Run Data Preparation", "Load and prepare datasets"))
    print(menu_item("2", "train_model", "Train Model", "Train ML/econometric models"))
    print(menu_item("3", "test_model", "Test Model", "Test trained models"))
    print(menu_item("4", "evaluate_models", "Evaluate Saved Models", "Compare all models"))
    print(menu_item("5", "visualization", "Run Visualization", "Generate plots and charts"))
    print(menu_item("6", "dashboard", "Show Dashboard Link", "Display Streamlit dashboard"))
    print(menu_item("7", "test_coverage", "Test Coverage", "Run coverage analysis"))

    print(f"\n{Colors.DIM}{'' * 80}{Colors.RESET}")
    print(f"  {Colors.BOLD}{Colors.BRIGHT_RED}[Q]{Colors.RESET}  Quit")
    print()


def main() -> None:
    """Main CLI loop."""
    while True:
        show_main_menu()

        choice = input(f"{Colors.BOLD}Enter choice [0-7, Q]: {Colors.RESET}").strip().upper()

        if choice == "Q":
            print()
            print_info("Exiting program...")
            break

        elif choice == "0":
            check_environment()

        elif choice == "1":
            run_data_preparation()

        elif choice == "2":
            train_model()

        elif choice == "3":
            test_model()

        elif choice == "4":
            evaluate_saved_models()

        elif choice == "5":
            run_visualization()

        elif choice == "6":
            show_dashboard_link()

        elif choice == "7":
            test_coverage()

        else:
            print_error("Invalid choice! Please choose numbers 0-7 or Q to quit")
            pause()

    print()
    print(f"{Colors.BRIGHT_CYAN}Thank you for using Political Stability Prediction!{Colors.RESET}")
    print()


if __name__ == "__main__":
    main()
