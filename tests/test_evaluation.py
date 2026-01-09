"""
Evaluation module tests.
Tests for visualization and evaluation functions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# BASIC VISUALIZATION TESTS (Parameterized)
# ============================================================================


@pytest.mark.parametrize(
    "plot_func,filename,needs_importance",
    [
        ("plot_predictions", "predictions.png", False),
        ("plot_residuals", "residuals.png", False),
        ("plot_feature_importance", "importance.png", True),
    ],
)
def test_basic_plots_create_file(
    sample_data, temp_dir, plot_func, filename, needs_importance
):
    """Verify basic plot functions create output files."""
    from src import evaluation

    func = getattr(evaluation, plot_func)

    _, _, _, y_test = sample_data
    output_path = temp_dir / filename

    if needs_importance:
        # Feature importance plot needs Series input
        importance = pd.Series({"f0": 0.4, "f1": 0.3, "f2": 0.2, "f3": 0.1})
        func(importance, model_name="Test", output_path=output_path)
    else:
        # Prediction/residual plots need y_test, y_pred
        y_pred = y_test + np.random.randn(len(y_test)) * 0.1
        func(y_test, y_pred, model_name="Test", output_path=output_path)

    assert output_path.exists()


def test_evaluation_display_results_table():
    """Verify formatted results table displays without error."""
    from src.evaluation import display_results_table

    results = pd.DataFrame(
        {
            "model": ["RF", "XGB"],
            "r2": [0.85, 0.87],
            "rmse": [0.5, 0.45],
            "mae": [0.4, 0.35],
        }
    )

    display_results_table(results, title="Test Results")


def test_evaluation_plot_model_comparison(temp_dir):
    """Verify multi-model comparison bar chart creation."""
    from src.evaluation import plot_model_comparison

    results = pd.DataFrame(
        {
            "model": ["RF", "XGB", "SVR"],
            "r2": [0.85, 0.87, 0.82],
            "rmse": [0.5, 0.45, 0.55],
            "mae": [0.4, 0.35, 0.45],
        }
    )

    output_path = temp_dir / "comparison.png"
    plot_model_comparison(results, output_path=output_path)

    assert output_path.exists()


def test_evaluation_create_correlation_heatmap(temp_dir):
    """Verify correlation heatmap creation for variable relationships."""
    from src.evaluation import create_correlation_heatmap

    data = pd.DataFrame(
        {
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
            "f3": np.random.randn(50),
            "target": np.random.randn(50),
        }
    )

    output_path = temp_dir / "heatmap.png"
    create_correlation_heatmap(data, output_path=output_path)

    assert output_path.exists()


def test_evaluation_save_results_summary(temp_dir):
    """Verify results saved to CSV for metric persistence."""
    from src.evaluation import save_results_summary

    results = pd.DataFrame(
        {"model": ["RF", "XGB"], "r2": [0.85, 0.87], "rmse": [0.5, 0.45]}
    )

    output_path = temp_dir / "results.csv"
    save_results_summary(results, output_path, description="Test Results")

    assert output_path.exists()


def test_evaluation_compare_models():
    """Verify overfitting detection via train/test comparison."""
    from src.evaluation import compare_models

    benchmark_df = pd.DataFrame(
        {"model": ["RF", "XGB"], "r2": [0.90, 0.92], "rmse": [0.4, 0.35]}
    )

    test_df = pd.DataFrame(
        {"model": ["RF", "XGB"], "r2": [0.85, 0.87], "rmse": [0.5, 0.45]}
    )

    compare_models(benchmark_df, test_df)


# ============================================================================
# COMPREHENSIVE VISUALIZATION TESTS
# ============================================================================


def test_evaluation_generate_model_comparison_chart(trained_models, temp_dir):
    """Verify multi-metric comparison chart generation (R², MAE, RMSE)."""
    from src.evaluation import generate_model_comparison_chart
    from tests.conftest import models_data_from_trained

    models_data = models_data_from_trained(trained_models)

    result = generate_model_comparison_chart(models_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_feature_importance_plot(trained_models, temp_dir):
    """Verify feature importance plot generation for tree models."""
    from src.evaluation import generate_feature_importance_plot
    from tests.conftest import models_data_from_trained

    models_data = models_data_from_trained(trained_models, include=("RF",))

    result = generate_feature_importance_plot(models_data, temp_dir)
    assert result is not None
    assert result.exists()


def test_evaluation_generate_predictions_plot(trained_models, temp_dir):
    """Verify predictions vs actual scatter plot generation."""
    from src.evaluation import generate_predictions_plot
    from tests.conftest import models_data_from_trained

    models_data = models_data_from_trained(trained_models)

    result = generate_predictions_plot(models_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_residual_plots(trained_models, temp_dir):
    """Verify residual plot generation for error pattern analysis."""
    from src.evaluation import generate_residual_plots
    from tests.conftest import models_data_from_trained

    models_data = models_data_from_trained(trained_models)

    result = generate_residual_plots(models_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_learning_curves(trained_models, temp_dir):
    """Verify learning curve generation for overfitting diagnosis."""
    from src.evaluation import generate_learning_curves
    from tests.conftest import models_data_from_trained

    train_data = trained_models["X_train"].copy()
    train_data["political_stability"] = trained_models["y_train"]

    models_data = models_data_from_trained(trained_models, include=("RF",))

    result = generate_learning_curves(models_data, train_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_time_series_predictions(trained_models, temp_dir):
    """Verify time series prediction plot generation."""
    from src.evaluation import generate_time_series_predictions
    from tests.conftest import models_data_from_trained

    test_data = trained_models["X_test"].copy()
    test_data["political_stability"] = trained_models["y_test"]
    test_data["Year"] = np.random.choice(range(2015, 2021), size=len(test_data))
    test_data = test_data.set_index("Year", append=True)

    models_data = models_data_from_trained(trained_models)

    result = generate_time_series_predictions(models_data, test_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_cv_scores_comparison(trained_models, temp_dir):
    """Verify CV scores comparison plot generation."""
    from src.evaluation import generate_cv_scores_comparison
    from tests.conftest import models_data_from_trained

    train_data = trained_models["X_train"].copy()
    train_data["political_stability"] = trained_models["y_train"]

    models_data = models_data_from_trained(trained_models)

    result = generate_cv_scores_comparison(models_data, train_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_error_distribution(trained_models, temp_dir):
    """Verify error distribution plot generation with KDE."""
    from src.evaluation import generate_error_distribution
    from tests.conftest import models_data_from_trained

    models_data = models_data_from_trained(trained_models)

    result = generate_error_distribution(models_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_regional_analysis(trained_models, temp_dir):
    """Verify regional performance analysis plot generation."""
    from src.evaluation import generate_regional_analysis
    from tests.conftest import models_data_from_trained

    test_data = trained_models["X_test"].copy()
    test_data["political_stability"] = trained_models["y_test"]
    countries = ["France", "Germany", "United States", "Japan", "Brazil"]
    test_data["Country Name"] = np.random.choice(countries, size=len(test_data))
    test_data = test_data.set_index("Country Name", append=True)

    models_data = models_data_from_trained(trained_models, include=("RF",))

    result = generate_regional_analysis(models_data, test_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_ml_vs_benchmark_comparison(trained_models, temp_dir):
    """Verify ML vs econometric benchmark comparison chart generation."""
    from src.evaluation import generate_ml_vs_benchmark_comparison
    from tests.conftest import models_data_from_trained

    ml_results = models_data_from_trained(trained_models, include=("RF",))
    panel_metrics = {"r2": 0.82, "mae": 0.45, "rmse": 0.55}

    result = generate_ml_vs_benchmark_comparison(ml_results, panel_metrics, temp_dir)
    assert result.exists()


def test_evaluation_generate_political_stability_evolution(sample_data, temp_dir):
    """Verify temporal trend visualization across countries."""
    from src.evaluation import generate_political_stability_evolution

    _, _, X_test, y_test = sample_data

    test_data = X_test.copy()
    test_data["political_stability"] = y_test
    test_data["Year"] = np.random.choice(range(2000, 2021), size=len(test_data))
    test_data["Country Name"] = np.random.choice(
        ["France", "Germany", "USA", "Japan"], size=len(test_data)
    )

    result = generate_political_stability_evolution(test_data, temp_dir)
    assert result.exists()
