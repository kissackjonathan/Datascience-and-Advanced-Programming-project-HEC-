"""
Shared pytest fixtures for all test modules.

Resources used for creating tests:
- YouTube Tutorial: https://www.youtube.com/watch?v=-PDCeME4MtM
- Python unittest documentation: https://docs.python.org/fr/3.9/library/unittest.html
- Pytest getting started guide: https://docs.pytest.org/en/latest/getting-started.html
"""
# Set matplotlib backend before any imports
import matplotlib

matplotlib.use("Agg")

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def sample_data():
    """Generate synthetic regression data for testing."""
    np.random.seed(42)
    n_train, n_test, n_features = 100, 30, 5

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y_train = pd.Series(
        X_train["f0"] * 2 + X_train["f1"] * 1.5 + np.random.randn(n_train) * 0.2
    )

    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y_test = pd.Series(
        X_test["f0"] * 2 + X_test["f1"] * 1.5 + np.random.randn(n_test) * 0.2
    )

    return X_train, y_train, X_test, y_test


@pytest.fixture
def temp_dir():
    """Create a temporary directory for saving models/files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="module")
def logger():
    """Create a test logger."""
    return logging.getLogger("test")


@pytest.fixture
def panel_data():
    """Generate synthetic panel data for PanelAnalyzer tests."""
    np.random.seed(42)
    countries = [f"Country_{i}" for i in range(20)]
    years = list(range(2000, 2020))

    data = []
    for country in countries:
        country_effect = np.random.randn()
        for year in years:
            data.append(
                {
                    "Country Name": country,
                    "Year": year,
                    "political_stability": 0.5
                    + country_effect * 0.3
                    + np.random.randn() * 0.2,
                    "gdp_per_capita": 40000
                    + country_effect * 5000
                    + np.random.randn() * 2000,
                    "unemployment": 5 + np.random.randn() * 1.5,
                    "inflation": 2 + np.random.randn() * 0.8,
                    "gdp_growth": 2.5 + np.random.randn() * 1.2,
                    "effectiveness": 0.5
                    + country_effect * 0.2
                    + np.random.randn() * 0.15,
                    "rule_of_law": 0.5
                    + country_effect * 0.2
                    + np.random.randn() * 0.15,
                    "trade": 50 + np.random.randn() * 10,
                    "hdi": 0.7 + country_effect * 0.1 + np.random.randn() * 0.05,
                }
            )

    return pd.DataFrame(data)


# ============================================================================
# TEST HELPERS - Reduce duplication and improve test clarity
# ============================================================================


def fit_small_model(model, X_train, y_train, n_trials=10, cv=2):
    """
    Fit model with minimal trials for fast testing.

    Args:
        model: Model instance (RandomForestPredictor, XGBoostPredictor, etc.)
        X_train: Training features
        y_train: Training target
        n_trials: Number of Optuna trials (default: 10 for speed)
        cv: Cross-validation folds (default: 2 for speed)

    Returns:
        Fitted model instance
    """
    model.fit(X_train, y_train, n_trials=n_trials, cv=cv)
    return model


def assert_basic_metrics(metrics):
    """
    Assert standard regression metrics are valid.

    Args:
        metrics: Dict with r2, rmse, mae, n_samples, etc.
    """
    assert "r2" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "n_samples" in metrics
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


def models_data_from_trained(trained_models, include=("RF", "XGB")):
    """
    Generate models_data list from trained_models fixture.

    Automatically constructs the models_data format expected by evaluation functions,
    eliminating repetitive dict construction in tests.

    Args:
        trained_models: Dict from trained_models fixture
        include: Tuple of model names to include ("RF", "XGB")

    Returns:
        List[Dict]: Models data in format expected by evaluation functions
    """
    models_data = []

    for model_key in include:
        model_lower = model_key.lower()
        model_dict = {
            "model": model_key,
            "y_test": trained_models["y_test"],
            "y_pred": trained_models[f"{model_lower}_pred"],
            **trained_models[f"{model_lower}_metrics"],
        }

        # Add model object if needed (for feature importance, learning curves)
        if f"{model_lower}" in trained_models:
            model_dict["model_obj"] = trained_models[model_lower]

        # Add training data if needed (for learning curves)
        if "X_train" in trained_models:
            model_dict["X_train"] = trained_models["X_train"]

        # Add test data if needed (for regional analysis)
        if "X_test" in trained_models:
            model_dict["X_test"] = trained_models["X_test"]

        models_data.append(model_dict)

    return models_data


@pytest.fixture(scope="module")
def trained_models(sample_data, logger):
    """
    Pre-train RF and XGBoost for visualization tests (avoid repeated training).

    This fixture is expensive but runs only once per test module, significantly
    speeding up visualization tests that need trained models.

    Returns:
        Dict with trained models, metrics, and predictions:
            - rf: Trained RandomForestPredictor
            - xgb: Trained XGBoostPredictor
            - rf_metrics: Evaluation metrics for RF
            - xgb_metrics: Evaluation metrics for XGBoost
            - rf_pred: RF predictions on test set
            - xgb_pred: XGBoost predictions on test set
            - X_train, y_train, X_test, y_test: Data splits
    """
    from src.models import RandomForestPredictor, XGBoostPredictor

    X_train, y_train, X_test, y_test = sample_data

    # Train Random Forest with minimal trials
    rf = RandomForestPredictor(logger=logger)
    rf.fit(X_train, y_train, n_trials=10, cv=2)

    # Train XGBoost with minimal trials
    xgb = XGBoostPredictor(logger=logger)
    xgb.fit(X_train, y_train, n_trials=10, cv=2)

    return {
        "rf": rf,
        "xgb": xgb,
        "rf_metrics": rf.evaluate(X_test, y_test),
        "xgb_metrics": xgb.evaluate(X_test, y_test),
        "rf_pred": rf.predict(X_test),
        "xgb_pred": xgb.predict(X_test),
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
