"""
All model tests: ML models and Panel analyzer.
Tests for RF, XGBoost, GradientBoosting, SVR, KNN, MLP, ElasticNet, and PanelAnalyzer.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import models after path setup
from src.models import (  # noqa: E402
    ElasticNetPredictor,
    GradientBoostingPredictor,
    KNNPredictor,
    MLPPredictor,
    RandomForestPredictor,
    SVRPredictor,
    XGBoostPredictor,
)

# ============================================================================
# NON-PIPELINE ML MODELS (RF, XGBoost, GradientBoosting)
# ============================================================================


@pytest.mark.parametrize(
    "ModelClass,param_grid",
    [
        (RandomForestPredictor, {"n_estimators": [10, 20], "max_depth": [3, 5]}),
        (
            XGBoostPredictor,
            {"n_estimators": [10], "max_depth": [3], "learning_rate": [0.1]},
        ),
        (
            GradientBoostingPredictor,
            {"n_estimators": [20], "max_depth": [3], "learning_rate": [0.1]},
        ),
    ],
)
class TestNonPipelineModels:
    """Test suite for models without Pipeline (RF, XGBoost, GradientBoosting)."""

    def test_initialization(self, ModelClass, param_grid):
        """Verify model can be initialized without errors."""
        model = ModelClass()
        assert model is not None
        assert model.model is None
        assert model.grid_search is None  # Legacy attribute for backward compatibility

    def test_fit(self, ModelClass, param_grid, sample_data, logger):
        """Verify model trains with Optuna hyperparameter optimization."""
        from tests.conftest import fit_small_model

        X_train, y_train, _, _ = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)

        assert model.model is not None
        assert model.best_params_ is not None

    def test_predict(self, ModelClass, param_grid, sample_data, logger):
        """Verify model generates valid predictions after training."""
        from tests.conftest import fit_small_model

        X_train, y_train, X_test, _ = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)

    def test_evaluate(self, ModelClass, param_grid, sample_data, logger):
        """Verify model evaluation returns valid metrics."""
        from tests.conftest import assert_basic_metrics, fit_small_model

        X_train, y_train, X_test, y_test = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        assert_basic_metrics(metrics)
        assert metrics["n_samples"] == len(X_test)

    def test_feature_importance(self, ModelClass, param_grid, sample_data, logger):
        """Verify feature importance extraction for tree-based models."""
        from tests.conftest import fit_small_model

        X_train, y_train, _, _ = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)
        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == X_train.shape[1]
        assert importance.sum() > 0

    def test_save_load(self, ModelClass, param_grid, sample_data, logger, temp_dir):
        """Verify model persistence (save/load produces identical predictions)."""
        from tests.conftest import fit_small_model

        X_train, y_train, X_test, _ = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)
        pred_before = model.predict(X_test)

        model_path = temp_dir / f"{ModelClass.__name__}.pkl"
        model.save_model(model_path)
        assert model_path.exists()

        model_loaded = ModelClass(logger=logger)
        model_loaded.load_model(model_path)
        pred_after = model_loaded.predict(X_test)

        np.testing.assert_array_almost_equal(pred_before, pred_after)

    def test_cv_results(self, ModelClass, param_grid, sample_data, logger):
        """Verify access to Optuna optimization results."""
        from tests.conftest import fit_small_model

        X_train, y_train, _, _ = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)
        cv_results = model.get_cv_results()

        assert isinstance(cv_results, pd.DataFrame)
        assert len(cv_results) > 0

    def test_overfitting_detection(self, ModelClass, param_grid, sample_data, logger):
        """Verify calculation of train-CV gap for overfitting detection."""
        from tests.conftest import fit_small_model

        X_train, y_train, _, _ = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)

        assert model.train_score_ is not None
        assert model.cv_score_ is not None
        assert model.overfitting_gap_ is not None
        assert model.overfitting_gap_ >= 0

    def test_prediction_without_fit_raises_error(
        self, ModelClass, param_grid, sample_data, logger
    ):
        """Verify prediction without fitting raises ValueError."""
        _, _, X_test, _ = sample_data
        model = ModelClass(logger=logger)

        with pytest.raises(ValueError, match="not fitted|model"):
            model.predict(X_test)


# ============================================================================
# PIPELINE ML MODELS (SVR, KNN, MLP, ElasticNet)
# ============================================================================


@pytest.mark.parametrize(
    "ModelClass,param_grid",
    [
        (SVRPredictor, {"model__C": [1, 10], "model__kernel": ["rbf"]}),
        (KNNPredictor, {"model__n_neighbors": [3, 5]}),
        (
            MLPPredictor,
            {"model__hidden_layer_sizes": [(20,)], "model__max_iter": [100]},
        ),
        (ElasticNetPredictor, {"model__alpha": [0.1, 1.0], "model__l1_ratio": [0.5]}),
    ],
)
class TestPipelineModels:
    """Test suite for models using Pipeline (SVR, KNN, MLP, ElasticNet)."""

    def test_initialization(self, ModelClass, param_grid):
        """Verify Pipeline model can be initialized (scaler + estimator)."""
        model = ModelClass()
        assert model is not None
        assert model.model is None
        assert model.grid_search is None  # Legacy attribute for backward compatibility

    def test_fit(self, ModelClass, param_grid, sample_data, logger):
        """Verify Pipeline trains with Optuna optimization and StandardScaler."""
        from tests.conftest import fit_small_model

        X_train, y_train, _, _ = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)

        assert model.model is not None
        assert model.best_params_ is not None

    def test_pipeline_structure(self, ModelClass, param_grid, sample_data, logger):
        """Verify Pipeline contains both scaler and model steps."""
        from tests.conftest import fit_small_model

        X_train, y_train, _, _ = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)

        assert hasattr(model.model, "named_steps")
        assert "scaler" in model.model.named_steps
        assert "model" in model.model.named_steps

    def test_predict(self, ModelClass, param_grid, sample_data, logger):
        """Verify Pipeline generates valid predictions (preprocessing + inference)."""
        from tests.conftest import fit_small_model

        X_train, y_train, X_test, _ = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)

    def test_evaluate(self, ModelClass, param_grid, sample_data, logger):
        """Verify Pipeline evaluation returns valid metrics."""
        from tests.conftest import assert_basic_metrics, fit_small_model

        X_train, y_train, X_test, y_test = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        assert_basic_metrics(metrics)
        assert metrics["n_samples"] == len(X_test)

    def test_save_load(self, ModelClass, param_grid, sample_data, logger, temp_dir):
        """Verify Pipeline persistence (scaler + model saved together)."""
        from tests.conftest import fit_small_model

        X_train, y_train, X_test, _ = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)
        pred_before = model.predict(X_test)

        model_path = temp_dir / f"{ModelClass.__name__}.pkl"
        model.save_model(model_path)
        assert model_path.exists()

        model_loaded = ModelClass(logger=logger)
        model_loaded.load_model(model_path)
        pred_after = model_loaded.predict(X_test)

        np.testing.assert_array_almost_equal(pred_before, pred_after)

    def test_cv_results(self, ModelClass, param_grid, sample_data, logger):
        """Verify access to Optuna optimization results."""
        from tests.conftest import fit_small_model

        X_train, y_train, _, _ = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)
        cv_results = model.get_cv_results()

        assert isinstance(cv_results, pd.DataFrame)
        assert len(cv_results) > 0

    def test_overfitting_detection(self, ModelClass, param_grid, sample_data, logger):
        """Verify calculation of train-CV gap for overfitting detection."""
        from tests.conftest import fit_small_model

        X_train, y_train, _, _ = sample_data
        model = fit_small_model(ModelClass(logger=logger), X_train, y_train)

        assert model.train_score_ is not None
        assert model.cv_score_ is not None
        assert model.overfitting_gap_ is not None
        assert model.overfitting_gap_ >= 0

    def test_prediction_without_fit_raises_error(
        self, ModelClass, param_grid, sample_data, logger
    ):
        """Verify prediction without fitting raises ValueError."""
        _, _, X_test, _ = sample_data
        model = ModelClass(logger=logger)

        with pytest.raises(ValueError, match="not fitted|model"):
            model.predict(X_test)


# ============================================================================
# ELASTIC NET SPECIFIC TESTS
# ============================================================================


def test_elastic_net_coefficients(sample_data, logger):
    """Verify ElasticNet coefficient extraction with L1/L2 regularization."""
    from tests.conftest import fit_small_model

    X_train, y_train, _, _ = sample_data
    model = ElasticNetPredictor(logger=logger)
    fit_small_model(model, X_train, y_train, n_trials=2, cv=2)

    coefficients = model.get_coefficients()

    assert coefficients is not None
    assert len(coefficients) == X_train.shape[1]
    assert all(fname in coefficients.index for fname in X_train.columns)


# ============================================================================
# PANEL ANALYZER TESTS
# ============================================================================


class TestPanelAnalyzer:
    """Test suite for PanelAnalyzer (Fixed Effects panel model)."""

    def test_initialization(self, panel_data, temp_dir):
        """Verify PanelAnalyzer initializes with longitudinal data."""
        from src.models import PanelAnalyzer

        csv_path = temp_dir / "panel_data.csv"
        panel_data.to_csv(csv_path, index=False)

        analyzer = PanelAnalyzer(
            data_path=csv_path,
            target="political_stability",
            predictors=[
                "gdp_per_capita",
                "unemployment",
                "inflation",
                "gdp_growth",
                "effectiveness",
                "rule_of_law",
                "trade",
                "hdi",
            ],
        )

        assert analyzer is not None
        assert analyzer.target == "political_stability"

    def test_fit_dynamic_panel(self, panel_data, temp_dir):
        """Verify dynamic panel model with Fixed Effects and lagged variables."""
        from src.models import PanelAnalyzer

        csv_path = temp_dir / "panel_data.csv"
        panel_data.to_csv(csv_path, index=False)

        analyzer = PanelAnalyzer(
            data_path=csv_path,
            target="political_stability",
            predictors=[
                "gdp_per_capita",
                "unemployment",
                "inflation",
                "gdp_growth",
                "effectiveness",
                "rule_of_law",
                "trade",
                "hdi",
            ],
        )

        # Load data first
        analyzer.load_data()

        # Prepare panel structure
        analyzer.prepare_panel_structure()

        # Fit dynamic panel model
        analyzer.fit_dynamic_panel(lags=1)

        assert analyzer.dynamic_results is not None
        assert hasattr(analyzer, "dynamic_lags_")
        assert hasattr(analyzer, "dynamic_lag_name_")
        assert analyzer.dynamic_lags_ == 1

    def test_evaluate_on_test(self, panel_data, temp_dir):
        """Verify panel model evaluation with out-of-sample predictions."""
        from src.models import PanelAnalyzer

        csv_path = temp_dir / "panel_data.csv"
        panel_data.to_csv(csv_path, index=False)

        analyzer = PanelAnalyzer(
            data_path=csv_path,
            target="political_stability",
            predictors=[
                "gdp_per_capita",
                "unemployment",
                "inflation",
                "gdp_growth",
                "effectiveness",
                "rule_of_law",
                "trade",
                "hdi",
            ],
        )

        analyzer.load_data()
        analyzer.prepare_panel_structure()

        # Perform train/test split
        analyzer.train_test_split(train_end_year=2015)

        # Fit on training data
        analyzer.fit_dynamic_panel(lags=1)

        # Evaluate on test set
        metrics = analyzer.evaluate_on_test()

        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mse" in metrics
        assert "n_test" in metrics

    def test_persistence_analysis(self, panel_data, temp_dir):
        """Verify shock persistence calculation via AR coefficient and half-life."""
        from src.models import PanelAnalyzer

        csv_path = temp_dir / "panel_data.csv"
        panel_data.to_csv(csv_path, index=False)

        analyzer = PanelAnalyzer(
            data_path=csv_path,
            target="political_stability",
            predictors=[
                "gdp_per_capita",
                "unemployment",
                "inflation",
                "gdp_growth",
                "effectiveness",
                "rule_of_law",
                "trade",
                "hdi",
            ],
        )

        analyzer.load_data()
        analyzer.prepare_panel_structure()
        analyzer.fit_dynamic_panel(lags=1)

        persistence = analyzer.analyze_persistence()

        # Always present
        assert isinstance(persistence, dict)
        assert "coefficient" in persistence
        assert "p_value" in persistence
        assert "interpretation" in persistence

        # half_life only exists when 0 < coefficient < 1
        coef = persistence["coefficient"]
        if 0 < coef < 1:
            assert "half_life" in persistence
            assert persistence["half_life"] > 0
        else:
            # No half_life for non-stationary or negative coefficients
            assert "half_life" not in persistence or persistence["half_life"] is None

    def test_diagnostic_tests(self, panel_data, temp_dir):
        """Verify econometric diagnostics: Hausman, autocorrelation, heteroscedasticity."""
        from src.models import PanelAnalyzer

        csv_path = temp_dir / "panel_data.csv"
        panel_data.to_csv(csv_path, index=False)

        analyzer = PanelAnalyzer(
            data_path=csv_path,
            target="political_stability",
            predictors=[
                "gdp_per_capita",
                "unemployment",
                "inflation",
                "gdp_growth",
                "effectiveness",
                "rule_of_law",
                "trade",
                "hdi",
            ],
        )

        analyzer.load_data()
        analyzer.prepare_panel_structure()
        analyzer.fit_dynamic_panel(lags=1)

        diagnostics = analyzer.run_diagnostic_tests()

        assert "hausman" in diagnostics
        assert "autocorrelation" in diagnostics
        assert "heteroscedasticity" in diagnostics

        # Hausman test validation
        hausman = diagnostics["hausman"]
        assert "statistic" in hausman
        assert "p_value" in hausman
        assert "conclusion" in hausman
        # Statistic should be non-negative chi-squared value
        assert hausman["statistic"] >= 0
        # P-value should be between 0 and 1
        assert 0 <= hausman["p_value"] <= 1

        # Autocorrelation test validation
        autocorr = diagnostics["autocorrelation"]
        assert "rho" in autocorr
        assert "p_value" in autocorr
        assert "conclusion" in autocorr
        # Rho (AR1 coefficient) should be between -1 and 1
        assert -1 <= autocorr["rho"] <= 1
        # P-value should be between 0 and 1
        assert 0 <= autocorr["p_value"] <= 1

        # Heteroscedasticity test validation
        hetero = diagnostics["heteroscedasticity"]
        assert "lm_stat" in hetero
        assert "lm_pvalue" in hetero
        assert "f_stat" in hetero
        assert "f_pvalue" in hetero
        assert "conclusion" in hetero
        # Statistics should be non-negative
        assert hetero["lm_stat"] >= 0
        assert hetero["f_stat"] >= 0
        # P-values should be between 0 and 1
        assert 0 <= hetero["lm_pvalue"] <= 1
        assert 0 <= hetero["f_pvalue"] <= 1


def test_regression_metrics():
    """
    Test regression_metrics utility function.

    Verifies that the regression metrics function correctly computes
    R-squared, adjusted R-squared, F-statistic, RMSE, and MAE for
    a given set of predictions and ground truth values.
    """
    from src.models import regression_metrics

    # Create synthetic predictions with good fit
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y_pred = np.array([1.1, 2.0, 2.9, 4.2, 4.8, 6.1, 7.0, 7.9])

    metrics = regression_metrics(y_true, y_pred, n_features=2)

    # Verify all expected metrics are present
    assert "r2" in metrics
    assert "adj_r2" in metrics
    assert "f_stat" in metrics
    assert "f_pvalue" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "n_samples" in metrics
    assert "n_features" in metrics

    # Verify metric values are reasonable
    assert 0 <= metrics["r2"] <= 1
    assert metrics["adj_r2"] <= metrics["r2"]
    assert metrics["f_stat"] > 0
    assert 0 <= metrics["f_pvalue"] <= 1
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert metrics["n_samples"] == 8
    assert metrics["n_features"] == 2


def test_panel_analyzer_with_invalid_index(panel_data):
    """
    Test PanelAnalyzer with non-MultiIndex data.

    Verifies that PanelAnalyzer properly handles data without a MultiIndex
    by raising an appropriate error rather than producing incorrect results.
    """
    from src.models import PanelAnalyzer

    # Create data without MultiIndex (should fail)
    X = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [2, 3, 4, 5, 6]})
    y = pd.Series([1, 2, 3, 4, 5])

    analyzer = PanelAnalyzer()

    # Should raise an exception because data lacks proper panel structure
    with pytest.raises(Exception):
        analyzer.fit(X, y)


def test_panel_analyzer_save_load(panel_data, temp_dir):
    """
    Test PanelAnalyzer model saving and loading.

    Verifies that a trained PanelAnalyzer can be serialized to disk
    and loaded back with preserved parameters and predictions.
    """
    from src.models import PanelAnalyzer

    # Prepare panel data
    panel_data = panel_data.set_index(["Country Name", "Year"])
    X = panel_data[["gdp_per_capita", "unemployment"]]
    y = panel_data["political_stability"]

    # Train and save
    analyzer = PanelAnalyzer()
    analyzer.fit(X, y)

    save_path = temp_dir / "panel_model.pkl"
    analyzer.save(str(save_path))

    # Load and verify
    loaded_analyzer = PanelAnalyzer.load(str(save_path))
    assert loaded_analyzer is not None

    # Verify predictions match
    original_pred = analyzer.predict(X)
    loaded_pred = loaded_analyzer.predict(X)

    np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)


def test_base_predictor_evaluate_without_fit():
    """
    Test BasePredictor evaluate method before fitting.

    Verifies that calling evaluate on an unfitted model raises
    an appropriate exception.
    """
    from src.models import RandomForestPredictor

    predictor = RandomForestPredictor()
    X = np.random.rand(50, 5)
    y = np.random.rand(50)

    # Should raise an exception because model is not fitted
    with pytest.raises(Exception):
        predictor.evaluate(X, y)


def test_xgboost_predictor_without_xgboost():
    """
    Test XGBoostPredictor when XGBoost is not available.

    Verifies that the predictor gracefully handles the absence of
    the XGBoost library rather than crashing during import.
    """
    from src.models import XGBOOST_AVAILABLE

    # If XGBoost is not available, this test verifies the flag works correctly
    if not XGBOOST_AVAILABLE:
        from src.models import XGBoostPredictor

        predictor = XGBoostPredictor()
        assert predictor is not None


def test_model_cv_results_structure(small_panel_data):
    """
    Test cross-validation results structure.

    Verifies that the cv_results attribute contains expected keys
    and has proper structure after model fitting with cross-validation.
    """
    from src.models import RandomForestPredictor

    X = small_panel_data[["gdp_per_capita", "unemployment"]]
    y = small_panel_data["political_stability"]

    predictor = RandomForestPredictor()

    # Fit with minimal hyperparameter grid
    param_grid = {"n_estimators": [10, 20], "max_depth": [3, 5]}
    predictor.fit(X, y, param_grid=param_grid, n_trials=2, cv=2)

    # Verify cv_results structure
    assert predictor.cv_results is not None
    assert "mean_test_score" in predictor.cv_results
    assert "std_test_score" in predictor.cv_results
    assert "params" in predictor.cv_results

    # Verify scores are valid
    assert all(
        isinstance(score, (int, float))
        for score in predictor.cv_results["mean_test_score"]
    )
    assert all(score <= 1.0 for score in predictor.cv_results["mean_test_score"])
