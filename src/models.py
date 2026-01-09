"""
Model Definitions for Political Stability Prediction

Models:
- RandomForestPredictor          - XGBoostPredictor
- GradientBoostingPredictor      - ElasticNetPredictor
- SVRPredictor                   - KNNPredictor
- MLPPredictor                   - PanelAnalyzer (Dynamic Panel with FE)

Each model is a standalone class with fit(), predict(), and evaluate() methods.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd

# ============================================================================
# PANEL DATA & ECONOMETRIC LIBRARIES
# ============================================================================
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools import add_constant

# ============================================================================
# REPRODUCIBILITY: Fix all random seeds for consistent results
# ============================================================================
# This ensures that running the code multiple times produces identical results,
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def regression_metrics(
    y_true: pd.Series, y_pred: np.ndarray, n_features: int
) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.

    Factorizes metrics calculation to avoid repetition across predictors.

    Args:
        y_true: True target values
        y_pred: Predicted values
        n_features: Number of features (for adjusted R^2 and F-statistic)

    Returns:
        Dict with r2, adj_r2, f_stat, f_pvalue, rmse, mae, n_samples, n_features
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)

    # Adjusted R^2 penalizes model complexity
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    # F-statistic for overall model significance
    if r2 < 1.0 and n_features > 0:
        f_stat = (r2 / n_features) / ((1 - r2) / (n - n_features - 1))
        f_pvalue = float(1 - stats.f.cdf(f_stat, n_features, n - n_features - 1))
    else:
        f_stat = np.inf
        f_pvalue = 0.0

    return {
        "r2": r2,
        "adj_r2": adj_r2,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "n_samples": n,
        "n_features": n_features,
    }


# ============================================================================
# MACHINE LEARNING MODELS - BASE CLASS
# ============================================================================


class BasePredictor:
    """Base class for ML predictors with Optuna Bayesian optimization."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize base predictor attributes."""
        self.model = None
        self.grid_search = None  # Legacy attribute for backward compatibility
        self.best_params_ = None
        self.train_score_ = None
        self.cv_score_ = None
        self.overfitting_gap_ = None
        self.logger = logger or logging.getLogger(__name__)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model on test set."""
        y_pred = self.predict(X_test)
        metrics = regression_metrics(y_test, y_pred, X_test.shape[1])

        # Add overfitting diagnostics from CV
        metrics.update(
            {
                "train_score": self.train_score_,
                "cv_score": self.cv_score_,
                "overfitting_gap": self.overfitting_gap_,
                "has_overfitting": self.overfitting_gap_ > 0.1
                if self.overfitting_gap_ is not None
                else None,
            }
        )
        return metrics

    def save_model(self, filepath: Path) -> None:
        """Save trained model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path) -> "BasePredictor":
        """Load trained model from disk."""
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from: {filepath}")
        return self

    def get_cv_results(self) -> pd.DataFrame:
        """Get Optuna optimization results."""
        if self.best_params_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return pd.DataFrame(
            {
                "best_params": [self.best_params_],
                "cv_score": [self.cv_score_],
                "train_score": [self.train_score_],
                "overfitting_gap": [self.overfitting_gap_],
            }
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 100,
        cv: int = 5,
        n_jobs: int = -1,
        param_grid: Optional[Dict] = None,  # Deprecated
        verbose: int = 0,  # Deprecated
    ) -> "BasePredictor":
        """Template method: Optuna Bayesian optimization with TPE and MedianPruner."""
        self.logger.info(
            f"Starting Optuna Bayesian optimization with {n_trials} trials and {cv}-fold CV"
        )

        def objective(trial):
            params = self._suggest_params(trial)
            model = self._build_model(params, n_jobs)
            scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="r2", n_jobs=1
            )
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=RANDOM_SEED
            ),  # Fix seed for reproducibility
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

        self.best_params_ = study.best_params
        self.cv_score_ = study.best_value
        self._process_best_params()

        self.logger.info(f"Best CV R^2 score: {self.cv_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params_}")

        self.model = self._build_model(self.best_params_, n_jobs)
        self.model.fit(X_train, y_train)

        self.train_score_ = self.model.score(X_train, y_train)
        self.overfitting_gap_ = self.train_score_ - self.cv_score_

        self.logger.info(f"Train R^2: {self.train_score_:.4f}")
        self.logger.info(f"Train-CV gap: {self.overfitting_gap_:.4f}")

        if self.overfitting_gap_ > 0.1:
            self.logger.warning(f"[WARNING] Potential overfitting detected (gap > 0.1)")
        else:
            self.logger.info(f"[OK] No significant overfitting (gap < 0.1)")

        self._post_fit(X_train, y_train)
        return self

    def _suggest_params(self, trial) -> Dict:
        """Override in subclass to suggest hyperparameters."""
        raise NotImplementedError

    def _build_model(self, params: Dict, n_jobs: int):
        """Override in subclass to build model with params."""
        raise NotImplementedError

    def _process_best_params(self) -> None:
        """Hook to process best_params_ (e.g., format conversion)."""
        pass

    def _post_fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Hook for post-processing (e.g., feature importance)."""
        pass


class RandomForestPredictor(BasePredictor):
    """Random Forest with Optuna Bayesian optimization."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.feature_importance_ = None

    def _suggest_params(self, trial) -> Dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5, 0.7]
            ),
            "min_impurity_decrease": trial.suggest_float(
                "min_impurity_decrease", 0.0, 0.2
            ),
        }

    def _build_model(self, params: Dict, n_jobs: int):
        return RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=n_jobs)

    def _post_fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

    def get_feature_importance(self) -> pd.Series:
        """Get Random Forest feature importance rankings."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.feature_importance_


class XGBoostPredictor(BasePredictor):
    """XGBoost Regressor with Optuna optimization."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )
        super().__init__(logger)
        self.feature_importance_ = None

    def _suggest_params(self, trial) -> Dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def _build_model(self, params: Dict, n_jobs: int):
        return XGBRegressor(
            **params,
            random_state=RANDOM_SEED,
            objective="reg:squarederror",
            tree_method="auto",
            n_jobs=n_jobs,
        )

    def _post_fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance rankings."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.feature_importance_


class KNNPredictor(BasePredictor):
    """K-Nearest Neighbors with Optuna optimization and StandardScaler."""

    def _suggest_params(self, trial) -> Dict:
        return {
            "n_neighbors": trial.suggest_int(
                "n_neighbors", 3, 50
            ),  # Min=3 to avoid overfitting (k=1 memorizes)
            "weights": "uniform",  # Fixed to 'uniform' (distance weighting causes Train R²=1.0 due to self-inclusion)
            "p": trial.suggest_int("p", 1, 3),
            "leaf_size": trial.suggest_int("leaf_size", 10, 50),
        }

    def _build_model(self, params: Dict, n_jobs: int):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(**params, n_jobs=n_jobs)),
            ]
        )


class SVRPredictor(BasePredictor):
    """Support Vector Regression with Optuna optimization and StandardScaler."""

    def _suggest_params(self, trial) -> Dict:
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        params = {
            "kernel": kernel,
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.001, 1.0, log=True),
        }
        if kernel == "rbf":
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        elif kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        return params

    def _build_model(self, params: Dict, n_jobs: int):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(**params)),
            ]
        )


class MLPPredictor(BasePredictor):
    """Multi-Layer Perceptron with Optuna optimization and StandardScaler."""

    def _suggest_params(self, trial) -> Dict:
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                ["50", "100", "100_50", "100_100", "200", "200_100"],
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
            "max_iter": trial.suggest_int("max_iter", 300, 1000),
        }

    def _build_model(self, params: Dict, n_jobs: int):
        hls = params["hidden_layer_sizes"]
        # Handle both string (from Optuna) and tuple (from _process_best_params)
        if isinstance(hls, str):
            hls = tuple(map(int, hls.split("_"))) if "_" in hls else (int(hls),)
        # else: hls is already a tuple, use as-is
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=hls,
                        activation=params["activation"],
                        alpha=params["alpha"],
                        learning_rate_init=params["learning_rate_init"],
                        max_iter=params["max_iter"],
                        random_state=RANDOM_SEED,
                        early_stopping=True,
                        validation_fraction=0.1,
                    ),
                ),
            ]
        )

    def _process_best_params(self) -> None:
        hls_str = self.best_params_["hidden_layer_sizes"]
        self.best_params_["hidden_layer_sizes"] = (
            tuple(map(int, hls_str.split("_"))) if "_" in hls_str else (int(hls_str),)
        )


class GradientBoostingPredictor(BasePredictor):
    """Gradient Boosting with Optuna optimization."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.feature_importance_ = None

    def _suggest_params(self, trial) -> Dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5, 0.7]
            ),
        }

    def _build_model(self, params: Dict, n_jobs: int):
        return GradientBoostingRegressor(**params, random_state=RANDOM_SEED)

    def _post_fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

    def get_feature_importance(self) -> pd.Series:
        """Get Gradient Boosting feature importance rankings."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.feature_importance_


class ElasticNetPredictor(BasePredictor):
    """Elastic Net with Optuna optimization and StandardScaler."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.feature_names_ = None

    def _suggest_params(self, trial) -> Dict:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "max_iter": trial.suggest_categorical("max_iter", [1000, 5000, 10000]),
        }

    def _build_model(self, params: Dict, n_jobs: int):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(**params, random_state=RANDOM_SEED)),
            ]
        )

    def _post_fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.feature_names_ = X_train.columns.tolist()

    def get_coefficients(self) -> pd.Series:
        """Get model coefficients sorted by absolute value."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        if self.feature_names_ is None:
            raise ValueError("Feature names not stored.")
        elastic_net_model = self.model.named_steps["model"]
        coefficients = pd.Series(elastic_net_model.coef_, index=self.feature_names_)
        return coefficients.reindex(
            coefficients.abs().sort_values(ascending=False).index
        )


# ============================================================================
# PANEL DATA ECONOMETRIC MODELS
# ============================================================================


class PanelAnalyzer:
    # Panel data analysis with Dynamic Panel (Fixed Effects) estimation

    def __init__(
        self,
        data_path: Path,
        target: str,
        predictors: List[str],
        entity_col: str = "Country Name",
        time_col: str = "Year",
        logger: Optional[logging.Logger] = None,
    ):
        # Initialize PanelAnalyzer
        self.data_path = Path(data_path)
        self.target = target
        self.predictors = predictors
        self.entity_col = entity_col
        self.time_col = time_col
        self.logger = logger or self._create_logger()

        # Placeholders
        self.df: Optional[pd.DataFrame] = None
        self.df_panel: Optional[pd.DataFrame] = None
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.dynamic_results = None

    def _create_logger(self) -> logging.Logger:
        # Create default logger if none provided
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def load_data(self) -> "PanelAnalyzer":
        # Load and validate panel dataset
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        try:
            self.df = pd.read_csv(self.data_path)
        except Exception as e:
            raise IOError(f"Failed to read CSV file: {e}")

        if self.df.empty:
            raise ValueError("Loaded dataframe is empty")

        required_cols = [self.entity_col, self.time_col, self.target] + self.predictors
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.logger.info(f"Loaded {len(self.df):,} observations")
        self.logger.info(f"Entities: {self.df[self.entity_col].nunique()}")
        self.logger.info(
            f"Period: {self.df[self.time_col].min():.0f} - {self.df[self.time_col].max():.0f}"
        )

        return self

    def prepare_panel_structure(self) -> "PanelAnalyzer":
        # Convert dataframe to panel structure with multi-index
        if self.df is None:
            raise RuntimeError("Must call load_data() first")

        if self.entity_col not in self.df.columns:
            raise ValueError(f"Entity column '{self.entity_col}' not found")
        if self.time_col not in self.df.columns:
            raise ValueError(f"Time column '{self.time_col}' not found")

        try:
            self.df_panel = self.df.set_index(
                [self.entity_col, self.time_col]
            ).sort_index()
            self.logger.info("Panel structure created")
            self.logger.info(
                f"Entities: {self.df_panel.index.get_level_values(0).nunique()}"
            )
            self.logger.info(
                f"Time periods: {self.df_panel.index.get_level_values(1).nunique()}"
            )
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to create panel structure: {e}")

    def train_test_split(
        self,
        test_years: Optional[int] = None,
        train_end_year: Optional[int] = None,
        test_ratio: float = 0.2,
    ) -> "PanelAnalyzer":
        # Split panel data by time (temporal split for out-of-sample validation)
        if self.df_panel is None:
            raise RuntimeError("Must call prepare_panel_structure() first")

        max_year = self.df_panel.index.get_level_values(1).max()
        min_year = self.df_panel.index.get_level_values(1).min()

        # Determine cutoff year
        if train_end_year is not None:
            # Use explicit train end year
            train_cutoff = train_end_year + 1
        elif test_years is not None:
            # Use test_years parameter (original behavior)
            train_cutoff = max_year - test_years + 1
        else:
            # Calculate based on test_ratio (default 80/20 split)
            total_years = max_year - min_year + 1
            test_years_calc = int(total_years * test_ratio)
            train_cutoff = max_year - test_years_calc + 1

        self.df_train = self.df_panel[
            self.df_panel.index.get_level_values(1) < train_cutoff
        ].copy()
        self.df_test = self.df_panel[
            self.df_panel.index.get_level_values(1) >= train_cutoff
        ].copy()

        train_years = self.df_train.index.get_level_values(1).unique()
        test_years_actual = self.df_test.index.get_level_values(1).unique()

        self.logger.info(f"Train/Test split: cutoff year = {train_cutoff}")
        self.logger.info(
            f"Training period: {train_years.min():.0f}-{train_years.max():.0f}"
        )
        self.logger.info(
            f"Test period: {test_years_actual.min():.0f}-{test_years_actual.max():.0f}"
        )
        self.logger.info(f"Training set: {len(self.df_train)} observations")
        self.logger.info(f"Test set: {len(self.df_test)} observations")

        return self

    def create_lagged_variable(self, variable: str, lags: int = 1) -> "PanelAnalyzer":
        # Create lagged variable for dynamic panel models
        if self.df_panel is None:
            raise RuntimeError("Must call prepare_panel_structure() first")

        try:
            lag_name = f"{variable}_lag{lags}"
            self.df_panel[lag_name] = self.df_panel.groupby(level=0)[variable].shift(
                lags
            )

            if self.df_train is not None:
                self.df_train[lag_name] = self.df_train.groupby(level=0)[
                    variable
                ].shift(lags)
            if self.df_test is not None:
                self.df_test[lag_name] = self.df_test.groupby(level=0)[variable].shift(
                    lags
                )

            self.logger.info(f"Created lagged variable: {lag_name}")
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to create lagged variable: {e}")

    def fit_dynamic_panel(
        self, lags: int = 1, use_train_only: bool = False
    ) -> "PanelAnalyzer":
        # Fit dynamic panel model with lagged dependent variable
        # Create lagged variable and store lag info
        lag_name = f"{self.target}_lag{lags}"
        self.dynamic_lags_ = lags
        self.dynamic_lag_name_ = lag_name

        if lag_name not in self.df_panel.columns:
            self.create_lagged_variable(self.target, lags)

        df = self.df_train if use_train_only else self.df_panel
        df_clean = df.dropna()

        self.logger.info("Fitting Dynamic Panel model...")
        self.logger.info(f"Observations after lagging: {len(df_clean)}")

        y_dynamic = df_clean[self.target]
        X_dynamic = df_clean[self.predictors + [lag_name]]

        try:
            dynamic_model = PanelOLS(
                y_dynamic, X_dynamic, entity_effects=True, time_effects=True
            )
            self.dynamic_results = dynamic_model.fit(
                cov_type="clustered", cluster_entity=True
            )

            self.logger.info(
                f"Dynamic Panel R^2 (within): {self.dynamic_results.rsquared_within:.4f}"
            )
            self.logger.info("Dynamic Panel model fitted successfully")

            return self
        except Exception as e:
            self.logger.error(f"Dynamic Panel model failed: {e}")
            raise

    def analyze_persistence(self) -> Dict[str, float]:
        """
        Analyze persistence of shocks using lagged coefficient.

        Args:
            None

        Returns:
            Dict[str, float]: Dictionary containing coefficient, p-value, significance, and optionally half-life
        """
        if self.dynamic_results is None:
            raise RuntimeError("Must fit dynamic panel model first")

        # Use stored lag name instead of hardcoding lag1
        lag_name = self.dynamic_lag_name_
        if lag_name not in self.dynamic_results.params.index:
            raise RuntimeError("Lagged variable not found in model")

        # Extract lagged coefficient (rho) from regression results
        lag_coef = self.dynamic_results.params[lag_name]
        lag_pval = self.dynamic_results.pvalues[lag_name]

        # Test statistical significance at 5% level
        significant = lag_pval < 0.05

        self.logger.info(f"Persistence coefficient (rho): {lag_coef:.4f}")
        self.logger.info(f"P-value: {lag_pval:.6f}")
        self.logger.info(f"Significant: {'YES' if significant else 'NO'}")

        result = {
            "coefficient": lag_coef,
            "p_value": lag_pval,
            "significant": significant,
        }

        if 0 < lag_coef < 1:
            # Calculate half-life: time for shock to decay to 50% of original magnitude
            half_life = -np.log(2) / np.log(lag_coef)
            # Classify persistence strength based on coefficient magnitude
            interpretation = (
                "STRONG" if lag_coef > 0.7 else "MODERATE" if lag_coef > 0.4 else "WEAK"
            )
            result["half_life"] = half_life
            result["interpretation"] = interpretation

            self.logger.info(f"Half-life: {half_life:.2f} years")
            self.logger.info(f"Interpretation: {interpretation} persistence")

        return result

    def calculate_nickell_bias(self) -> Dict[str, float]:
        """
        Calculate Nickell bias for dynamic panel model with fixed effects.

        The Nickell (1981) bias arises in dynamic panel models with fixed effects
        when the lagged dependent variable is correlated with the entity-specific effects.

        Formula: Bias(α̂) ≈ -(1+α)/(T-1)

        Where:
            α = autoregressive coefficient (persistence parameter)
            T = number of time periods

        The bias is:
            - NEGATIVE (downward bias) -> OLS underestimates persistence
            - Decreases as T increases -> negligible when T >> 10
            - More severe for small T and high α

        Returns:
            Dict containing:
                - bias: Theoretical Nickell bias
                - bias_percentage: Bias as % of OLS coefficient
                - alpha_ols: Biased OLS estimate
                - alpha_corrected: Bias-corrected estimate
                - half_life_ols: Half-life using biased estimate (years)
                - half_life_corrected: Half-life using corrected estimate (years)
                - N: Number of entities (countries)
                - T: Number of time periods
                - interpretation: Text interpretation of bias severity

        References:
            Nickell, S. (1981). "Biases in Dynamic Models with Fixed Effects"
            Econometrica, 49(6), 1417-1426.
        """
        if self.dynamic_results is None:
            raise RuntimeError("Must fit dynamic panel model first")

        # Get dimensions
        if self.df_train is not None:
            df_analysis = self.df_train
        else:
            df_analysis = self.df_panel

        df_clean = df_analysis.dropna()
        N = df_clean.index.get_level_values(0).nunique()
        T = df_clean.index.get_level_values(1).nunique()

        # Get autoregressive coefficient
        lag_name = self.dynamic_lag_name_
        alpha_ols = self.dynamic_results.params[lag_name]

        # Calculate Nickell bias: Bias(α̂) = -(1+α)/(T-1)
        nickell_bias = -(1 + alpha_ols) / (T - 1)

        # Bias-corrected coefficient
        alpha_corrected = alpha_ols - nickell_bias

        # Bias magnitude as percentage
        bias_percentage = abs(nickell_bias / alpha_ols) * 100

        # Calculate half-life for both estimates
        # Half-life = -ln(2) / ln(α)
        half_life_ols = np.nan
        half_life_corrected = np.nan

        if 0 < alpha_ols < 1:
            half_life_ols = -np.log(2) / np.log(alpha_ols)

        if 0 < alpha_corrected < 1:
            half_life_corrected = -np.log(2) / np.log(alpha_corrected)

        # Interpret bias severity
        if bias_percentage < 5:
            interpretation = "NEGLIGIBLE"
            recommendation = f"T={T} is sufficiently large; OLS estimates are reliable"
        elif bias_percentage < 15:
            interpretation = "MODERATE"
            recommendation = "Consider bias correction or GMM estimation for robustness"
        else:
            interpretation = "SEVERE"
            recommendation = (
                "GMM estimation strongly recommended; OLS estimates unreliable"
            )

        # Log results
        self.logger.info("=" * 70)
        self.logger.info("NICKELL BIAS ANALYSIS")
        self.logger.info("=" * 70)
        self.logger.info(f"Panel dimensions: N={N} entities, T={T} periods")
        self.logger.info(f"Autoregressive coefficient (α̂_OLS): {alpha_ols:.4f}")
        self.logger.info(f"Nickell bias: {nickell_bias:.4f} ({bias_percentage:.1f}%)")
        self.logger.info(f"Bias-corrected coefficient (α̂*): {alpha_corrected:.4f}")
        self.logger.info(f"Bias severity: {interpretation}")
        self.logger.info(f"Recommendation: {recommendation}")

        if not np.isnan(half_life_ols) and not np.isnan(half_life_corrected):
            hl_diff = half_life_corrected - half_life_ols
            hl_diff_pct = (hl_diff / half_life_ols) * 100
            self.logger.info(f"Half-life (OLS): {half_life_ols:.2f} years")
            self.logger.info(f"Half-life (corrected): {half_life_corrected:.2f} years")
            self.logger.info(
                f"Half-life difference: {hl_diff:+.2f} years ({hl_diff_pct:+.1f}%)"
            )

        return {
            "bias": nickell_bias,
            "bias_percentage": bias_percentage,
            "alpha_ols": alpha_ols,
            "alpha_corrected": alpha_corrected,
            "half_life_ols": half_life_ols,
            "half_life_corrected": half_life_corrected,
            "N": N,
            "T": T,
            "interpretation": interpretation,
            "recommendation": recommendation,
        }

    def evaluate_on_test(self) -> Dict[str, float]:
        # Evaluate dynamic panel model on test set
        if self.dynamic_results is None:
            raise RuntimeError("Must fit dynamic panel model first")
        if self.df_test is None:
            raise RuntimeError("Must call train_test_split() first")

        # Use stored lag name instead of hardcoding lag1
        lag_name = self.dynamic_lag_name_
        df_test_clean = self.df_test.dropna()

        if len(df_test_clean) == 0:
            raise ValueError("Test set is empty after removing NaN values")

        y_test = df_test_clean[self.target]
        X_test = df_test_clean[self.predictors + [lag_name]]

        # Get predictions (entity and time effects handled by linearmodels)
        try:
            y_pred = self.dynamic_results.predict(exog=X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.logger.info("=" * 50)
            self.logger.info("TEST SET EVALUATION")
            self.logger.info("=" * 50)
            self.logger.info(f"Test observations: {len(y_test)}")
            self.logger.info(f"RMSE: {rmse:.4f}")
            self.logger.info(f"MAE: {mae:.4f}")
            self.logger.info(f"R^2: {r2:.4f}")

            return {
                "n_test": len(y_test),
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "y_pred": y_pred.values if hasattr(y_pred, "values") else y_pred,
                "y_test": y_test.values if hasattr(y_test, "values") else y_test,
                "test_index": df_test_clean.index,
            }
        except Exception as e:
            self.logger.error(f"Test evaluation failed: {e}")
            raise

    def get_summary(self) -> pd.DataFrame:
        # Get summary of all fitted models
        summary_data = []

        if self.dynamic_results is not None:
            summary_data.append(
                {
                    "Model": "Dynamic Panel (FE)",
                    "R^2 (within)": self.dynamic_results.rsquared_within,
                    "R^2 (overall)": self.dynamic_results.rsquared_overall,
                    "N": self.dynamic_results.nobs,
                }
            )

        return pd.DataFrame(summary_data)

    # =========================================================================
    # ECONOMETRIC DIAGNOSTIC TESTS (POST-ESTIMATION)
    # =========================================================================

    def run_diagnostic_tests(self) -> Dict[str, Dict]:
        # Run econometric diagnostic tests (Hausman, autocorrelation, heteroscedasticity)
        if self.dynamic_results is None:
            raise RuntimeError(
                "Must fit dynamic panel model first. Call fit_dynamic_panel()"
            )

        self.logger.info("\n" + "=" * 70)
        self.logger.info("ECONOMETRIC DIAGNOSTIC TESTS")
        self.logger.info("=" * 70)

        diagnostics = {}

        # Test 1: Hausman Test (FE vs RE)
        try:
            diagnostics["hausman"] = self._hausman_test()
        except Exception as e:
            self.logger.warning(f"Hausman test failed: {e}")
            diagnostics["hausman"] = {"error": str(e)}

        # Test 2: Autocorrelation Test
        try:
            diagnostics["autocorrelation"] = self._test_autocorrelation()
        except Exception as e:
            self.logger.warning(f"Autocorrelation test failed: {e}")
            diagnostics["autocorrelation"] = {"error": str(e)}

        # Test 3: Heteroscedasticity Test
        try:
            diagnostics["heteroscedasticity"] = self._test_heteroscedasticity()
        except Exception as e:
            self.logger.warning(f"Heteroscedasticity test failed: {e}")
            diagnostics["heteroscedasticity"] = {"error": str(e)}

        self.logger.info("=" * 70)

        return diagnostics

    def _hausman_test(self) -> Dict[str, float]:
        # Hausman Test: Fixed Effects vs Random Effects specification
        self.logger.info("\n1. HAUSMAN TEST (Fixed Effects vs Random Effects)")
        self.logger.info("-" * 70)

        try:
            # Golden Rule A: Use exact sample from fitted model
            y = self.dynamic_results.model.dependent
            X = self.dynamic_results.model.exog

            # Refit FE/RE on same sample with unadjusted covariance for Hausman
            fe = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
                cov_type="unadjusted"
            )
            re = RandomEffects(y, X).fit(cov_type="unadjusted")

            b_fe = fe.params
            b_re = re.params

            V_fe = fe.cov
            V_re = re.cov

            diff = b_fe - b_re
            V_diff = V_fe - V_re

            # Stable inversion (use pseudo-inverse if needed)
            try:
                inv_V = np.linalg.inv(V_diff.values)
            except np.linalg.LinAlgError:
                inv_V = np.linalg.pinv(V_diff.values)

            stat = float(diff.values.T @ inv_V @ diff.values)
            df = int(len(diff))
            pval = float(1 - stats.chi2.cdf(stat, df))

            conclusion = "Prefer FE (RE rejected)" if pval < 0.05 else "RE acceptable"

            self.logger.info(f"Hausman chi^2 statistic: {stat:.4f}")
            self.logger.info(f"Degrees of freedom: {df}")
            self.logger.info(f"P-value: {pval:.6f}")
            self.logger.info(f"Conclusion: {conclusion}")

            return {
                "test": "Hausman (unadjusted cov)",
                "statistic": stat,
                "df": df,
                "p_value": pval,
                "conclusion": conclusion,
            }

        except Exception as e:
            self.logger.warning(f"Hausman test failed: {e}")
            return {
                "test": "Hausman (unadjusted cov)",
                "error": str(e),
                "conclusion": "Using Fixed Effects (standard for panel data)",
            }

    def _test_autocorrelation(self) -> Dict[str, float]:
        # AR(1) test on panel residuals via auxiliary regression
        self.logger.info("\n2. AUTOCORRELATION TEST (AR1 Auxiliary Regression)")
        self.logger.info("-" * 70)

        try:
            import statsmodels.api as sm

            # Get residuals from fitted model
            resids = self.dynamic_results.resids.copy()
            df = pd.DataFrame({"e": resids})
            df["entity"] = df.index.get_level_values(0)
            df["time"] = df.index.get_level_values(1)

            # Lag residuals by entity
            df["e_lag"] = df.groupby("entity")["e"].shift(1)
            df = df.dropna(subset=["e", "e_lag"])

            if len(df) == 0:
                raise ValueError("Insufficient data after lagging residuals")

            # Auxiliary regression: e_it = rho * e_i,t-1 + u_it (no constant)
            aux = sm.OLS(df["e"].values, df[["e_lag"]].values).fit()

            rho = float(aux.params[0])
            pval = float(aux.pvalues[0])

            conclusion = (
                "Autocorrelation (AR1) detected"
                if pval < 0.05
                else "No evidence of AR1"
            )

            self.logger.info(f"Rho (AR1 coefficient): {rho:.4f}")
            self.logger.info(f"P-value: {pval:.6f}")
            self.logger.info(f"Conclusion: {conclusion}")
            self.logger.info(
                f"Note: Model uses clustered SE to handle potential autocorrelation"
            )

            return {
                "test": "Auxiliary AR(1) on residuals",
                "rho": rho,
                "p_value": pval,
                "conclusion": conclusion,
            }

        except Exception as e:
            self.logger.warning(f"Autocorrelation test failed: {e}")
            return {
                "test": "Auxiliary AR(1) on residuals",
                "error": str(e),
                "conclusion": "Test failed - using clustered SE as precaution",
            }

    def _test_heteroscedasticity(self) -> Dict[str, float]:
        # Breusch-Pagan Test: Heteroscedasticity in residuals
        self.logger.info("\n3. HETEROSCEDASTICITY TEST (Breusch-Pagan)")
        self.logger.info("-" * 70)

        try:
            import statsmodels.api as sm
            from statsmodels.stats.diagnostic import het_breuschpagan

            # Golden Rule A: Use exact sample from fitted model
            # Extract raw numpy arrays from PanelData objects
            resid = self.dynamic_results.resids
            if hasattr(resid, "dataframe"):
                # PanelData object
                resid = resid.dataframe.values.ravel()
            elif hasattr(resid, "values"):
                resid = resid.values.ravel()
            else:
                resid = np.asarray(resid).ravel()

            X = self.dynamic_results.model.exog
            if hasattr(X, "dataframe"):
                # PanelData object
                X = X.dataframe.values
            elif hasattr(X, "values"):
                X = X.values
            else:
                X = np.asarray(X)

            # Ensure we have pure numpy arrays
            resid = np.asarray(resid, dtype=float).ravel()
            X = np.asarray(X, dtype=float)

            # BP expects a constant in exog
            X_bp = np.column_stack([np.ones(len(X)), X])

            lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(resid, X_bp)

            conclusion = (
                "Heteroscedasticity detected"
                if lm_pvalue < 0.05
                else "No evidence of heteroscedasticity"
            )

            self.logger.info(f"LM statistic: {lm_stat:.4f}")
            self.logger.info(f"LM p-value: {lm_pvalue:.6f}")
            self.logger.info(f"F statistic: {f_stat:.4f}")
            self.logger.info(f"F p-value: {f_pvalue:.6f}")
            self.logger.info(f"Conclusion: {conclusion}")
            self.logger.info(
                f"Note: Model uses clustered SE to handle potential heteroscedasticity"
            )

            return {
                "test": "Breusch-Pagan",
                "lm_stat": float(lm_stat),
                "lm_pvalue": float(lm_pvalue),
                "f_stat": float(f_stat),
                "f_pvalue": float(f_pvalue),
                "conclusion": conclusion,
            }

        except Exception as e:
            self.logger.warning(f"Heteroscedasticity test failed: {e}")
            return {
                "test": "Breusch-Pagan",
                "error": str(e),
                "conclusion": "Test failed - using robust SE as precaution",
            }


# ============================================================================
# TRAINING ORCHESTRATION - Train all ML models
# ============================================================================


def train_ml_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results_dir: Path,
    cv: int = 5,
    n_jobs: int = -1,
) -> List[Dict]:
    """
    Train all 7 ML models and return results.

    Business logic for model training - separated from presentation layer.
    Instantiates, trains, evaluates models and saves detailed results.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        results_dir: Directory to save detailed result files
        cv: Cross-validation folds (default: 5)
        n_jobs: Parallel jobs for training (default: -1)

    Returns:
        List of dicts with model results: {model, model_obj, y_pred, r2, mae, rmse, time, ...}
    """
    import time

    # Ensure results directory exists
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate all 7 ML models
    models = [
        ("Random Forest", RandomForestPredictor()),
        ("XGBoost", XGBoostPredictor()),
        ("Gradient Boosting", GradientBoostingPredictor()),
        ("Elastic Net", ElasticNetPredictor()),
        ("SVR", SVRPredictor()),
        ("KNN", KNNPredictor()),
        ("MLP", MLPPredictor()),
    ]

    results = []

    # Train each model
    for model_name, model in models:
        start_time = time.time()

        try:
            # Train with Optuna Bayesian optimization
            model.fit(X_train, y_train, cv=cv, n_jobs=n_jobs)

            # Evaluate on test set
            metrics = model.evaluate(X_test, y_test)

            elapsed = time.time() - start_time

            # Generate predictions
            y_pred = model.predict(X_test)

            # Store comprehensive results
            results.append(
                {
                    "model": model_name,
                    "model_obj": model,
                    "y_pred": y_pred,
                    "y_test": y_test,
                    "X_test": X_test,
                    "X_train": X_train,
                    "r2": metrics["r2"],
                    "adj_r2": metrics["adj_r2"],
                    "f_stat": metrics.get("f_stat"),
                    "f_pvalue": metrics.get("f_pvalue"),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "time": elapsed,
                    "train_score": metrics.get("train_score"),
                    "cv_score": metrics.get("cv_score"),
                    "overfitting_gap": metrics.get("overfitting_gap"),
                    "has_overfitting": metrics.get("has_overfitting", False),
                    "n_samples": metrics.get("n_samples"),
                    "n_features": metrics.get("n_features"),
                }
            )

            # Save detailed results to text file
            results_file = (
                results_dir / f"ml_{model_name.lower().replace(' ', '_')}_results.txt"
            )
            _save_detailed_results(
                results_file,
                model_name,
                model,
                metrics,
                X_train,
                y_train,
                X_test,
                y_test,
                elapsed,
            )

        except Exception as e:
            # Store error result
            results.append(
                {
                    "model": model_name,
                    "error": str(e),
                    "time": time.time() - start_time,
                }
            )

    return results


def _save_detailed_results(
    filepath: Path,
    model_name: str,
    model,
    metrics: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    elapsed: float,
) -> None:
    """Save detailed model results to text file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        f.write(
            f"""{"=" * 100}
{model_name.upper()} RESULTS - POLITICAL STABILITY PREDICTION
{"=" * 100}

CONFIGURATION
{"-" * 100}
Target: political_stability
Predictors: {', '.join(X_train.columns)}
Train period: 1996-2017
Test period: 2018-2023
Train size: {len(X_train)}
Test size: {len(X_test)}
Training time: {elapsed:.2f}s

"""
        )

        if hasattr(model, "best_params_") and model.best_params_:
            f.write(f"BEST HYPERPARAMETERS\n{'-' * 100}\n")
            for param, value in sorted(model.best_params_.items()):
                f.write(f"{param}: {value}\n")
            f.write("\n")

        if metrics.get("train_score") is not None:
            overfit_status = (
                "[WARNING]  YES (gap > 0.1)"
                if metrics.get("has_overfitting", False)
                else "OK NO (gap < 0.1)"
            )
            f.write(
                f"""OVERFITTING DIAGNOSTICS
{"-" * 100}
Mean Train R^2: {metrics['train_score']:.4f}
Mean CV R^2: {metrics['cv_score']:.4f}
Train-CV Gap: {metrics['overfitting_gap']:.4f}
Overfitting Status: {overfit_status}

"""
            )

        f.write(
            f"""CROSS-VALIDATION RESULTS
{"-" * 100}
Best CV R^2: {metrics.get('cv_score', 'N/A') if metrics.get('cv_score') is None else f"{metrics['cv_score']:.4f}"}

TEST SET PERFORMANCE
{"-" * 100}
Test R^2: {metrics['r2']:.4f}
Test Adjusted R^2: {metrics['adj_r2']:.4f}
Test RMSE: {metrics['rmse']:.4f}
Test MAE: {metrics['mae']:.4f}
Number of samples: {metrics.get('n_samples', 'N/A')}
Number of features: {metrics.get('n_features', 'N/A')}

"""
        )

        if metrics.get("f_stat") is not None:
            if metrics["f_pvalue"] < 0.001:
                sig = "*** (p < 0.001) - Highly significant"
            elif metrics["f_pvalue"] < 0.01:
                sig = "** (p < 0.01) - Very significant"
            elif metrics["f_pvalue"] < 0.05:
                sig = "* (p < 0.05) - Significant"
            else:
                sig = "Not significant (p >= 0.05)"
            f.write(
                f"""MODEL SIGNIFICANCE (F-STATISTIC)
{"-" * 100}
F-statistic: {metrics['f_stat']:.4f}
F p-value: {metrics['f_pvalue']:.6f}
Significance: {sig}

"""
            )

        if hasattr(model.model, "feature_importances_"):
            importances = model.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            f.write(f"FEATURE IMPORTANCE\n{'-' * 100}\n")
            for i in indices:
                f.write(f"{X_train.columns[i]:30s}: {importances[i]:.4f}\n")
            f.write("\n")
