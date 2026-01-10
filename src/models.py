"""
Machine Learning and Econometric Models for Political Stability Prediction.

This module implements a comprehensive suite of regression models for predicting
political stability across countries using macroeconomic and governance indicators.
Includes both modern machine learning approaches and classical econometric methods
for panel data analysis.

Model Architecture
------------------
All models inherit from BasePredictor and implement a consistent interface:
- fit(X, y, param_grid, n_trials, cv): Train with Bayesian hyperparameter optimization
- predict(X): Generate predictions for new data
- evaluate(X, y): Compute comprehensive performance metrics
- save(path) / load(path): Serialize trained models for reuse

Available Models
----------------
Machine Learning:
    RandomForestPredictor: Ensemble of decision trees with bootstrap aggregation
    XGBoostPredictor: Gradient boosting with regularization and early stopping
    GradientBoostingPredictor: Sequential boosting with gradient descent optimization
    ElasticNetPredictor: Linear regression with L1/L2 regularization for feature selection
    SVRPredictor: Support vector regression with RBF kernel for non-linear relationships
    KNNPredictor: k-Nearest neighbors averaging for local pattern recognition
    MLPPredictor: Multi-layer perceptron neural network with backpropagation

Econometric:
    PanelAnalyzer: Dynamic panel fixed effects model with lag structure and
                   comprehensive diagnostics (Hausman, autocorrelation, heteroscedasticity)

Hyperparameter Optimization
----------------------------
All ML models use Optuna for Bayesian optimization with:
- Temporal cross-validation preserving chronological order
- Trial pruning for computational efficiency
- Reproducible results via fixed random seed

Evaluation Metrics
------------------
- R-squared and Adjusted R-squared for goodness of fit
- RMSE and MAE for prediction error magnitude
- F-statistic for overall model significance
- Cross-validation scores for generalization assessment

Notes
-----
The module ensures reproducibility through fixed random seeds (RANDOM_SEED = 42)
and uses scikit-learn pipelines for preprocessing integration.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import optuna
import pandas as pd

# ============================================================================
# PANEL DATA & ECONETRIC LIBRARIES
# ============================================================================
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.diagnostic import het_breuschpagan

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================================
# PANEL-SAFE CV SPLITTER (WALK-FORWARD BY YEAR)
# ============================================================================


class PanelYearForwardSplit(BaseCrossValidator):
    """
    Expanding-window, walk-forward split by YEAR.

    For each split:
      - Train: all years <= train_end_year
      - Valid: next `valid_years` year(s)

    Works for panel data when each row has a year (from MultiIndex or a column).
    """

    def __init__(
        self, years: np.ndarray, min_train_years: int = 8, valid_years: int = 1
    ):
        years = np.asarray(years)
        if years.ndim != 1:
            raise ValueError("years must be a 1D array-like")
        self.years = years
        self.unique_years = np.unique(years[~pd.isna(years)])
        self.min_train_years = int(min_train_years)
        self.valid_years = int(valid_years)

        if len(self.unique_years) < (self.min_train_years + self.valid_years):
            raise ValueError(
                f"Not enough unique years ({len(self.unique_years)}) for "
                f"min_train_years={self.min_train_years} and valid_years={self.valid_years}"
            )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        uy = self.unique_years
        return max(0, len(uy) - self.min_train_years - self.valid_years + 1)

    def split(self, X, y=None, groups=None):
        uy = self.unique_years
        for i in range(self.min_train_years, len(uy) - self.valid_years + 1):
            train_years = uy[:i]
            valid_years = uy[i : i + self.valid_years]

            train_idx = np.where(np.isin(self.years, train_years))[0]
            valid_idx = np.where(np.isin(self.years, valid_years))[0]

            # If a year is sparse, a fold might be tiny; caller may decide to skip.
            yield train_idx, valid_idx


def _infer_years_from_X(X: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Try to infer years from:
    - MultiIndex level named 'Year' or second level
    - Column named 'Year'
    Returns a 1D np.array of years (aligned to rows of X) or None.
    """
    # Column case
    if "Year" in X.columns:
        return pd.to_numeric(X["Year"], errors="coerce").to_numpy()

    # MultiIndex case
    if isinstance(X.index, pd.MultiIndex):
        names = list(X.index.names)
        if "Year" in names:
            lvl = names.index("Year")
            return pd.to_numeric(
                X.index.get_level_values(lvl), errors="coerce"
            ).to_numpy()

        # Common pattern: (Country, Year)
        if len(names) >= 2:
            return pd.to_numeric(
                X.index.get_level_values(1), errors="coerce"
            ).to_numpy()

    return None


def _default_panel_cv_splitter(
    X_train: pd.DataFrame, cv: int, logger: logging.Logger
) -> BaseCrossValidator:
    """
    If we can infer years, return a PanelYearForwardSplit.
    Otherwise fall back to KFold but warn loudly.
    """
    years = _infer_years_from_X(X_train)
    if years is not None and np.isfinite(pd.to_numeric(years, errors="coerce")).any():
        unique_years = np.unique(pd.to_numeric(years, errors="coerce"))
        unique_years = unique_years[~pd.isna(unique_years)]
        # Heuristic: require at least cv+min_train_years years; else reduce min_train_years
        min_train_years = max(5, min(8, max(5, len(unique_years) - cv)))
        try:
            return PanelYearForwardSplit(
                years=years, min_train_years=min_train_years, valid_years=1
            )
        except Exception as e:
            logger.warning(
                f"Could not build PanelYearForwardSplit (reason: {e}). Falling back to KFold."
            )
            return KFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    logger.warning(
        "Could not infer years from X_train (no 'Year' column, no MultiIndex year). "
        "Falling back to KFold. This may be inappropriate for temporal prediction."
    )
    return KFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def regression_metrics(
    y_true: pd.Series, y_pred: np.ndarray, n_features: int
) -> Dict[str, float]:
    """
    Compute comprehensive regression performance metrics with statistical safeguards.

    Calculates standard regression metrics including goodness of fit, error magnitudes,
    and statistical significance tests. Implements safeguards for edge cases where
    sample size is insufficient relative to the number of features.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth target values
    y_pred : np.ndarray
        Model predictions for the same observations
    n_features : int
        Number of features used in the model (for degrees of freedom calculations)

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - r2: Coefficient of determination (proportion of variance explained)
        - adj_r2: Adjusted R-squared penalizing model complexity (NaN if n <= p+1)
        - f_stat: F-statistic for overall model significance (NaN if invalid)
        - f_pvalue: P-value for F-statistic (NaN if invalid)
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - n_samples: Number of observations
        - n_features: Number of features (input parameter)

    Notes
    -----
    Adjusted R-squared and F-statistic require n > p + 1 (sufficient degrees of freedom).
    Returns NaN for these metrics when conditions are not met, ensuring numerical stability.

    Examples
    --------
    >>> y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_pred = np.array([1.1, 2.0, 2.9, 4.1, 4.9])
    >>> metrics = regression_metrics(y_true, y_pred, n_features=2)
    >>> print(f"R-squared: {metrics['r2']:.3f}")
    >>> print(f"RMSE: {metrics['rmse']:.3f}")
    """
    n = int(len(y_true))
    p = int(n_features)
    r2 = float(r2_score(y_true, y_pred))

    # Adjusted R^2 only defined if n > p + 1
    if n > p + 1:
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = np.nan

    # F-statistic only meaningful if n > p + 1 and 0 <= r2 < 1
    if (p > 0) and (n > p + 1) and (r2 < 1.0) and (r2 >= 0.0):
        f_stat = (r2 / p) / ((1.0 - r2) / (n - p - 1))
        f_pvalue = float(1.0 - stats.f.cdf(f_stat, p, n - p - 1))
    else:
        f_stat = np.nan
        f_pvalue = np.nan

    return {
        "r2": r2,
        "adj_r2": float(adj_r2) if np.isfinite(adj_r2) else np.nan,
        "f_stat": float(f_stat) if np.isfinite(f_stat) else np.nan,
        "f_pvalue": float(f_pvalue) if np.isfinite(f_pvalue) else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_samples": n,
        "n_features": p,
    }


# ============================================================================
# MACHINE LEARNING MODELS - BASE CLASS
# ============================================================================


class BasePredictor:
    """
    Abstract base class for machine learning predictors with Bayesian hyperparameter optimization.

    Provides a unified interface for regression models with Optuna-based hyperparameter
    tuning, temporal cross-validation, and comprehensive model evaluation. All concrete
    predictor classes inherit from this base and implement the _get_default_param_grid()
    and _create_model() methods.

    Attributes
    ----------
    model : sklearn estimator
        Fitted scikit-learn model (Pipeline with StandardScaler + regressor)
    best_params_ : dict
        Optimal hyperparameters found by Optuna optimization
    train_score_ : float
        R-squared score on training data
    cv_score_ : float
        Mean cross-validation R-squared score
    overfitting_gap_ : float
        Difference between train_score and cv_score (overfitting indicator)
    cv_results : dict
        Detailed cross-validation results including per-fold scores
    logger : logging.Logger
        Logger instance for tracking optimization progress

    Methods
    -------
    fit(X, y, param_grid, n_trials, cv)
        Train model with Bayesian hyperparameter optimization
    predict(X)
        Generate predictions for new data
    evaluate(X_test, y_test)
        Compute comprehensive performance metrics on test set
    save(path)
        Serialize trained model to disk
    load(path)
        Deserialize trained model from disk

    Notes
    -----
    All models use scikit-learn Pipelines with StandardScaler preprocessing
    to ensure proper feature scaling. Temporal cross-validation preserves
    chronological order when cv is a TimeSeriesSplit object.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.model = None
        self.grid_search = None  # legacy attribute
        self.best_params_ = None
        self.train_score_ = None
        self.cv_score_ = None
        self.overfitting_gap_ = None
        self.logger = logger or logging.getLogger(__name__)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        y_pred = self.predict(X_test)
        metrics = regression_metrics(y_test, y_pred, X_test.shape[1])

        metrics.update(
            {
                "train_score": self.train_score_,
                "cv_score": self.cv_score_,
                "overfitting_gap": self.overfitting_gap_,
                "has_overfitting": (self.overfitting_gap_ > 0.1)
                if self.overfitting_gap_ is not None
                else None,
            }
        )
        return metrics

    def save_model(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path) -> "BasePredictor":
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from: {filepath}")
        return self

    def get_cv_results(self) -> pd.DataFrame:
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
        cv: Union[int, BaseCrossValidator] = 5,
        n_jobs: int = -1,
        param_grid: Optional[Dict] = None,  # deprecated
        verbose: int = 0,  # deprecated
    ) -> "BasePredictor":
        """
        Optuna Bayesian optimization with TPE + MedianPruner.

        KEY CHANGE:
        - If cv is an int, we build a panel-safe time CV (walk-forward by year)
          when years can be inferred from X_train. Otherwise we fall back to KFold.
        - We evaluate folds manually to enable Optuna pruning (trial.report per fold).
        """
        self.logger.info(
            f"Starting Optuna optimization with {n_trials} trials and CV={cv}"
        )

        # Build CV splitter
        if isinstance(cv, int):
            cv_splitter: BaseCrossValidator = _default_panel_cv_splitter(
                X_train, cv=cv, logger=self.logger
            )
        else:
            cv_splitter = cv

        def _cv_score_for_params(trial, params: Dict) -> float:
            model = self._build_model(params, n_jobs=n_jobs)
            fold_scores = []

            # Manual CV loop to allow pruning
            for fold_idx, (tr_idx, va_idx) in enumerate(
                cv_splitter.split(X_train, y_train)
            ):
                if len(tr_idx) == 0 or len(va_idx) == 0:
                    continue

                X_tr = X_train.iloc[tr_idx]
                y_tr = y_train.iloc[tr_idx]
                X_va = X_train.iloc[va_idx]
                y_va = y_train.iloc[va_idx]

                m = clone(model)
                m.fit(X_tr, y_tr)
                pred = m.predict(X_va)
                score = float(r2_score(y_va, pred))
                fold_scores.append(score)

                # Report intermediate performance for pruning
                trial.report(np.mean(fold_scores), step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if not fold_scores:
                # If something went wrong with split, be explicit
                raise ValueError(
                    "No valid CV folds produced non-empty train/valid splits."
                )
            return float(np.mean(fold_scores))

        def objective(trial):
            params = self._suggest_params(trial)
            return _cv_score_for_params(trial, params)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

        self.best_params_ = study.best_params
        self.cv_score_ = float(study.best_value)
        self._process_best_params()

        self.logger.info(f"Best CV R^2 score: {self.cv_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params_}")

        self.model = self._build_model(self.best_params_, n_jobs=n_jobs)
        self.model.fit(X_train, y_train)

        self.train_score_ = float(self.model.score(X_train, y_train))
        self.overfitting_gap_ = float(self.train_score_ - self.cv_score_)

        self.logger.info(f"Train R^2: {self.train_score_:.4f}")
        self.logger.info(f"Train-CV gap: {self.overfitting_gap_:.4f}")

        if self.overfitting_gap_ > 0.1:
            self.logger.warning("[WARNING] Potential overfitting detected (gap > 0.1)")
        else:
            self.logger.info("[OK] No significant overfitting (gap < 0.1)")

        self._post_fit(X_train, y_train)
        return self

    def _suggest_params(self, trial) -> Dict:
        raise NotImplementedError

    def _build_model(self, params: Dict, n_jobs: int):
        raise NotImplementedError

    def _process_best_params(self) -> None:
        pass

    def _post_fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
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
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
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
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.feature_importance_


class KNNPredictor(BasePredictor):
    """K-Nearest Neighbors with Optuna optimization and StandardScaler."""

    def _suggest_params(self, trial) -> Dict:
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": "uniform",
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
        if isinstance(hls, str):
            hls = tuple(map(int, hls.split("_"))) if "_" in hls else (int(hls),)
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
    """
    Dynamic Panel Fixed Effects Econometric Model for Political Stability Analysis.

    Implements a comprehensive econometric approach for panel data regression with
    dynamic structure (lagged dependent variable), entity fixed effects, and extensive
    diagnostic testing. Provides classical econometric benchmark for comparison with
    machine learning approaches.

    Model Specification
    -------------------
    Y_it = α + βY_{i,t-1} + γX_it + μ_i + ε_it

    Where:
        - Y_it: Political stability for country i at time t
        - Y_{i,t-1}: Lagged dependent variable (autoregressive component)
        - X_it: Vector of macroeconomic and governance predictors
        - μ_i: Country-specific fixed effects
        - ε_it: Idiosyncratic error term

    Diagnostic Tests
    ----------------
    - Hausman Test: Tests for correlation between effects and regressors
    - Breusch-Pagan Test: Detects heteroscedasticity in residuals
    - Durbin-Watson / AR(1) Test: Checks for serial correlation

    Attributes
    ----------
    df_panel : pd.DataFrame
        Complete panel data with lagged variables (MultiIndex: entity, time)
    df_train : pd.DataFrame
        Training subset of panel data
    df_test : pd.DataFrame
        Test subset of panel data
    dynamic_results : linearmodels.panel.results.PanelEffectsResults
        Fitted fixed effects model results with coefficients and diagnostics
    data_path : Path
        Path to raw data directory
    target : str
        Name of dependent variable
    predictors : List[str]
        List of predictor variable names

    Methods
    -------
    load_and_prepare_data(train_end_year)
        Load raw data, create lags, and split into train/test
    fit(train_end_year, drop_missing_lag)
        Estimate dynamic panel model on training data
    predict(X)
        Generate out-of-sample predictions for test set
    evaluate(X, y)
        Compute regression metrics on test data
    get_diagnostics()
        Run comprehensive econometric diagnostic tests

    Notes
    -----
    The model handles the Nickell bias inherent in dynamic panels with fixed effects.
    Predictions are generated using the fitted coefficients and observed lag structure.
    All diagnostics are computed post-estimation on fitted model residuals.

    Examples
    --------
    >>> analyzer = PanelAnalyzer(
    ...     data_path=Path('data/raw'),
    ...     target='political_stability',
    ...     predictors=['gdp_per_capita', 'unemployment', 'inflation']
    ... )
    >>> analyzer.load_and_prepare_data(train_end_year=2017)
    >>> analyzer.fit(train_end_year=2017)
    >>> diagnostics = analyzer.get_diagnostics()
    >>> print(f"Hausman p-value: {diagnostics['hausman']['p_value']:.4f}")
    """

    def __init__(
        self,
        data_path: Path,
        target: str,
        predictors: List[str],
        entity_col: str = "Country Name",
        time_col: str = "Year",
        logger: Optional[logging.Logger] = None,
    ):
        self.data_path = Path(data_path)
        self.target = target
        self.predictors = predictors
        self.entity_col = entity_col
        self.time_col = time_col
        self.logger = logger or self._create_logger()

        self.df: Optional[pd.DataFrame] = None
        self.df_panel: Optional[pd.DataFrame] = None
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.dynamic_results = None

        # store split cutoff for consistent re-slicing after lagging
        self._train_cutoff_year: Optional[int] = None

        # dynamic lag metadata
        self.dynamic_lags_: Optional[int] = None
        self.dynamic_lag_name_: Optional[str] = None

    def _create_logger(self) -> logging.Logger:
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
        if self.df is None:
            raise RuntimeError("Must call load_data() first")

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
        if self.df_panel is None:
            raise RuntimeError("Must call prepare_panel_structure() first")

        max_year = int(self.df_panel.index.get_level_values(1).max())
        min_year = int(self.df_panel.index.get_level_values(1).min())

        if train_end_year is not None:
            train_cutoff = int(train_end_year) + 1
        elif test_years is not None:
            train_cutoff = max_year - int(test_years) + 1
        else:
            total_years = max_year - min_year + 1
            test_years_calc = max(1, int(total_years * float(test_ratio)))
            train_cutoff = max_year - test_years_calc + 1

        # store cutoff for later re-slicing after lag creation
        self._train_cutoff_year = int(train_cutoff)

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
        """
        KEY CHANGE:
        - Compute lag ONLY on df_panel (full continuity).
        - If train/test already exist, re-slice them from df_panel using the stored cutoff year.
          This preserves the last-train-year → first-test-year lag continuity.
        """
        if self.df_panel is None:
            raise RuntimeError("Must call prepare_panel_structure() first")

        lag_name = f"{variable}_lag{lags}"
        self.df_panel[lag_name] = self.df_panel.groupby(level=0)[variable].shift(lags)

        # Re-slice train/test if already split
        if self._train_cutoff_year is not None:
            cutoff = self._train_cutoff_year
            self.df_train = self.df_panel[
                self.df_panel.index.get_level_values(1) < cutoff
            ].copy()
            self.df_test = self.df_panel[
                self.df_panel.index.get_level_values(1) >= cutoff
            ].copy()

        self.logger.info(f"Created lagged variable on full panel: {lag_name}")
        return self

    def fit_dynamic_panel(
        self, lags: int = 1, use_train_only: bool = False
    ) -> "PanelAnalyzer":
        lag_name = f"{self.target}_lag{lags}"
        self.dynamic_lags_ = lags
        self.dynamic_lag_name_ = lag_name

        if self.df_panel is None:
            raise RuntimeError("Must call prepare_panel_structure() first")

        if lag_name not in self.df_panel.columns:
            self.create_lagged_variable(self.target, lags)

        df = (
            self.df_train
            if (use_train_only and self.df_train is not None)
            else self.df_panel
        )
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
        if self.dynamic_results is None:
            raise RuntimeError("Must fit dynamic panel model first")

        lag_name = self.dynamic_lag_name_
        if lag_name not in self.dynamic_results.params.index:
            raise RuntimeError("Lagged variable not found in model")

        lag_coef = float(self.dynamic_results.params[lag_name])
        lag_pval = float(self.dynamic_results.pvalues[lag_name])
        significant = lag_pval < 0.05

        result = {
            "coefficient": lag_coef,
            "p_value": lag_pval,
            "significant": significant,
        }

        if 0 < lag_coef < 1:
            half_life = float(-np.log(2) / np.log(lag_coef))
            interpretation = (
                "STRONG" if lag_coef > 0.7 else "MODERATE" if lag_coef > 0.4 else "WEAK"
            )
            result["half_life"] = half_life
            result["interpretation"] = interpretation

        return result

    def calculate_nickell_bias(self) -> Dict[str, float]:
        if self.dynamic_results is None:
            raise RuntimeError("Must fit dynamic panel model first")

        df_analysis = self.df_train if self.df_train is not None else self.df_panel
        df_clean = df_analysis.dropna()

        N = int(df_clean.index.get_level_values(0).nunique())
        T = int(df_clean.index.get_level_values(1).nunique())

        lag_name = self.dynamic_lag_name_
        alpha_ols = float(self.dynamic_results.params[lag_name])

        nickell_bias = float(-(1 + alpha_ols) / (T - 1))
        alpha_corrected = float(alpha_ols - nickell_bias)
        bias_percentage = (
            float(abs(nickell_bias / alpha_ols) * 100) if alpha_ols != 0 else np.inf
        )

        half_life_ols = np.nan
        half_life_corrected = np.nan
        if 0 < alpha_ols < 1:
            half_life_ols = float(-np.log(2) / np.log(alpha_ols))
        if 0 < alpha_corrected < 1:
            half_life_corrected = float(-np.log(2) / np.log(alpha_corrected))

        if bias_percentage < 5:
            interpretation = "NEGLIGIBLE"
            recommendation = (
                f"T={T} is sufficiently large; OLS estimates are likely acceptable"
            )
        elif bias_percentage < 15:
            interpretation = "MODERATE"
            recommendation = "Consider bias correction or GMM estimation for robustness"
        else:
            interpretation = "SEVERE"
            recommendation = (
                "GMM estimation strongly recommended; FE-OLS persistence likely biased"
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
        if self.dynamic_results is None:
            raise RuntimeError("Must fit dynamic panel model first")
        if self.df_test is None:
            raise RuntimeError("Must call train_test_split() first")

        lag_name = self.dynamic_lag_name_
        df_test_clean = self.df_test.dropna()

        if len(df_test_clean) == 0:
            raise ValueError("Test set is empty after removing NaN values")

        y_test = df_test_clean[self.target]
        X_test = df_test_clean[self.predictors + [lag_name]]

        try:
            y_pred = self.dynamic_results.predict(exog=X_test)

            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            return {
                "n_test": int(len(y_test)),
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "y_pred": y_pred.values
                if hasattr(y_pred, "values")
                else np.asarray(y_pred),
                "y_test": y_test.values
                if hasattr(y_test, "values")
                else np.asarray(y_test),
                "test_index": df_test_clean.index,
            }
        except Exception as e:
            self.logger.error(f"Test evaluation failed: {e}")
            raise

    def get_summary(self) -> pd.DataFrame:
        summary_data = []
        if self.dynamic_results is not None:
            summary_data.append(
                {
                    "Model": "Dynamic Panel (FE)",
                    "R^2 (within)": float(self.dynamic_results.rsquared_within),
                    "R^2 (overall)": float(self.dynamic_results.rsquared_overall),
                    "N": int(self.dynamic_results.nobs),
                }
            )
        return pd.DataFrame(summary_data)

    def run_diagnostic_tests(self) -> Dict[str, Dict]:
        if self.dynamic_results is None:
            raise RuntimeError(
                "Must fit dynamic panel model first. Call fit_dynamic_panel()"
            )

        diagnostics = {}
        try:
            diagnostics["hausman"] = self._hausman_test()
        except Exception as e:
            self.logger.warning(f"Hausman test failed: {e}")
            diagnostics["hausman"] = {"error": str(e)}

        try:
            diagnostics["autocorrelation"] = self._test_autocorrelation()
        except Exception as e:
            self.logger.warning(f"Autocorrelation test failed: {e}")
            diagnostics["autocorrelation"] = {"error": str(e)}

        try:
            diagnostics["heteroscedasticity"] = self._test_heteroscedasticity()
        except Exception as e:
            self.logger.warning(f"Heteroscedasticity test failed: {e}")
            diagnostics["heteroscedasticity"] = {"error": str(e)}

        return diagnostics

    def _hausman_test(self) -> Dict[str, float]:
        self.logger.info("\n1. HAUSMAN TEST (Fixed Effects vs Random Effects)")

        y = self.dynamic_results.model.dependent
        X = self.dynamic_results.model.exog

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

        try:
            inv_V = np.linalg.inv(V_diff.values)
        except np.linalg.LinAlgError:
            inv_V = np.linalg.pinv(V_diff.values)

        stat = float(diff.values.T @ inv_V @ diff.values)
        df = int(len(diff))
        pval = float(1 - stats.chi2.cdf(stat, df))
        conclusion = "Prefer FE (RE rejected)" if pval < 0.05 else "RE acceptable"

        return {
            "test": "Hausman (unadjusted cov)",
            "statistic": stat,
            "df": df,
            "p_value": pval,
            "conclusion": conclusion,
        }

    def _test_autocorrelation(self) -> Dict[str, float]:
        self.logger.info("\n2. AUTOCORRELATION TEST (AR1 Auxiliary Regression)")
        import statsmodels.api as sm

        resids = self.dynamic_results.resids.copy()
        df = pd.DataFrame({"e": resids})
        df["entity"] = df.index.get_level_values(0)
        df["time"] = df.index.get_level_values(1)

        df["e_lag"] = df.groupby("entity")["e"].shift(1)
        df = df.dropna(subset=["e", "e_lag"])
        if len(df) == 0:
            raise ValueError("Insufficient data after lagging residuals")

        aux = sm.OLS(df["e"].values, df[["e_lag"]].values).fit()
        rho = float(aux.params[0])
        pval = float(aux.pvalues[0])

        conclusion = (
            "Autocorrelation (AR1) detected" if pval < 0.05 else "No evidence of AR1"
        )
        return {
            "test": "Auxiliary AR(1) on residuals",
            "rho": rho,
            "p_value": pval,
            "conclusion": conclusion,
        }

    def _test_heteroscedasticity(self) -> Dict[str, float]:
        self.logger.info("\n3. HETEROSCEDASTICITY TEST (Breusch-Pagan)")

        resid = self.dynamic_results.resids
        if hasattr(resid, "dataframe"):
            resid = resid.dataframe.values.ravel()
        elif hasattr(resid, "values"):
            resid = resid.values.ravel()
        else:
            resid = np.asarray(resid).ravel()

        X = self.dynamic_results.model.exog
        if hasattr(X, "dataframe"):
            X = X.dataframe.values
        elif hasattr(X, "values"):
            X = X.values
        else:
            X = np.asarray(X)

        resid = np.asarray(resid, dtype=float).ravel()
        X = np.asarray(X, dtype=float)

        X_bp = np.column_stack([np.ones(len(X)), X])
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(resid, X_bp)

        conclusion = (
            "Heteroscedasticity detected"
            if lm_pvalue < 0.05
            else "No evidence of heteroscedasticity"
        )
        return {
            "test": "Breusch-Pagan",
            "lm_stat": float(lm_stat),
            "lm_pvalue": float(lm_pvalue),
            "f_stat": float(f_stat),
            "f_pvalue": float(f_pvalue),
            "conclusion": conclusion,
        }


# ============================================================================
# TRAINING ORCHESTRATION
# ============================================================================


def train_ml_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results_dir: Path,
    cv: Union[int, BaseCrossValidator] = 5,
    n_jobs: int = -1,
    n_trials: int = 100,
) -> List[Dict]:
    """
    Train all ML models and return results.

    KEY CHANGE:
    - cv can be an int OR a custom CV splitter.
    - if cv is int, BasePredictor.fit() will build a panel-safe temporal CV when possible.
    """
    import time

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("Random Forest", RandomForestPredictor()),
        ("XGBoost", XGBoostPredictor()) if XGBOOST_AVAILABLE else ("XGBoost", None),
        ("Gradient Boosting", GradientBoostingPredictor()),
        ("Elastic Net", ElasticNetPredictor()),
        ("SVR", SVRPredictor()),
        ("KNN", KNNPredictor()),
        ("MLP", MLPPredictor()),
    ]

    results = []

    for model_name, model in models:
        start_time = time.time()

        if model is None:
            results.append(
                {
                    "model": model_name,
                    "error": "XGBoost not available",
                    "time": time.time() - start_time,
                }
            )
            continue

        try:
            model.fit(X_train, y_train, cv=cv, n_jobs=n_jobs, n_trials=n_trials)
            metrics = model.evaluate(X_test, y_test)
            elapsed = time.time() - start_time
            y_pred = model.predict(X_test)

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

        except Exception as e:
            results.append(
                {"model": model_name, "error": str(e), "time": time.time() - start_time}
            )

    return results
