"""
Evaluation and Visualization Functions

This module contains all functions for:
- Displaying model results (benchmarks, test scores)
- Creating visualizations (plots, charts)
- Comparing model performances

"""

# Visualization Resources:
# - Predicted vs Actual: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PredictionErrorDisplay.html
# - Residual Analysis:   https://seaborn.pydata.org/generated/seaborn.residplot.html
# - Feature Importance:  https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, learning_curve

COLORS = {
    "primary": "#1f77b4",
    "secondary": "#4a90c4",
    "tertiary": "#2c5f8d",
    "accent": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#d62728",
    "neutral": "#7f7f7f",
}

BLUE_PALETTE = ["#e3f2fd", "#90caf9", "#42a5f5", "#1e88e5", "#1565c0", "#0d47a1"]
DIVERGING_PALETTE = ["#1f77b4", "#ff7f0e"]

# Geographic regions for regional analysis
REGIONS = {
    "Europe": [
        "France",
        "Germany",
        "Italy",
        "Spain",
        "United Kingdom",
        "Poland",
        "Netherlands",
        "Belgium",
        "Greece",
        "Portugal",
        "Sweden",
        "Austria",
        "Switzerland",
        "Norway",
        "Denmark",
        "Finland",
        "Ireland",
    ],
    "Asia": [
        "China",
        "India",
        "Japan",
        "South Korea",
        "Indonesia",
        "Thailand",
        "Malaysia",
        "Singapore",
        "Philippines",
        "Vietnam",
        "Pakistan",
        "Bangladesh",
    ],
    "Americas": [
        "United States",
        "Canada",
        "Brazil",
        "Mexico",
        "Argentina",
        "Chile",
        "Colombia",
        "Peru",
        "Venezuela",
    ],
    "Africa": [
        "South Africa",
        "Nigeria",
        "Egypt",
        "Kenya",
        "Ghana",
        "Ethiopia",
        "Morocco",
        "Algeria",
        "Tunisia",
    ],
    "Middle East": [
        "Saudi Arabia",
        "United Arab Emirates",
        "Israel",
        "Turkey",
        "Iran",
    ],
}


# ============================================================================
# INTERNAL HELPERS
# ============================================================================


def _timestamped_path(output_dir: Path, prefix: str, extension: str = "png") -> Path:
    """Generate timestamped filepath for plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{prefix}_{timestamp}.{extension}"


def _save_fig(fig_or_plt, filepath: Path, dpi: int = 300, close: bool = True):
    """Save matplotlib figure and optionally close it."""
    from matplotlib.figure import Figure

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(fig_or_plt, Figure):
        fig_or_plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        if close:
            plt.close(fig_or_plt)
    else:
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        if close:
            plt.close()

    return filepath


def _make_grid(n_items: int, n_cols: int = 3, base_w: int = 18, base_h: int = 6):
    """Create standardized grid of subplots for multiple models."""
    n_rows = math.ceil(n_items / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(base_w, base_h * n_rows))
    axes = np.array(axes).reshape(-1) if n_items > 1 else [axes]
    return fig, axes, n_rows, n_cols


def _hide_unused_axes(axes, used: int):
    """Hide unused axes in grid layout."""
    for ax in axes[used:]:
        ax.set_visible(False)


def _annotate_barh(ax, values, fmt="{:.4f}", dx=0.0):
    """Add value labels to horizontal bar chart."""
    for i, v in enumerate(values):
        ax.text(
            v + dx, i, f" {fmt.format(v)}", va="center", fontsize=10, fontweight="bold"
        )


def add_region_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add Region column to DataFrame based on Country Name."""

    def get_region(country):
        """Map country name to its corresponding region, returns 'Other' if not found."""
        for region, countries in REGIONS.items():
            if country in countries:
                return region
        return "Other"

    df = df.copy()
    if "Country Name" in df.columns:
        df["Region"] = df["Country Name"].apply(get_region)
    return df


# ============================================================================
# DISPLAY AND COMPARISON FUNCTIONS
# ============================================================================


def display_results_table(results_df: pd.DataFrame, title: str = "Model Results"):
    """Display model results in a professional formatted table with box-drawing characters."""
    import numpy as np

    results_sorted = results_df.sort_values("r2", ascending=False).reset_index(
        drop=True
    )

    # Table dimensions
    table_width = 82

    # Print header
    print(f"\n{'=' * table_width}")
    print(f"{title:^{table_width}}")
    print(f"{'=' * table_width}")

    # Column headers
    print(f"┌{'─' * 6}┬{'─' * 25}┬{'─' * 16}┬{'─' * 16}┬{'─' * 16}┐")
    print(
        f"│ {'Rank':^4} │ {'Model':^23} │ {'R² Score':^14} │ {'MAE':^14} │ {'RMSE':^14} │"
    )
    print(f"├{'─' * 6}┼{'─' * 25}┼{'─' * 16}┼{'─' * 16}┼{'─' * 16}┤")

    # Data rows
    for idx, row in results_sorted.iterrows():
        rank = idx + 1
        model = row["model"] if "model" in row else row.name
        r2 = row["r2"]
        rmse = row.get("rmse", 0)
        mae = row.get("mae", 0)

        # Format values
        r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
        mae_str = f"{mae:.4f}" if not np.isnan(mae) else "N/A"
        rmse_str = f"{rmse:.4f}" if not np.isnan(rmse) else "N/A"

        print(
            f"│ {rank:^4} │ {model:<23} │ {r2_str:^14} │ {mae_str:^14} │ {rmse_str:^14} │"
        )

    # Footer
    print(f"└{'─' * 6}┴{'─' * 25}┴{'─' * 16}┴{'─' * 16}┴{'─' * 16}┘")
    print(f"{'=' * table_width}\n")


def compare_models(benchmark_df: pd.DataFrame, test_df: pd.DataFrame):
    """Compare training (benchmark) vs test performance for all models."""
    print(f"\n{'=' * 80}")
    print(f"{'MODEL COMPARISON: TRAINING VS TEST':^80}")
    print("=" * 80)
    print(f"{'Model':<25} {'Train R^2':>12} {'Test R^2':>12} {'Difference':>12}")
    print("-" * 80)

    comparison = pd.merge(
        benchmark_df[["model", "r2"]],
        test_df[["model", "r2"]],
        on="model",
        suffixes=("_train", "_test"),
    )

    comparison["diff"] = comparison["r2_train"] - comparison["r2_test"]
    comparison = comparison.sort_values("r2_test", ascending=False)

    for _, row in comparison.iterrows():
        print(
            f"{row['model']:<25} {row['r2_train']:>12.4f} {row['r2_test']:>12.4f} {row['diff']:>12.4f}"
        )

    print("=" * 80)
    print("\nInterpretation:")
    print("- Difference > 0.1: Potential overfitting (model memorized training data)")
    print("- Difference ~= 0: Good generalization")
    print("- Difference < 0: Underfitting (model too simple)")


# ============================================================================
# SIMPLE PLOTTING FUNCTIONS
# ============================================================================


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
    show_metrics: bool = True,
):
    """Create scatter plot of actual vs predicted values."""
    plt.figure(figsize=(10, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.6,
        color=COLORS["primary"],
        edgecolors="k",
        linewidth=0.5,
    )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color=COLORS["warning"],
        linestyle="--",
        lw=2,
        label="Perfect Prediction",
    )

    if show_metrics:
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        metrics_text = f"R^2 = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}"
        plt.text(
            0.05,
            0.95,
            metrics_text,
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            verticalalignment="top",
            fontsize=10,
        )

    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(f"{model_name}: Predicted vs Actual", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return _save_fig(plt, output_path)


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_path: Path
):
    """Create residual plot to check for patterns (model diagnostics)."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(
        y_pred,
        residuals,
        alpha=0.6,
        color=COLORS["primary"],
        edgecolors="k",
        linewidth=0.5,
    )
    axes[0].axhline(y=0, color=COLORS["warning"], linestyle="--", lw=2)
    axes[0].set_xlabel("Predicted Values", fontsize=12)
    axes[0].set_ylabel("Residuals", fontsize=12)
    axes[0].set_title("Residuals vs Predicted", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        residuals, bins=30, color=COLORS["primary"], edgecolor="black", alpha=0.7
    )
    axes[1].axvline(x=0, color=COLORS["warning"], linestyle="--", lw=2)
    axes[1].set_xlabel("Residuals", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Residuals Distribution", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"{model_name}: Residual Analysis", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    return _save_fig(fig, output_path)


def plot_feature_importance(
    importance: pd.Series, model_name: str, output_path: Path, top_n: int = 10
):
    """Plot top N feature importances."""
    plt.figure(figsize=(10, 6))

    top_features = importance.head(top_n)
    n_features = len(top_features)
    colors = [BLUE_PALETTE[min(i, len(BLUE_PALETTE) - 1)] for i in range(n_features)]
    top_features.sort_values().plot(kind="barh", color=colors, edgecolor="black")

    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.title(
        f"{model_name}: Top {top_n} Feature Importances", fontsize=14, fontweight="bold"
    )
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    return _save_fig(plt, output_path)


def plot_model_comparison(
    results_df: pd.DataFrame, output_path: Path, metric: str = "r2"
):
    """Create bar chart comparing models on a single metric."""
    plt.figure(figsize=(12, 6))

    results_sorted = results_df.sort_values(metric, ascending=(metric != "r2"))

    n_models = len(results_sorted)
    blue_gradient = [
        BLUE_PALETTE[min(i, len(BLUE_PALETTE) - 1)] for i in range(n_models)
    ]
    bars = plt.bar(
        range(n_models), results_sorted[metric], color=blue_gradient, edgecolor="black"
    )

    plt.xlabel("Model", fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.title(f"Model Comparison: {metric.upper()}", fontsize=14, fontweight="bold")
    plt.xticks(
        range(len(results_sorted)), results_sorted["model"], rotation=45, ha="right"
    )
    plt.grid(True, axis="y", alpha=0.3)

    for i, (bar, value) in enumerate(zip(bars, results_sorted[metric])):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    return _save_fig(plt, output_path)


def create_correlation_heatmap(
    data: pd.DataFrame, output_path: Path, title: str = "Feature Correlation Matrix"
):
    """Create correlation heatmap for features."""
    plt.figure(figsize=(10, 8))

    corr = data.corr()

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    return _save_fig(plt, output_path)


def save_results_summary(
    results_df: pd.DataFrame, output_path: Path, description: str = ""
):
    """Save results to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    if description:
        print(f"OK {description}: {output_path}")
    else:
        print(f"OK Saved: {output_path}")


# ============================================================================
# COMPREHENSIVE VISUALIZATION FUNCTIONS
# ============================================================================


def generate_model_comparison_chart(
    models_data: list,
    output_dir: Path,
    panel_metrics: dict = None,
    highlight_best: bool = False,
):
    """
    Generate 3-metric comparison chart (R²/MAE/RMSE) for all models.

    Args:
        models_data: List of model result dictionaries
        output_dir: Directory to save the plot
        panel_metrics: Optional panel model metrics to include
        highlight_best: If True, highlight best models with gold edge and show benchmark line
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    names = [m["model"] for m in models_data]
    r2_scores = [m["r2"] for m in models_data]
    mae_scores = [m["mae"] for m in models_data]
    rmse_scores = [m["rmse"] for m in models_data]

    # Add Dynamic Panel if available
    if panel_metrics:
        names.append("Dynamic Panel (FE)")
        r2_scores.append(panel_metrics["r2"])
        mae_scores.append(panel_metrics.get("mae", 0))
        rmse_scores.append(panel_metrics["rmse"])

    num_total_models = len(names)

    # Colors: ML models in primary blue, Panel in accent orange if highlight_best
    if highlight_best and panel_metrics:
        colors = [COLORS["primary"]] * len(models_data) + [COLORS["accent"]]
    else:
        colors = [COLORS["primary"]] * num_total_models

    # R^2 comparison
    bars1 = axes[0].barh(names, r2_scores, color=colors, alpha=0.8, edgecolor="black")
    axes[0].set_xlabel("R^2 Score", fontsize=12, fontweight="bold")
    axes[0].set_title("Model Comparison - R^2", fontsize=14, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)
    _annotate_barh(axes[0], r2_scores)

    if highlight_best:
        best_idx = r2_scores.index(max(r2_scores))
        bars1[best_idx].set_edgecolor("gold")
        bars1[best_idx].set_linewidth(3)
        if panel_metrics:
            axes[0].axvline(
                x=panel_metrics["r2"],
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.5,
            )

    # MAE comparison
    bars2 = axes[1].barh(names, mae_scores, color=colors, alpha=0.8, edgecolor="black")
    axes[1].set_xlabel("MAE", fontsize=12, fontweight="bold")
    axes[1].set_title("Model Comparison - MAE", fontsize=14, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)
    _annotate_barh(axes[1], mae_scores)

    if highlight_best and mae_scores:
        best_mae_idx = mae_scores.index(min([m for m in mae_scores if m > 0]))
        bars2[best_mae_idx].set_edgecolor("gold")
        bars2[best_mae_idx].set_linewidth(3)
        if panel_metrics and panel_metrics.get("mae"):
            axes[1].axvline(
                x=panel_metrics["mae"],
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.5,
            )

    # RMSE comparison
    bars3 = axes[2].barh(names, rmse_scores, color=colors, alpha=0.8, edgecolor="black")
    axes[2].set_xlabel("RMSE", fontsize=12, fontweight="bold")
    axes[2].set_title("Model Comparison - RMSE", fontsize=14, fontweight="bold")
    axes[2].grid(axis="x", alpha=0.3)
    _annotate_barh(axes[2], rmse_scores)

    if highlight_best:
        best_rmse_idx = rmse_scores.index(min(rmse_scores))
        bars3[best_rmse_idx].set_edgecolor("gold")
        bars3[best_rmse_idx].set_linewidth(3)
        if panel_metrics:
            axes[2].axvline(
                x=panel_metrics["rmse"],
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.5,
            )

    plt.tight_layout()

    output_file = _timestamped_path(output_dir, "model_comparison")
    return _save_fig(fig, output_file)


def generate_ml_vs_benchmark_comparison(
    ml_results: list, panel_metrics: dict, output_dir: Path
):
    """
    Generate comparison chart between ML models and econometric benchmark.

    Args:
        ml_results: List of ML model result dictionaries
        panel_metrics: Dictionary containing panel model metrics (r2, mae, rmse)
        output_dir: Directory to save the plot

    Returns:
        Path to the saved chart file
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ML model data
    ml_names = [m["model"] for m in ml_results]
    ml_r2 = [m["r2"] for m in ml_results]
    ml_mae = [m["mae"] for m in ml_results]
    ml_rmse = [m["rmse"] for m in ml_results]

    # Add benchmark
    all_names = ml_names + ["Dynamic Panel (FE)"]
    all_r2 = ml_r2 + [panel_metrics["r2"]]
    all_mae = ml_mae + [panel_metrics.get("mae", 0)]
    all_rmse = ml_rmse + [panel_metrics["rmse"]]

    # Colors: ML in blue, benchmark in orange
    colors = [COLORS["primary"]] * len(ml_results) + [COLORS["accent"]]

    # R² comparison
    bars1 = axes[0].barh(all_names, all_r2, color=colors, alpha=0.8, edgecolor="black")
    axes[0].set_xlabel("R² Score", fontsize=12, fontweight="bold")
    axes[0].set_title("ML vs Benchmark - R²", fontsize=14, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)
    _annotate_barh(axes[0], all_r2)

    # Highlight best
    best_r2_idx = all_r2.index(max(all_r2))
    bars1[best_r2_idx].set_edgecolor("gold")
    bars1[best_r2_idx].set_linewidth(3)

    # MAE comparison
    bars2 = axes[1].barh(all_names, all_mae, color=colors, alpha=0.8, edgecolor="black")
    axes[1].set_xlabel("MAE", fontsize=12, fontweight="bold")
    axes[1].set_title("ML vs Benchmark - MAE", fontsize=14, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)
    _annotate_barh(axes[1], all_mae)

    # Highlight best MAE
    if all_mae:
        best_mae_idx = all_mae.index(min([m for m in all_mae if m > 0]))
        bars2[best_mae_idx].set_edgecolor("gold")
        bars2[best_mae_idx].set_linewidth(3)

    # RMSE comparison
    bars3 = axes[2].barh(
        all_names, all_rmse, color=colors, alpha=0.8, edgecolor="black"
    )
    axes[2].set_xlabel("RMSE", fontsize=12, fontweight="bold")
    axes[2].set_title("ML vs Benchmark - RMSE", fontsize=14, fontweight="bold")
    axes[2].grid(axis="x", alpha=0.3)
    _annotate_barh(axes[2], all_rmse)

    # Highlight best RMSE
    best_rmse_idx = all_rmse.index(min(all_rmse))
    bars3[best_rmse_idx].set_edgecolor("gold")
    bars3[best_rmse_idx].set_linewidth(3)

    plt.tight_layout()

    output_file = _timestamped_path(output_dir, "ml_vs_benchmark_comparison")
    return _save_fig(fig, output_file)


def generate_feature_importance_plot(models_data: list, output_dir: Path):
    """Generate feature importance plot for best model (if tree-based)."""
    best_model = max(models_data, key=lambda x: x["r2"])
    X_test = best_model["X_test"]

    if hasattr(best_model["model_obj"].model, "feature_importances_"):
        fig, ax = plt.subplots(figsize=(10, 6))
        importances = best_model["model_obj"].model.feature_importances_
        indices = np.argsort(importances)[::-1]

        n_features = len(importances)
        colors = [
            BLUE_PALETTE[min(i, len(BLUE_PALETTE) - 1)] for i in range(n_features)
        ]
        ax.bar(range(n_features), importances[indices], color=colors, edgecolor="black")
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(
            [X_test.columns[i] for i in indices], rotation=45, ha="right"
        )
        ax.set_xlabel("Features", fontsize=12, fontweight="bold")
        ax.set_ylabel("Importance", fontsize=12, fontweight="bold")
        ax.set_title(
            f'Feature Importance - {best_model["model"]} (R^2={best_model["r2"]:.4f})',
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        output_file = _timestamped_path(output_dir, "feature_importance")
        return _save_fig(fig, output_file)
    else:
        print(
            f"Warning: {best_model['model']} does not have feature_importances_ attribute"
        )
        return None


def generate_predictions_plot(models_data: list, output_dir: Path):
    """Generate predictions vs actual scatter plots for all models."""
    n_models = len(models_data)
    fig, axes, n_rows, n_cols = _make_grid(n_models)

    for idx, model_info in enumerate(models_data):
        ax = axes[idx]
        y_test = model_info["y_test"]
        y_pred = model_info["y_pred"]

        ax.scatter(y_test, y_pred, alpha=0.6, s=30, edgecolors="k", linewidth=0.5)

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )

        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(
            f'{model_info["model"]}\nR^2={model_info["r2"]:.4f}',
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    _hide_unused_axes(axes, n_models)

    plt.suptitle(
        "Predictions vs Actual - All Models", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout()

    output_file = _timestamped_path(output_dir, "predictions_vs_actual")
    return _save_fig(fig, output_file)


def generate_distribution_plot(test_data: pd.DataFrame, output_dir: Path):
    """Generate distribution histogram of target variable."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        test_data["political_stability"],
        bins=30,
        alpha=0.7,
        color=COLORS["primary"],
        edgecolor="black",
    )
    ax.set_xlabel("Political Stability Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title(
        "Distribution of Political Stability (Test Set)", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = _timestamped_path(output_dir, "distribution_plot")
    return _save_fig(fig, output_file)


def generate_correlation_heatmap(test_data: pd.DataFrame, output_dir: Path):
    """Generate feature correlation heatmap."""
    output_file = _timestamped_path(output_dir, "correlation_heatmap")
    return create_correlation_heatmap(
        test_data, output_file, title="Feature Correlation Heatmap"
    )


def generate_residual_plots(models_data: list, output_dir: Path):
    """Generate residual diagnostic plots for all models."""
    n_models = len(models_data)
    fig, axes, n_rows, n_cols = _make_grid(n_models)

    for idx, model_info in enumerate(models_data):
        ax = axes[idx]
        y_test = model_info["y_test"]
        y_pred = model_info["y_pred"]
        residuals = y_test - y_pred

        ax.scatter(
            y_pred,
            residuals,
            alpha=0.6,
            color=COLORS["primary"],
            edgecolors="k",
            linewidths=0.5,
        )
        ax.axhline(y=0, color=COLORS["warning"], linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted Values", fontsize=10)
        ax.set_ylabel("Residuals", fontsize=10)
        ax.set_title(
            f'{model_info["model"]} (R^2={model_info["r2"]:.4f})',
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

    _hide_unused_axes(axes, n_models)

    plt.suptitle(
        "Residual Plots - Model Diagnostics", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout()

    output_file = _timestamped_path(output_dir, "residual_plots")
    return _save_fig(fig, output_file)


def generate_learning_curves(
    models_data: list, train_data: pd.DataFrame, output_dir: Path
):
    """Generate learning curves for overfitting/underfitting diagnosis."""
    X_train = train_data.drop("political_stability", axis=1)
    y_train = train_data["political_stability"]

    n_models = len(models_data)
    fig, axes, n_rows, n_cols = _make_grid(n_models)

    for idx, model_info in enumerate(models_data):
        try:
            ax = axes[idx]
            model_name = model_info["model"]
            model = model_info["model_obj"]

            print(
                f"  -> Computing learning curve for {model_name}... ({idx+1}/{len(models_data)})"
            )

            X_to_use = model_info.get("X_train", X_train)

            train_sizes, train_scores, val_scores = learning_curve(
                model.model,
                X_to_use,
                y_train,
                cv=3,
                scoring="r2",
                train_sizes=np.linspace(0.2, 1.0, 5),
                n_jobs=-1,
                shuffle=True,
            )

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            ax.plot(
                train_sizes,
                train_mean,
                "o-",
                color=COLORS["primary"],
                label="Training score",
            )
            ax.fill_between(
                train_sizes,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.2,
                color=COLORS["primary"],
            )
            ax.plot(
                train_sizes, val_mean, "o-", color=COLORS["success"], label="CV score"
            )
            ax.fill_between(
                train_sizes,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.2,
                color=COLORS["success"],
            )

            ax.set_xlabel("Training Set Size", fontsize=10)
            ax.set_ylabel("R^2 Score", fontsize=10)
            ax.set_title(f"{model_name}", fontsize=11, fontweight="bold")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(
                f"Warning: Could not generate learning curve for {model_name}: {str(e)}"
            )

    _hide_unused_axes(axes, n_models)

    plt.suptitle(
        "Learning Curves - Overfitting/Underfitting Diagnosis",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    output_file = _timestamped_path(output_dir, "learning_curves")
    return _save_fig(fig, output_file)


def generate_time_series_predictions(
    models_data: list,
    test_data: pd.DataFrame,
    output_dir: Path,
    panel_metrics: dict = None,
):
    """Generate time series predictions showing temporal evolution."""
    test_df = test_data.reset_index()
    y_test = test_df["political_stability"]
    years = test_df["Year"]

    total_models = len(models_data) + (
        1 if panel_metrics and panel_metrics.get("y_pred") is not None else 0
    )
    fig, axes, n_rows, n_cols = _make_grid(total_models, n_cols=3, base_w=20, base_h=6)

    # Plot ML models
    for idx, model_info in enumerate(models_data):
        ax = axes[idx]
        y_pred = model_info["y_pred"]

        year_data = pd.DataFrame({"Year": years, "Actual": y_test, "Predicted": y_pred})
        year_avg = year_data.groupby("Year").mean()

        ax.plot(
            year_avg.index,
            year_avg["Actual"],
            "o-",
            color=COLORS["primary"],
            label="Actual",
            linewidth=2,
            markersize=6,
        )
        ax.plot(
            year_avg.index,
            year_avg["Predicted"],
            "s-",
            color=COLORS["accent"],
            label="Predicted",
            linewidth=2,
            markersize=6,
            alpha=0.7,
        )

        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("Political Stability Score", fontsize=10)
        ax.set_title(
            f'{model_info["model"]} (R^2={model_info["r2"]:.4f})',
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Add Dynamic Panel if available
    if panel_metrics and panel_metrics.get("y_pred") is not None:
        idx = len(models_data)
        ax = axes[idx]

        panel_test_index = panel_metrics.get("test_index")
        if hasattr(panel_test_index, "get_level_values"):
            panel_years = panel_test_index.get_level_values("Year").values
        else:
            panel_years = years

        panel_y_test = panel_metrics.get("y_test")
        panel_y_pred = panel_metrics.get("y_pred")

        if hasattr(panel_y_test, "ravel"):
            panel_y_test = panel_y_test.ravel()
        if hasattr(panel_y_pred, "ravel"):
            panel_y_pred = panel_y_pred.ravel()
        if hasattr(panel_years, "ravel"):
            panel_years = panel_years.ravel()

        panel_year_data = pd.DataFrame(
            {"Year": panel_years, "Actual": panel_y_test, "Predicted": panel_y_pred}
        )
        panel_year_avg = panel_year_data.groupby("Year").mean()

        ax.plot(
            panel_year_avg.index,
            panel_year_avg["Actual"],
            "o-",
            color=COLORS["primary"],
            label="Actual",
            linewidth=2,
            markersize=6,
        )
        ax.plot(
            panel_year_avg.index,
            panel_year_avg["Predicted"],
            "s-",
            color=COLORS["accent"],
            label="Predicted",
            linewidth=2,
            markersize=6,
            alpha=0.7,
        )

        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("Political Stability Score", fontsize=10)
        ax.set_title(
            f'Dynamic Panel (FE) (R^2={panel_metrics["r2"]:.4f})',
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    _hide_unused_axes(axes, total_models)

    plt.suptitle(
        "Time Series Predictions - Temporal Evolution",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    output_file = _timestamped_path(output_dir, "time_series_predictions")
    return _save_fig(fig, output_file)


def generate_cv_scores_comparison(
    models_data: list, train_data: pd.DataFrame, output_dir: Path
):
    """Generate cross-validation scores comparison for all models."""
    X_train = train_data.drop("political_stability", axis=1)
    y_train = train_data["political_stability"]

    cv_results = []

    for model_info in models_data:
        try:
            model_name = model_info["model"]
            model = model_info["model_obj"]

            X_to_use = model_info.get("X_train", X_train)

            cv_scores = cross_val_score(
                model.model, X_to_use, y_train, cv=5, scoring="r2", n_jobs=-1
            )

            cv_results.append(
                {
                    "name": model_name,
                    "scores": cv_scores,
                    "mean": np.mean(cv_scores),
                    "std": np.std(cv_scores),
                }
            )

        except Exception as e:
            print(f"Warning: Could not compute CV for {model_name}: {str(e)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bp = ax1.boxplot(
        [r["scores"] for r in cv_results],
        labels=[r["name"] for r in cv_results],
        patch_artist=True,
        showmeans=True,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["secondary"])
        patch.set_alpha(0.7)

    ax1.set_ylabel("R^2 Score", fontsize=12)
    ax1.set_title(
        "Cross-Validation Scores Distribution", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis="x", rotation=45)

    names = [r["name"] for r in cv_results]
    means = [r["mean"] for r in cv_results]
    stds = [r["std"] for r in cv_results]

    x_pos = np.arange(len(names))
    ax2.barh(x_pos, means, xerr=stds, capsize=5, alpha=0.7, color=COLORS["primary"])
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("R^2 Score (Mean +/- Std)", fontsize=12)
    ax2.set_title("Cross-Validation Performance", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    for i, (mean, std) in enumerate(zip(means, stds)):
        ax2.text(
            mean,
            i,
            f" {mean:.4f}+/-{std:.4f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    output_file = _timestamped_path(output_dir, "cv_scores_comparison")
    return _save_fig(fig, output_file)


def generate_error_distribution(models_data: list, output_dir: Path):
    """Generate error distribution analysis with KDE for all models."""
    n_models = len(models_data)
    fig, axes, n_rows, n_cols = _make_grid(n_models)

    for idx, model_info in enumerate(models_data):
        ax = axes[idx]
        y_test = model_info["y_test"]
        y_pred = model_info["y_pred"]
        errors = y_test - y_pred

        ax.hist(
            errors,
            bins=30,
            alpha=0.6,
            color=COLORS["primary"],
            edgecolor="black",
            density=True,
            label="Error Distribution",
        )

        if np.var(errors) > 1e-10:
            kde = stats.gaussian_kde(errors)
            x_range = np.linspace(errors.min(), errors.max(), 100)
            ax.plot(
                x_range, kde(x_range), color=COLORS["warning"], linewidth=2, label="KDE"
            )
        else:
            ax.text(
                0.5,
                0.95,
                "Perfect predictions (variance=0)",
                transform=ax.transAxes,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
            )

        ax.axvline(
            x=0,
            color=COLORS["success"],
            linestyle="--",
            linewidth=2,
            label="Zero Error",
        )
        ax.axvline(
            x=np.mean(errors),
            color=COLORS["accent"],
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Mean: {np.mean(errors):.3f}",
        )

        ax.set_xlabel("Prediction Error", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(
            f'{model_info["model"]} (MAE={model_info["mae"]:.4f})',
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    _hide_unused_axes(axes, n_models)

    plt.suptitle(
        "Prediction Error Distribution Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    output_file = _timestamped_path(output_dir, "error_distribution")
    return _save_fig(fig, output_file)


def generate_regional_analysis(
    models_data: list, test_data: pd.DataFrame, output_dir: Path
):
    """Generate regional performance analysis for best model."""
    test_df = add_region_column(test_data.reset_index())

    best_model = max(models_data, key=lambda x: x["r2"])

    regional_perf = []
    for region in ["Europe", "Asia", "Americas", "Africa", "Middle East", "Other"]:
        region_mask = test_df["Region"] == region
        if region_mask.sum() > 0:
            y_true_region = test_df.loc[region_mask, "political_stability"]
            y_pred_region = best_model["y_pred"][region_mask.to_numpy()]

            r2 = r2_score(y_true_region, y_pred_region)
            mae = mean_absolute_error(y_true_region, y_pred_region)
            n_samples = region_mask.sum()

            regional_perf.append(
                {"Region": region, "R^2": r2, "MAE": mae, "Samples": n_samples}
            )

    regional_df = pd.DataFrame(regional_perf)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    n_regions = len(regional_df)
    region_colors = [
        BLUE_PALETTE[min(i, len(BLUE_PALETTE) - 1)] for i in range(n_regions)
    ]
    ax1.barh(regional_df["Region"], regional_df["R^2"], color=region_colors, alpha=0.7)
    ax1.set_xlabel("R^2 Score", fontsize=12)
    ax1.set_title(
        f'Regional Performance - {best_model["model"]}', fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, axis="x")

    for i, (r2, n) in enumerate(zip(regional_df["R^2"], regional_df["Samples"])):
        ax1.text(
            r2, i, f" {r2:.3f} (n={n})", va="center", fontsize=10, fontweight="bold"
        )

    ax2.barh(regional_df["Region"], regional_df["MAE"], color=region_colors, alpha=0.7)
    ax2.set_xlabel("Mean Absolute Error", fontsize=12)
    ax2.set_title(
        f'Regional Error - {best_model["model"]}', fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3, axis="x")

    for i, mae in enumerate(regional_df["MAE"]):
        ax2.text(mae, i, f" {mae:.3f}", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()

    output_file = _timestamped_path(output_dir, "regional_analysis")
    return _save_fig(fig, output_file)


def generate_political_stability_evolution(data: pd.DataFrame, output_dir: Path):
    """Generate simple plot showing average political stability over time."""
    df = data.copy()

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    if "Year" not in df.columns or "political_stability" not in df.columns:
        raise ValueError("Data must contain 'Year' and 'political_stability' columns")

    yearly_avg = (
        df.groupby("Year").agg({"political_stability": ["sum", "count"]}).reset_index()
    )
    yearly_avg.columns = ["Year", "sum", "count"]
    yearly_avg["average"] = yearly_avg["sum"] / yearly_avg["count"]

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(
        yearly_avg["Year"],
        yearly_avg["average"],
        color=COLORS["primary"],
        linewidth=3,
        marker="o",
        markersize=8,
        label="Average Political Stability",
    )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)

    ax.set_xlabel("Year", fontsize=14, fontweight="bold")
    ax.set_ylabel("Political Stability Index (Average)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Global Political Stability Evolution\nAverage Across All Countries Over Time",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=12, framealpha=0.9)

    min_year = int(yearly_avg["Year"].min())
    max_year = int(yearly_avg["Year"].max())
    avg_value = yearly_avg["average"].mean()
    min_value = yearly_avg["average"].min()
    max_value = yearly_avg["average"].max()

    y_range = max_value - min_value
    y_margin = y_range * 0.20
    ax.set_ylim(min_value - y_margin, max_value + y_margin)

    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(nbins=12))

    min_year_val = int(yearly_avg[yearly_avg["average"] == min_value]["Year"].values[0])
    max_year_val = int(yearly_avg[yearly_avg["average"] == max_value]["Year"].values[0])

    first_value = yearly_avg["average"].iloc[0]
    last_value = yearly_avg["average"].iloc[-1]
    total_change = last_value - first_value

    stats_text = f"""Period: {min_year}-{max_year}
• Overall Average: {avg_value:.3f}
• Lowest: {min_value:.3f} (in {min_year_val})
• Highest: {max_value:.3f} (in {max_year_val})
• Total Change: {total_change:+.3f}
• Countries: {int(yearly_avg['count'].mean())} avg per year"""

    ax.text(
        0.02,
        0.75,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        fontsize=11,
        family="monospace",
    )

    plt.tight_layout()

    output_file = _timestamped_path(output_dir, "political_stability_evolution")
    return _save_fig(fig, output_file)


def generate_regional_stability_evolution(data: pd.DataFrame, output_dir: Path):
    """Generate plot showing evolution of political stability by region over time."""
    df = data.copy()

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    if "Year" not in df.columns or "political_stability" not in df.columns:
        raise ValueError("Data must contain 'Year' and 'political_stability' columns")

    df = add_region_column(df)

    regional_yearly = (
        df.groupby(["Region", "Year"])["political_stability"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    region_colors = {
        "Europe": "#2E86AB",
        "Asia": "#A23B72",
        "Americas": "#F18F01",
        "Africa": "#C73E1D",
        "Middle East": "#6A994E",
        "Other": "#95A3A4",
    }

    region_markers = {
        "Europe": "o",
        "Asia": "s",
        "Americas": "^",
        "Africa": "D",
        "Middle East": "v",
        "Other": "P",
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    region_list = ["Europe", "Asia", "Americas", "Africa", "Middle East", "Other"]

    for region in region_list:
        region_data = regional_yearly[regional_yearly["Region"] == region]
        if len(region_data) > 0:
            ax1.plot(
                region_data["Year"],
                region_data["mean"],
                color=region_colors[region],
                linewidth=2.5,
                marker=region_markers[region],
                markersize=6,
                label=f"{region} (n={region_data['count'].iloc[0]:.0f} countries)",
                alpha=0.8,
            )

    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax1.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Average Political Stability Index", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Political Stability Evolution by Region\nAverage Trends Across Geographic Areas",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="best", fontsize=10, framealpha=0.9, ncol=2)

    for region in region_list:
        region_data = regional_yearly[regional_yearly["Region"] == region]
        if len(region_data) > 0:
            ax2.plot(
                region_data["Year"],
                region_data["std"],
                color=region_colors[region],
                linewidth=2,
                marker=region_markers[region],
                markersize=5,
                label=region,
                alpha=0.7,
            )

    ax2.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Standard Deviation", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Political Stability Variability by Region\nWithin-Region Standard Deviation Over Time",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="best", fontsize=10, framealpha=0.9, ncol=2)

    min_year = df["Year"].min()
    max_year = df["Year"].max()

    stats_lines = [f"Regional Changes ({int(min_year)}-{int(max_year)}):"]
    for region in region_list:
        region_data = regional_yearly[regional_yearly["Region"] == region].sort_values(
            "Year"
        )
        if len(region_data) >= 2:
            first_val = region_data["mean"].iloc[0]
            last_val = region_data["mean"].iloc[-1]
            change = last_val - first_val
            stats_lines.append(f"• {region}: {change:+.3f}")

    stats_text = "\n".join(stats_lines)

    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        fontsize=9,
        family="monospace",
    )

    plt.tight_layout()

    output_file = _timestamped_path(output_dir, "regional_stability_evolution")
    return _save_fig(fig, output_file)


# ============================================================================
# CLI DISPLAY UTILITIES
# ============================================================================


class DisplayColors:
    """ANSI color codes for terminal output in display functions."""

    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def display_feature_importance_table(models_data: List[Dict]) -> None:
    """Display feature importance for tree-based models in formatted table."""
    if not models_data:
        print(
            f"{DisplayColors.BRIGHT_RED}[X] No models found. Please train models first{DisplayColors.RESET}"
        )
        return

    print()
    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_CYAN}{'=' * 100}{DisplayColors.RESET}"
    )
    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_CYAN}{'FEATURE IMPORTANCE ANALYSIS'.center(100)}{DisplayColors.RESET}"
    )
    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_CYAN}{'=' * 100}{DisplayColors.RESET}\n"
    )

    tree_models = []
    for model_dict in models_data:
        model_obj = model_dict.get("model_obj")
        if (
            model_obj
            and hasattr(model_obj, "model")
            and hasattr(model_obj.model, "feature_importances_")
        ):
            tree_models.append(model_dict)

    if not tree_models:
        print(
            f"{DisplayColors.YELLOW}[!] No tree-based models found.{DisplayColors.RESET}"
        )
        print(
            f"{DisplayColors.DIM}  Feature importance only available for Random Forest, XGBoost, Gradient Boosting.{DisplayColors.RESET}\n"
        )
        input(f"\n{DisplayColors.DIM}Press Enter to continue...{DisplayColors.RESET}")
        return

    print(
        f"Showing feature importance for {DisplayColors.BOLD}{len(tree_models)}{DisplayColors.RESET} tree-based models:\n"
    )
    for model_dict in tree_models:
        print(f"  {DisplayColors.CYAN}•{DisplayColors.RESET} {model_dict['model']}")
    print()

    if tree_models and "X_test" in tree_models[0]:
        feature_names = list(tree_models[0]["X_test"].columns)
    else:
        print(
            f"{DisplayColors.BRIGHT_RED}[X] Error: Could not extract feature names{DisplayColors.RESET}\n"
        )
        input(f"\n{DisplayColors.DIM}Press Enter to continue...{DisplayColors.RESET}")
        return

    importances_dict = {fname: [] for fname in feature_names}
    model_names = []

    for model_dict in tree_models:
        model_names.append(model_dict["model"])
        importances = model_dict["model_obj"].model.feature_importances_
        for fname, imp in zip(feature_names, importances):
            importances_dict[fname].append(imp)

    avg_importances = {fname: np.mean(imps) for fname, imps in importances_dict.items()}
    sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)

    table_width = 1 + 23 + (21 * len(model_names))
    border_line = f"+{'=' * (table_width - 2)}+"

    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_WHITE}{border_line}{DisplayColors.RESET}"
    )

    header = f" {'Feature':<20} |"
    for model_name in model_names:
        header += f" {model_name:^18} |"
    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_WHITE}|{DisplayColors.RESET}{DisplayColors.BOLD}{header}{DisplayColors.RESET}"
    )
    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_WHITE}{border_line}{DisplayColors.RESET}"
    )

    for feature, avg_imp in sorted_features:
        row = f" {feature:<20} |"
        for imp in importances_dict[feature]:
            if imp >= 0.20:
                color = DisplayColors.GREEN
            elif imp >= 0.10:
                color = DisplayColors.CYAN
            else:
                color = ""
            row += f" {color}{imp:^18.4f}{DisplayColors.RESET if color else ''} |"
        print(
            f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_WHITE}|{DisplayColors.RESET}{row}"
        )

    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_WHITE}{border_line}{DisplayColors.RESET}"
    )

    print()
    print(f"{DisplayColors.DIM}Legend:{DisplayColors.RESET}")
    print(
        f"  {DisplayColors.GREEN}[HIGH]{DisplayColors.RESET} High importance (>= 0.20)"
    )
    print(
        f"  {DisplayColors.CYAN}[MED]{DisplayColors.RESET} Medium importance (>= 0.10)"
    )
    print(f"  {DisplayColors.WHITE}[LOW]{DisplayColors.RESET} Low importance (< 0.10)")
    print()

    input(f"\n{DisplayColors.DIM}Press Enter to continue...{DisplayColors.RESET}")


def display_hyperparameters_table() -> None:
    """Display hyperparameter search spaces for all 7 ML models."""
    print()
    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_CYAN}{'=' * 120}{DisplayColors.RESET}"
    )
    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_CYAN}{'HYPERPARAMETER CONFIGURATION'.center(120)}{DisplayColors.RESET}"
    )
    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_CYAN}{'=' * 120}{DisplayColors.RESET}\n"
    )

    hyperparams = [
        ("Random Forest", "Optuna (TPE)", "n_estimators", "50-500"),
        ("", "", "max_depth", "5-30"),
        ("", "", "min_samples_split", "2-20"),
        ("", "", "min_samples_leaf", "1-10"),
        ("", "", "max_features", "[sqrt, log2, 0.5, 0.7]"),
        ("XGBoost", "Optuna (TPE)", "n_estimators", "50-500"),
        ("", "", "max_depth", "3-10"),
        ("", "", "learning_rate", "0.001-0.3 (log)"),
        ("", "", "subsample", "0.6-1.0"),
        ("", "", "colsample_bytree", "0.6-1.0"),
        ("", "", "reg_alpha", "1e-8-10.0 (log)"),
        ("", "", "reg_lambda", "1e-8-10.0 (log)"),
        ("Gradient Boosting", "Optuna (TPE)", "n_estimators", "50-500"),
        ("", "", "learning_rate", "0.001-0.3 (log)"),
        ("", "", "max_depth", "3-10"),
        ("", "", "subsample", "0.6-1.0"),
        ("", "", "min_samples_split", "2-20"),
        ("", "", "min_samples_leaf", "1-10"),
        ("KNN", "Optuna (TPE)", "n_neighbors", "1-50"),
        ("", "", "weights", "[uniform, distance]"),
        ("", "", "p", "1-3 (metric)"),
        (
            "MLP",
            "Optuna (TPE)",
            "hidden_layer_sizes",
            "[(50,), (100,), (100,50), (100,100), (200,), (200,100)]",
        ),
        ("", "", "activation", "[relu, tanh]"),
        ("", "", "alpha", "1e-5-1e-1 (log)"),
        ("", "", "learning_rate_init", "1e-4-1e-2 (log)"),
        ("", "", "max_iter", "300-1000"),
        ("", "", "early_stopping", "True"),
        ("SVR", "Optuna (TPE)", "C", "0.1-100.0 (log)"),
        ("", "", "epsilon", "0.001-1.0 (log)"),
        ("", "", "kernel", "[linear, rbf, poly]"),
        ("", "", "gamma", "[scale, auto]"),
        ("Elastic Net", "Optuna (TPE)", "alpha", "1e-4-10.0 (log)"),
        ("", "", "l1_ratio", "0.0-1.0"),
    ]

    model_width = 20
    opt_width = 16
    param_width = 24
    value_width = 50
    total_width = model_width + opt_width + param_width + value_width + 9

    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_WHITE}╔{'═' * total_width}╗{DisplayColors.RESET}"
    )
    header = f" {'Model':<{model_width}} │ {'Optimization':<{opt_width}} │ {'Parameter':<{param_width}} │ {'Values/Range':<{value_width}} "
    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_WHITE}║{DisplayColors.RESET}{DisplayColors.BOLD}{header}{DisplayColors.RESET}{DisplayColors.BOLD}{DisplayColors.BRIGHT_WHITE}║{DisplayColors.RESET}"
    )
    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_WHITE}╠{'═' * total_width}╣{DisplayColors.RESET}"
    )

    for model, opt, param, value in hyperparams:
        if model:
            model_color = DisplayColors.BRIGHT_CYAN
        else:
            model_color = ""

        row = f" {model_color}{model:<{model_width}}{DisplayColors.RESET if model_color else ''} │ {opt:<{opt_width}} │ {param:<{param_width}} │ {value:<{value_width}} "
        print(
            f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_WHITE}║{DisplayColors.RESET}{row}{DisplayColors.BOLD}{DisplayColors.BRIGHT_WHITE}║{DisplayColors.RESET}"
        )

    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_WHITE}╚{'═' * total_width}╝{DisplayColors.RESET}"
    )

    print()
    print(f"{DisplayColors.DIM}Notes:{DisplayColors.RESET}")
    print(
        f"  • Optuna (TPE): Bayesian optimization with Tree-structured Parzen Estimator"
    )
    print(f"  • (log): Log-scale sampling for better exploration of wide ranges")
    print(
        f"  • All models use 5-fold cross-validation with MedianPruner for early stopping"
    )
    print()

    input(f"\n{DisplayColors.DIM}Press Enter to continue...{DisplayColors.RESET}")


def copy_figures_to_desktop() -> None:
    """Copy all PNG files from results/figures/ to ~/Desktop/Political_Stability_Figures/."""
    import shutil

    print()
    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_CYAN}{'=' * 80}{DisplayColors.RESET}"
    )
    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_CYAN}{'COPY FIGURES TO DESKTOP'.center(80)}{DisplayColors.RESET}"
    )
    print(
        f"{DisplayColors.BOLD}{DisplayColors.BRIGHT_CYAN}{'=' * 80}{DisplayColors.RESET}\n"
    )

    project_root = Path(__file__).parent.parent
    figures_dir = project_root / "results" / "figures"

    desktop = Path.home() / "Desktop"
    if not desktop.exists():
        desktop = Path.home() / "Bureau"
        if not desktop.exists():
            print(
                f"{DisplayColors.BRIGHT_RED}[X] Desktop folder not found!{DisplayColors.RESET}"
            )
            input(
                f"\n{DisplayColors.DIM}Press Enter to continue...{DisplayColors.RESET}"
            )
            return

    if not figures_dir.exists() or not any(figures_dir.iterdir()):
        print(
            f"{DisplayColors.BRIGHT_RED}[X] No figures found to copy!{DisplayColors.RESET}"
        )
        print(f"  Figures directory: {figures_dir}")
        print(
            f"\n{DisplayColors.YELLOW}Tip:{DisplayColors.RESET} Generate visualizations first (option 5)"
        )
        input(f"\n{DisplayColors.DIM}Press Enter to continue...{DisplayColors.RESET}")
        return

    dest_folder = desktop / "Political_Stability_Figures"
    dest_folder.mkdir(exist_ok=True)

    copied_count = 0
    try:
        for fig_file in figures_dir.glob("*.png"):
            dest_file = dest_folder / fig_file.name
            shutil.copy2(fig_file, dest_file)
            copied_count += 1
            print(
                f"  {DisplayColors.BRIGHT_GREEN}[OK]{DisplayColors.RESET} Copied: {fig_file.name}"
            )

        print()
        print(f"{DisplayColors.BRIGHT_GREEN}{'=' * 80}{DisplayColors.RESET}")
        print(
            f"{DisplayColors.BRIGHT_GREEN}SUCCESS: Copied {copied_count} figures to Desktop{DisplayColors.RESET}"
        )
        print(f"{DisplayColors.BRIGHT_GREEN}{'=' * 80}{DisplayColors.RESET}\n")
        print(f"{DisplayColors.CYAN}Location:{DisplayColors.RESET} {dest_folder}")

    except Exception as e:
        print(
            f"{DisplayColors.BRIGHT_RED}[X] Failed to copy figures: {str(e)}{DisplayColors.RESET}"
        )

    input(f"\n{DisplayColors.DIM}Press Enter to continue...{DisplayColors.RESET}")
