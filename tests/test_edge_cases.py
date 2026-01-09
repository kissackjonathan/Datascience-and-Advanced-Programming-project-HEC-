"""
Edge cases and error handling tests.
Tests for edge cases, error conditions, and boundary behaviors.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_perfect_predictions(temp_dir):
    # Test: Verifies behavior with perfect predictions (R^2=1)
    # Purpose: Tests edge cases of visualization functions
    """Test evaluation functions with perfect predictions."""
    from src.evaluation import plot_predictions, plot_residuals

    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])  # Perfect predictions

    pred_path = temp_dir / "perfect_pred.png"
    plot_predictions(y_true, y_pred, model_name="Perfect", output_path=pred_path)
    assert pred_path.exists()

    resid_path = temp_dir / "perfect_resid.png"
    plot_residuals(y_true, y_pred, model_name="Perfect", output_path=resid_path)
    assert resid_path.exists()


def test_model_load_nonexistent_file(logger, temp_dir):
    # Test: Verifies that loading non-existent file raises an error
    # Purpose: Ensures proper handling of loading errors
    """Test loading model from non-existent file raises error."""
    from src.models import RandomForestPredictor

    model = RandomForestPredictor(logger=logger)
    nonexistent_path = temp_dir / "nonexistent.pkl"

    with pytest.raises((FileNotFoundError, ValueError)):
        model.load_model(nonexistent_path)
