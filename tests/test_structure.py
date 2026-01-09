"""
Project structure tests.
Tests for verifying project directory structure and module imports.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_project_structure():
    # Test: Verifies that all project directories exist
    # Purpose: Ensures structure integrity before execution
    """Verify project directories exist."""
    project_root = Path(__file__).parent.parent
    assert (project_root / "src").exists()
    assert (project_root / "data").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "main").exists()


def test_module_imports():
    # Test: Verifies that all main modules import without error
    # Purpose: Detects dependency issues or circular imports
    """Test that all main modules can be imported."""
    from src import data_loader, evaluation, models

    assert data_loader is not None
    assert models is not None
    assert evaluation is not None


def check_environment_test():
    """
    Check Python version, packages, directories, and data files.
    Returns dict with results for each check category.
    Used by main.py to verify environment setup.
    """
    project_root = Path(__file__).parent.parent

    # Helper function to get project paths
    def get_project_paths():
        return {
            "data_raw": project_root / "data" / "raw",
            "data_processed": project_root / "data" / "processed",
            "models": project_root / "src" / "models",
            "results": project_root / "results",
            "figures": project_root / "results" / "figures",
        }

    results = {
        "python": {"ok": True, "message": ""},
        "packages": {"ok": True, "issues": []},
        "directories": {"ok": True, "issues": []},
        "data_files": {"ok": True, "issues": []},
        "all_ok": True,
    }

    # 1. Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        results["python"][
            "message"
        ] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
    else:
        results["python"]["ok"] = False
        results["python"][
            "message"
        ] = f"{python_version.major}.{python_version.minor} (requires >= 3.8)"
        results["all_ok"] = False

    # 2. Check required packages
    import importlib

    required_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "optuna",
        "linearmodels",
        "statsmodels",
        "scipy",
        "joblib",
        "matplotlib",
        "seaborn",
        "plotly",
        "streamlit",
    ]

    for package in required_packages:
        try:
            # Try importing the package directly (more reliable than metadata)
            pkg_name = package.replace("-", "_")  # scikit-learn -> sklearn
            if package == "scikit-learn":
                pkg_name = "sklearn"

            mod = importlib.import_module(pkg_name)
            pkg_version = getattr(mod, "__version__", "unknown")
            results["packages"]["issues"].append(
                {"name": package, "version": pkg_version, "installed": True}
            )
        except (ImportError, ModuleNotFoundError):
            results["packages"]["issues"].append(
                {"name": package, "version": None, "installed": False}
            )
            results["packages"]["ok"] = False
            results["all_ok"] = False

    # 3. Check directories
    paths = get_project_paths()
    for name, path in paths.items():
        if path.exists():
            results["directories"]["issues"].append(
                {"name": name, "path": str(path), "exists": True}
            )
        else:
            results["directories"]["issues"].append(
                {"name": name, "path": str(path), "exists": False}
            )
            # Create directory
            path.mkdir(parents=True, exist_ok=True)

    # 4. Check raw data files
    data_raw = paths["data_raw"]
    expected_files = [
        ("political stability", [".csv", ".numbers", ".xlsx"]),
        ("GDP per capita", [".csv", ".numbers", ".xlsx"]),
        ("UNEMPLOYMENT_TOTAL", [".csv", ".numbers", ".xlsx"]),
        ("inflation consumer", [".csv", ".numbers", ".xlsx"]),
        ("GDP_GROWTH_%", [".csv", ".numbers", ".xlsx"]),
        ("effectiveness", [".csv", ".numbers", ".xlsx"]),
        ("rule of law", [".csv", ".numbers", ".xlsx"]),
        ("trade", [".csv", ".numbers", ".xlsx"]),
        ("hdi_data", [".xlsx"]),
    ]

    for base_name, extensions in expected_files:
        found = False
        found_ext = None
        found_size = 0

        for ext in extensions:
            filepath = data_raw / f"{base_name}{ext}"
            if filepath.exists():
                found = True
                found_ext = ext
                found_size = filepath.stat().st_size / (1024 * 1024)  # MB
                break

        results["data_files"]["issues"].append(
            {
                "name": base_name,
                "found": found,
                "extension": found_ext,
                "size_mb": found_size,
                "checked_extensions": extensions,
            }
        )

        if not found:
            results["data_files"]["ok"] = False
            results["all_ok"] = False

    return results
