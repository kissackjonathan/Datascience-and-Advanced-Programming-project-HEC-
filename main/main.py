"""
Political Stability Prediction - Main Entry Point
==================================================

This is the main entry point for the entire project workflow.
It provides an interactive menu to execute each step of the ML pipeline:

0. Check Environment       - Verify dependencies and data
1. Run Data Preparation    - Load, clean, and prepare datasets
2. Train Model             - Train ML/econometric models
3. Test Model              - Test trained models
4. Evaluate Saved Models   - Compare all saved models
5. Run Visualization       - Generate plots and charts
6. Show Dashboard Link     - Display Streamlit dashboard URL
7. Test Coverage           - Run test coverage analysis
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# LOGGING CONFIGURATION (MUST BE DONE FIRST, BEFORE ALL IMPORTS)
# ============================================================================
# Configure logging before importing any project modules to ensure all modules
# use this centralized configuration and prevent duplicate logging setups.
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# ============================================================================
# REPRODUCIBILITY: Fix all random seeds for consistent results
# ============================================================================
# Ensures that running the code multiple times produces identical results,
# which is essential for scientific reproducibility and model comparison.
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
# ============================================================================


# ============================================================================
# SESSION STATE - In-memory storage for current workflow
# ============================================================================

TRAINING_COMPLETED_THIS_SESSION = False

SESSION_MODELS_DATA = None
SESSION_TRAIN_DATA = None
SESSION_TEST_DATA = None
SESSION_FULL_DATA = None
SESSION_PANEL_METRICS = None

WORKFLOW_COMPLETION = {
    "check_environment": False,
    "data_preparation": False,
    "train_model": False,
    "test_model": False,
    "evaluate_models": False,
    "visualization": False,
    "dashboard": False,
    "test_coverage": False,
}


# ============================================================================
# TERMINAL FORMATTING & COLORS
# ============================================================================


class Colors:
    """ANSI color codes for terminal output (works on macOS/Linux/Windows 10+)."""

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def clear_screen() -> None:
    """Clear terminal screen (cls on Windows, clear on Unix/macOS)."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title: str, subtitle: str = "") -> None:
    """Print formatted header with optional subtitle."""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{title.center(80)}{Colors.RESET}")
    if subtitle:
        print(f"{Colors.CYAN}{subtitle.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}\n")


def print_section(title: str) -> None:
    """Print formatted section header for actions (0-7)."""
    line = "-" * 80
    print(f"\n{Colors.BRIGHT_CYAN}{line}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{title}{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}{line}{Colors.RESET}\n")


def print_success(message: str) -> None:
    """Print success message in green."""
    print(f"{Colors.BRIGHT_GREEN} {message}{Colors.RESET}")


def print_error(message: str) -> None:
    """Print error message in red."""
    print(f"{Colors.BRIGHT_RED} {message}{Colors.RESET}")


def print_warning(message: str) -> None:
    """Print warning message in yellow."""
    print(f"{Colors.BRIGHT_YELLOW} {message}{Colors.RESET}")


def print_info(message: str) -> None:
    """Print info message in cyan."""
    print(f"{Colors.CYAN}i {message}{Colors.RESET}")


def pause(message: str = "Press Enter to continue...") -> None:
    """Pause execution and wait for user input."""
    input(f"\n{Colors.DIM}{message}{Colors.RESET}")


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================


def get_project_paths() -> Dict[str, Path]:
    """Get all project directory paths."""
    root = PROJECT_ROOT
    return {
        "root": root,
        "data_raw": root / "data" / "raw",
        "data_processed": root / "data" / "processed",
        "results": root / "results",
        "figures": root / "results" / "figures",
        "src": root / "src",
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    from terminal import main
    main()
