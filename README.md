# Political Stability Observatory

## Research Questions
1. Can political stability be systematically predicted using machine learning models and econometric approaches?
2. What predictive accuracy (R², RMSE) is achievable for this complex phenomenon?
3. Do ML models (Random Forest, XGBoost, Gradient Boosting, Elastic Net, SVR, KNN, MLP) outperform Dynamic Panel Fixed Effects?

## Overview
This project predicts political stability across 191 countries using 28 years of data (1996-2023) from World Bank, UNDP, and Worldwide Governance Indicators. We compare 7 machine learning models (Random Forest, XGBoost, Gradient Boosting, Elastic Net, SVR, KNN, MLP) against a Dynamic Panel Fixed Effects econometric benchmark. All ML models use Optuna Bayesian hyperparameter optimization with temporal train-test splitting (1996-2017 train, 2018-2023 test). The system includes an interactive terminal for model training/testing and a Streamlit dashboard for visualization and country-level analysis.

## Prerequisites

**Git must be installed to clone this repository.**

- **Mac**: Install via Homebrew: `brew install git` or [Xcode Command Line Tools](https://developer.apple.com/xcode/)
- **Windows**: Download from [git-scm.com](https://git-scm.com/download/win)
- **Linux**: `sudo apt-get install git` (Ubuntu/Debian) or `sudo yum install git` (RedHat/CentOS)

Verify installation: `git --version`

## Setup

### Option 1: With Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/kissackjonathan/Datascience-and-Advanced-Programming-project-HEC-.git
cd Datascience-and-Advanced-Programming-project-HEC-

# Create and activate environment
conda env create -f environment.yml
conda activate political-stability-prediction

# Launch interactive terminal
python main/main.py
```

### Option 2: With pip + venv

#### On Mac/Linux:
```bash
# Clone repository
git clone https://github.com/kissackjonathan/Datascience-and-Advanced-Programming-project-HEC-.git
cd Datascience-and-Advanced-Programming-project-HEC-

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch interactive terminal
python main/main.py
```

#### On Windows (PowerShell):
```powershell
# Clone repository
git clone https://github.com/kissackjonathan/Datascience-and-Advanced-Programming-project-HEC-.git
cd Datascience-and-Advanced-Programming-project-HEC-

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch interactive terminal
python main/main.py
```

### Important Note for Mac Users

If you have `.numbers` data files (Apple Numbers format), convert them to `.csv` for cross-platform compatibility:

```bash
# Install numbers-parser
pip install numbers-parser

# Run conversion utility
python src/data_loader.py

# Commit the CSV files
git add data/raw/*.csv
git commit -m "Add CSV data files"
git push
```

## Usage

### Interactive Terminal (Main Interface)

The interactive terminal is the primary way to use this project. Launch it with:

```bash
python main/main.py
```

**Menu Options:**
```
================================================================================
                    POLITICAL STABILITY PREDICTION SYSTEM
================================================================================

[0] Check Environment          - Verify setup and data files
[1] Run Data Preparation       - Load, clean, and merge datasets
[2] Train Model                - Train ML models with hyperparameter tuning
[3] Test Model                 - Evaluate trained models on test set
[4] Evaluate Saved Models      - Compare all saved model results
[5] Run Visualization          - Generate plots and analysis charts
[6] Show Dashboard Link        - Get Streamlit dashboard URL
[7] Test Coverage              - Run pytest test suite
[Q] Quit                       - Exit the program
```

**Typical Workflow:**
1. **[0]** Check Environment - Verify all packages and data files
2. **[1]** Run Data Preparation - Load and process data
3. **[2]** Train Model - Train all 7 ML models + Panel model
4. **[3]** Test Model - Evaluate on test set
5. **[5]** Run Visualization - Generate result plots
6. **[6]** Show Dashboard - Launch interactive dashboard

### Interactive Dashboard

```bash
# Launch Streamlit dashboard
streamlit run main/dashboard.py
```

Dashboard opens at `http://localhost:8501` with 3 pages:
- **Home**: World map, KPIs, model leaderboard
- **Explorer**: Country-level analysis and trends
- **Project**: Complete ML workflow (Load Data → Train → Test → Visualize)

**Note:** If the dashboard doesn't open automatically in your browser, manually copy and paste this URL: `http://localhost:8501`

## Project Structure

```
Datascience-and-Advanced-Programming-2025-2026-project-JK/
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI/CD pipeline
├── pre-commit/              # Pre-commit hooks configuration
│   ├── .pre-commit-config.yaml
│   ├── hooks/
│   └── scripts/
├── main/
│   ├── main.py              # Interactive CLI entry point
│   ├── terminal.py          # Terminal functions
│   └── dashboard.py         # Streamlit dashboard
├── src/
│   ├── data_loader.py       # Data loading/preprocessing
│   ├── models.py            # Model training (7 ML + Panel)
│   └── evaluation.py        # Evaluation metrics & visualizations
├── data/
│   ├── raw/                 # World Bank, UNDP data
│   └── processed/           # Train/test splits
├── results/
│   ├── figures/             # Generated plots (.png)
│   ├── benchmark_results.csv
│   └── test_results.csv
├── tests/                   # Unit tests
├── README.md                # Project documentation
├── REPORT.md                # Full academic report
├── PROPOSAL.md              # Project proposal
├── environment.yml          # Conda dependencies
└── requirements.txt         # Pip dependencies
```

## Results

### Model Performance (Test Set 2018-2023)

| Rank | Model                      | Test R² | Test RMSE | Test MAE |
|:----:|:---------------------------|--------:|----------:|---------:|
| 1st  | Gradient Boosting          |  0.8080 |    0.4242 |   0.3207 |
| 2nd  | XGBoost                    |  0.8013 |    0.4316 |   0.3242 |
| 3rd  | Random Forest              |  0.7952 |    0.4381 |   0.3409 |
| 4th  | KNN                        |  0.7141 |    0.5176 |   0.3680 |
| 5th  | MLP                        |  0.6845 |    0.5437 |   0.4119 |
| 6th  | SVR                        |  0.6778 |    0.5495 |   0.4232 |
| 7th  | Elastic Net                |  0.6322 |    0.5871 |   0.4648 |

**Winner**: Gradient Boosting (best test performance with R²=0.8080)

### Key Findings
- **Gradient Boosting achieves best test performance (R²=0.8080, RMSE=0.4242)**
- Tree-based ensemble models (GBM, XGBoost, RF) dominate top 3 positions
- All top-3 models achieve R² > 0.79, explaining over 79% of variance
- No significant overfitting detected (train-test gap < 0.1)

### Dynamic Panel Fixed Effects Model (Econometric Benchmark)

The Panel Analyzer implements a two-way fixed effects model with dynamic structure:

**Model Specification:**
```
Y_it = α + βY_{i,t-1} + γX_it + μ_i + ε_it
```

Where:
- Y_it: Political stability for country i at time t
- Y_{i,t-1}: Lagged dependent variable (captures persistence)
- X_it: Macroeconomic and governance predictors
- μ_i: Country-specific fixed effects
- ε_it: Idiosyncratic error term

**Diagnostic Tests:**
- Hausman Test: Tests for correlation between effects and regressors
- Breusch-Pagan Test: Detects heteroscedasticity
- Durbin-Watson Test: Checks for serial correlation

This classical econometric approach provides interpretable coefficients and serves as a benchmark for comparing ML model performance.

## Requirements

- Python 3.12+
- scikit-learn, xgboost, optuna (ML frameworks)
- pandas, numpy (data processing)
- linearmodels, statsmodels (econometrics)
- matplotlib, seaborn, plotly (visualization)
- streamlit (dashboard)
- pytest, pytest-cov (testing)

## Data Sources

- **World Bank Open Data**: GDP, unemployment, inflation (1996-2023)
- **UNDP**: Human Development Index (1996-2023)
- **Worldwide Governance Indicators**: Political stability, rule of law

## License

MIT License - Master's Project 2025-2026 | Data Science & Advanced Programming
