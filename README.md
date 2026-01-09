# Political Stability Observatory

## Research Questions
1. Can political stability be systematically predicted using machine learning models and econometric approaches?
2. What predictive accuracy (R², RMSE) is achievable for this complex phenomenon?
3. Do ML models (Random Forest, XGBoost, Gradient Boosting, Elastic Net, SVR, KNN, MLP) outperform Dynamic Panel Fixed Effects?

## Overview
This project predicts political stability across 191 countries using 28 years of data (1996-2023) from World Bank, UNDP, and Worldwide Governance Indicators. We compare 7 machine learning models (Random Forest, XGBoost, Gradient Boosting, Elastic Net, SVR, KNN, MLP) against a Dynamic Panel Fixed Effects econometric benchmark. All ML models use Optuna Bayesian hyperparameter optimization with temporal train-test splitting (1996-2017 train, 2018-2023 test). The system includes an interactive terminal for model training/testing and a Streamlit dashboard for visualization and country-level analysis.

## Setup

### Option 1: With Conda (Recommended)

**Prerequisites:** Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

#### On Mac/Linux:
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

#### On Windows (PowerShell or Command Prompt):
```powershell
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

# Run conversion script
python scripts/convert_data.py

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
| 1st  | Random Forest              |  0.7528 |    0.4813 |   0.3775 |
| 2nd  | XGBoost                    |  0.7344 |    0.4989 |   0.3925 |
| 3rd  | Gradient Boosting          |  0.7245 |    0.5081 |   0.3990 |
| 4th  | KNN                        |  0.7127 |    0.5188 |   0.4030 |
| 5th  | MLP                        |  0.6969 |    0.5329 |   0.4225 |
| 6th  | Elastic Net                |  0.6297 |    0.5891 |   0.4701 |
| 7th  | SVR                        |  0.6237 |    0.5938 |   0.4650 |

**Winner**: Random Forest (best test performance with R²=0.7528)

### Key Findings
- **Random Forest achieves best test performance (R²=0.7528, RMSE=0.4813)**
- Tree-based models (RF, XGBoost, GBM) are best ML approaches
- No significant overfitting detected (train-test gap < 0.1)

### Most Important Features (Random Forest)
1. Rule of Law (37.34%)
2. Government Effectiveness (21.25%)
3. GDP per Capita (12.60%)
4. HDI (8.73%)
5. Trade (8.65%)
6. Unemployment (4.95%)
7. Inflation (4.00%)
8. GDP Growth (2.47%)

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
