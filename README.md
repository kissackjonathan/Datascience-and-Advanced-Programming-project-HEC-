# Political Stability Observatory

## Research Questions
1. Can political stability be systematically predicted using machine learning models and econometric approaches?
2. What predictive accuracy (R², RMSE) is achievable for this complex phenomenon?
3. Do ML models (Random Forest, XGBoost, Gradient Boosting, Elastic Net, SVR, KNN, MLP) outperform Dynamic Panel Fixed Effects?

## Overview
This project predicts political stability across 191 countries using 28 years of data (1996-2023) from World Bank, UNDP, and Worldwide Governance Indicators. We compare 7 machine learning models (Random Forest, XGBoost, Gradient Boosting, Elastic Net, SVR, KNN, MLP) against a Dynamic Panel Fixed Effects econometric benchmark. All ML models use Optuna Bayesian hyperparameter optimization with temporal train-test splitting (1996-2017 train, 2018-2023 test). The system includes an interactive terminal for model training/testing and a Streamlit dashboard for visualization and country-level analysis.

## Setup

```bash
# Clone repository
git clone https://github.com/yourusername/political-stability-observatory.git
cd Datascience-and-Advanced-Programming-2025-2026-project-JK

# Create environment
conda env create -f environment.yml
conda activate political-stability-prediction
```

## Usage

### Getting Started from VSCode

If you're opening the project for the first time in VSCode:

1. **Open the project folder**
   - Launch VSCode
   - `File > Open Folder...` (or `Ctrl+K Ctrl+O` on Windows/Linux, `Cmd+K Cmd+O` on Mac)
   - Navigate to and select `Datascience-and-Advanced-Programming-2025-2026-project-JK`

2. **Open a terminal**
   - `Terminal > New Terminal` (or press `` Ctrl+` `` / `` Cmd+` ``)
   - The terminal should automatically open in the project root directory

3. **Verify you're in the correct directory**
   ```bash
   pwd  # Should display: /path/to/Datascience-and-Advanced-Programming-2025-2026-project-JK
   ls   # Should show: main/ src/ data/ tests/ README.md environment.yml
   ```

4. **Activate the conda environment** (if not already active)
   ```bash
   conda activate political-stability-prediction
   ```

5. **Launch the interactive terminal**
   ```bash
   python main/main.py
   ```

### Interactive Terminal

Once launched, the terminal menu offers:

Menu options:
- **[1]** Run Data Preparation
- **[2]** Train Model (7 ML + Dynamic Panel)
- **[3]** Test Model
- **[4]** Evaluate Saved Models
- **[5]** Run Visualization
- **[6]** Show Dashboard Link
- **[7]** Test Coverage
- **[Q]** Quit

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
