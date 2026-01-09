"""
Data loader tests.
Tests for data loading functions and train/test splitting.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_load_data_file_function_exists():
    # Test: Verifies that load_data_file function exists
    # Purpose: Basic test to ensure function is importable
    """Test load_data_file function exists."""
    from src.data_loader import load_data_file

    assert callable(load_data_file)


def test_load_data_file_with_synthetic_data(temp_dir):
    # Test: Verifies loading of World Bank format data (wide → long)
    # Purpose: Ensures correct conversion of World Bank CSV files
    """Test load_data_file with synthetic World Bank format data."""
    from src.data_loader import load_data_file

    # Create a sample World Bank format CSV file
    csv_path = temp_dir / "test_wb_data.csv"

    with open(csv_path, "w") as f:
        f.write("Indicator Name,GDP\n")
        f.write("Indicator Code,NY.GDP.MKTP.CD\n")
        f.write("Source,World Bank\n")
        f.write("Last Updated,2024\n")
        f.write("Country Name,Country Code,2018,2019,2020\n")
        f.write("France,FRA,2600,2700,2800\n")
        f.write("Germany,DEU,3500,3600,3700\n")

    df = load_data_file(csv_path, "GDP")

    assert "Country Name" in df.columns
    assert "Year" in df.columns
    assert "GDP" in df.columns
    assert len(df) > 0


def test_get_train_test_split(panel_data):
    # Test: Verifies temporal train/test split on panel data
    # Purpose: Ensures split respects chronological order
    """Test temporal train/test split."""
    from src.data_loader import get_train_test_split

    train_data, test_data = get_train_test_split(panel_data, train_end_year=2015)

    assert len(train_data) > 0
    assert len(test_data) > 0
    assert train_data["Year"].max() <= 2015
    assert test_data["Year"].min() > 2015


def test_load_data_with_mock_files(temp_dir):
    # Test: Verifies loading with mock World Bank files
    # Purpose: Tests pipeline without depending on real data
    """Test load_data with mock World Bank files (no real data needed)."""
    from src.data_loader import load_data_file

    # Create mock raw data directory
    raw_dir = temp_dir / "raw"
    raw_dir.mkdir()

    # Create mock World Bank format files
    countries = ["United States", "France", "Germany"]
    years = ["2015", "2016", "2017", "2018", "2019", "2020"]

    # Create GDP file
    with open(raw_dir / "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_*.csv", "w") as f:
        f.write("Indicator Name,GDP per capita\n")
        f.write("Indicator Code,NY.GDP.PCAP.CD\n")
        f.write("Source,World Bank\n")
        f.write("Last Updated,2024\n")
        f.write("Country Name,Country Code," + ",".join(years) + "\n")
        for i, country in enumerate(countries):
            values = [str(40000 + i * 5000 + j * 100) for j in range(len(years))]
            f.write(f"{country},{country[:3].upper()}," + ",".join(values) + "\n")

    # Create unemployment file
    with open(raw_dir / "API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_*.csv", "w") as f:
        f.write("Indicator Name,Unemployment\n")
        f.write("Indicator Code,SL.UEM.TOTL.ZS\n")
        f.write("Source,World Bank\n")
        f.write("Last Updated,2024\n")
        f.write("Country Name,Country Code," + ",".join(years) + "\n")
        for country in countries:
            values = [str(5 + np.random.rand()) for _ in years]
            f.write(f"{country},{country[:3].upper()}," + ",".join(values) + "\n")

    # Create political stability file
    with open(raw_dir / "API_PV.EST_DS2_en_csv_v2_*.csv", "w") as f:
        f.write("Indicator Name,Political Stability\n")
        f.write("Indicator Code,PV.EST\n")
        f.write("Source,World Bank\n")
        f.write("Last Updated,2024\n")
        f.write("Country Name,Country Code," + ",".join(years) + "\n")
        for country in countries:
            values = [str(0.5 + np.random.randn() * 0.2) for _ in years]
            f.write(f"{country},{country[:3].upper()}," + ",".join(values) + "\n")

    # Note: Full load_data requires many more files, so this is a simplified test
    # that verifies the basic file loading mechanism works
    df = load_data_file(
        raw_dir / "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_*.csv", "gdp_per_capita"
    )

    assert "Country Name" in df.columns
    assert "Year" in df.columns
    assert "gdp_per_capita" in df.columns
    assert len(df) > 0
    assert df["Year"].min() >= 2015
    assert df["Year"].max() <= 2020


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "data" / "raw").exists(),
    reason="Raw data directory not available",
)
def test_load_data_full_pipeline():
    # Test: Verifies complete loading pipeline with real data
    # Purpose: End-to-end integration test of data loading
    """Test full data loading pipeline with real data."""
    from src.data_loader import load_data

    data_path = Path(__file__).parent.parent / "data" / "raw"

    result = load_data(
        data_path=data_path, target="political_stability", train_end_year=2017
    )

    assert isinstance(result, dict)
    assert "X_train" in result
    assert "X_test" in result
    assert "y_train" in result
    assert "y_test" in result
    assert "df_train" in result
    assert "df_test" in result

    # Verify MultiIndex
    assert isinstance(result["df_train"].index, pd.MultiIndex)
    assert isinstance(result["df_test"].index, pd.MultiIndex)

    # Verify no missing values in features
    assert result["X_train"].isnull().sum().sum() == 0
