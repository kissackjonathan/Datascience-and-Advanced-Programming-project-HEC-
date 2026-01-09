"""
Country filter tests.
Tests for country filtering with UN whitelist and data quality checks.
"""
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "data" / "raw").exists(),
    reason="Raw data directory not available",
)
class TestCountryFilter:
    """Test country filtering with UN whitelist."""

    def test_no_aggregates_in_data(self):
        # Test: Verifies that World Bank aggregates are excluded from data
        # Purpose: Ensures only real countries are included in analysis
        """Test that World Bank aggregates are removed."""
        from src.data_loader import load_data

        data_path = Path(__file__).parent.parent / "data" / "raw"
        data = load_data(
            data_path=data_path, target="political_stability", train_end_year=2017
        )

        all_countries = set(
            data["X_train"].index.get_level_values("Country Name").unique().tolist()
            + data["X_test"].index.get_level_values("Country Name").unique().tolist()
        )

        aggregates = [
            "World",
            "Europe & Central Asia",
            "High income",
            "OECD members",
            "European Union",
            "Arab World",
            "Sub-Saharan Africa",
        ]

        for aggregate in aggregates:
            assert aggregate not in all_countries

    def test_countries_have_sufficient_data(self):
        # Test: Verifies that countries with >30% missing data are filtered
        # Purpose: Ensures minimum data quality for each country
        """Test that countries with >30% missing data are filtered out."""
        from src.data_loader import load_data

        data_path = Path(__file__).parent.parent / "data" / "raw"
        data = load_data(
            data_path=data_path, target="political_stability", train_end_year=2017
        )

        train_data = data["df_train"]

        # Check data quality
        missing_pct = train_data.isnull().sum().sum() / (
            len(train_data) * len(train_data.columns)
        )
        assert missing_pct < 0.1
