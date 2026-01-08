"""
Data loading and preprocessing module.

Loads raw macroeconomic data (World Bank, UNDP) and prepares panel datasets
for machine learning and econometric analysis (wide → long, filtering,
imputation, temporal train/test split).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# Optional: numbers-parser for Apple Numbers files
try:
    from numbers_parser import Document

    NUMBERS_PARSER_AVAILABLE = True
except ImportError:
    NUMBERS_PARSER_AVAILABLE = False


# ============================================================================
# COUNTRY WHITELISTS
# ============================================================================
# 193 UN member states (2024), using World Bank naming
UN_MEMBER_STATES = [
    # A
    "Afghanistan",
    "Albania",
    "Algeria",
    "Andorra",
    "Angola",
    "Antigua and Barbuda",
    "Argentina",
    "Armenia",
    "Australia",
    "Austria",
    "Azerbaijan",
    # B
    "Bahamas, The",
    "Bahrain",
    "Bangladesh",
    "Barbados",
    "Belarus",
    "Belgium",
    "Belize",
    "Benin",
    "Bhutan",
    "Bolivia",
    "Bosnia and Herzegovina",
    "Botswana",
    "Brazil",
    "Brunei Darussalam",
    "Bulgaria",
    "Burkina Faso",
    "Burundi",
    # C
    "Cabo Verde",
    "Cambodia",
    "Cameroon",
    "Canada",
    "Central African Republic",
    "Chad",
    "Chile",
    "China",
    "Colombia",
    "Comoros",
    "Congo, Dem. Rep.",
    "Congo, Rep.",
    "Costa Rica",
    "Cote d'Ivoire",
    "Croatia",
    "Cuba",
    "Cyprus",
    "Czechia",
    # D
    "Denmark",
    "Djibouti",
    "Dominica",
    "Dominican Republic",
    # E
    "Ecuador",
    "Egypt, Arab Rep.",
    "El Salvador",
    "Equatorial Guinea",
    "Eritrea",
    "Estonia",
    "Eswatini",
    "Ethiopia",
    # F
    "Fiji",
    "Finland",
    "France",
    # G
    "Gabon",
    "Gambia, The",
    "Georgia",
    "Germany",
    "Ghana",
    "Greece",
    "Grenada",
    "Guatemala",
    "Guinea",
    "Guinea-Bissau",
    "Guyana",
    # H
    "Haiti",
    "Honduras",
    "Hungary",
    # I
    "Iceland",
    "India",
    "Indonesia",
    "Iran, Islamic Rep.",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    # J
    "Jamaica",
    "Japan",
    "Jordan",
    # K
    "Kazakhstan",
    "Kenya",
    "Kiribati",
    "Korea, Dem. People's Rep.",
    "Korea, Rep.",
    "Kuwait",
    "Kyrgyz Republic",
    # L
    "Lao PDR",
    "Latvia",
    "Lebanon",
    "Lesotho",
    "Liberia",
    "Libya",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    # M
    "Madagascar",
    "Malawi",
    "Malaysia",
    "Maldives",
    "Mali",
    "Malta",
    "Marshall Islands",
    "Mauritania",
    "Mauritius",
    "Mexico",
    "Micronesia, Fed. Sts.",
    "Moldova",
    "Monaco",
    "Mongolia",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Myanmar",
    # N
    "Namibia",
    "Nauru",
    "Nepal",
    "Netherlands",
    "New Zealand",
    "Nicaragua",
    "Niger",
    "Nigeria",
    "North Macedonia",
    "Norway",
    # O
    "Oman",
    # P
    "Pakistan",
    "Palau",
    "Panama",
    "Papua New Guinea",
    "Paraguay",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    # Q
    "Qatar",
    # R
    "Romania",
    "Russian Federation",
    "Rwanda",
    # S
    "Samoa",
    "San Marino",
    "Sao Tome and Principe",
    "Saudi Arabia",
    "Senegal",
    "Serbia",
    "Seychelles",
    "Sierra Leone",
    "Singapore",
    "Slovak Republic",
    "Slovenia",
    "Solomon Islands",
    "Somalia",
    "South Africa",
    "South Sudan",
    "Spain",
    "Sri Lanka",
    "St. Kitts and Nevis",
    "St. Lucia",
    "St. Vincent and the Grenadines",
    "Sudan",
    "Suriname",
    "Sweden",
    "Switzerland",
    "Syrian Arab Republic",
    # T
    "Tajikistan",
    "Tanzania",
    "Thailand",
    "Timor-Leste",
    "Togo",
    "Tonga",
    "Trinidad and Tobago",
    "Tunisia",
    "Turkey",
    "Turkmenistan",
    "Tuvalu",
    # U
    "Uganda",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
    "Uruguay",
    "Uzbekistan",
    # V
    "Vanuatu",
    "Venezuela, RB",
    "Vietnam",
    # Y
    "Yemen, Rep.",
    # Z
    "Zambia",
    "Zimbabwe",
]

# Additional territories with strong/usable economic data
ADDITIONAL_TERRITORIES = [
    "Taiwan, China",
    "Kosovo",
    "West Bank and Gaza",
    "Hong Kong SAR, China",
    "Macao SAR, China",
    "Puerto Rico",
    "Guam",
]

# Complete whitelist
COUNTRIES_TO_KEEP = UN_MEMBER_STATES + ADDITIONAL_TERRITORIES

# Country name normalization (World Bank → UN standard)
COUNTRY_NAME_MAPPING = {
    "Turkiye": "Turkey",
    "Viet Nam": "Vietnam",
    "Somalia, Fed. Rep.": "Somalia",
    "Puerto Rico (US)": "Puerto Rico",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _read_with_optional_skip(reader, path, skip=4, **kwargs):
    """
    Read file with optional metadata skipping (World Bank format fallback).
    """
    try:
        return reader(path, skiprows=skip, **kwargs)
    except Exception as e:
        logger.debug(f"Failed with skiprows={skip}, trying without: {e}")
        return reader(path, **kwargs)


def _detect_year_columns(df):
    """
    Detect year columns (1900–2100), handling int/float labels from mixed file formats.
    """
    year_cols = []
    for c in df.columns:
        if c in ["Country Name", "Country Code", "Year"]:
            continue
        try:
            # Handle both int and float year columns (e.g., 1960 or 1960.0 from Numbers files)
            year_val = int(float(c))
            if 1900 <= year_val <= 2100:
                year_cols.append(c)
        except (ValueError, TypeError):
            continue
    return year_cols


def _standardize_country_col(df):
    """Find and rename Country Name column to standard format."""
    candidates = {"Country Name", "Country", "Country name", "country_name", "country"}
    found = candidates & set(df.columns)
    country_col = next(iter(found), df.columns[0])
    return df.rename(columns={country_col: "Country Name"})


# ============================================================================
# MAIN LOADING FUNCTIONS
# ============================================================================


def load_data_file(file_path: str, indicator_name: str) -> Optional[pd.DataFrame]:
    """
    Load a single indicator file and normalize it to long format.

    Handles World Bank-style files (4 metadata rows, years as columns) and
    converts them to a standardized long panel format: Country Name × Year × indicator.
    Returns None if the file cannot be loaded or parsed.

    Parameters
    ----------
    file_path : str
        Path to the data file
    indicator_name : str
        Name for the indicator column

    Returns
    -------
    Optional[pd.DataFrame]
        Long format DataFrame with columns: Country Name, Year, indicator_name.
        Returns None if file cannot be loaded or parsed.
    """
    file_path = Path(file_path)

    try:
        # Read file based on extension
        if file_path.suffix == ".numbers":
            if not NUMBERS_PARSER_AVAILABLE:
                logger.warning("numbers-parser not installed")
                return None
            # Read Numbers file (skip 4 rows for World Bank format)
            doc = Document(str(file_path))
            sheet = doc.sheets[0]
            table = sheet.tables[0]

            data = []
            for row_idx in range(4, table.num_rows):  # Skip 4 metadata rows
                row_data = [
                    table.cell(row_idx, col_idx).value
                    for col_idx in range(table.num_cols)
                ]
                data.append(row_data)

            if not data:
                return None
            df = pd.DataFrame(data)
            if len(df) > 0:
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)

        elif file_path.suffix in [".xlsx", ".xls"]:
            df = _read_with_optional_skip(pd.read_excel, file_path)
        elif file_path.suffix == ".csv":
            df = _read_with_optional_skip(pd.read_csv, file_path)
        else:
            logger.warning(f"Unsupported file format: {file_path.suffix}")
            return None

    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None

    # Clean and standardize
    df = df.dropna(axis=1, how="all")
    df = _standardize_country_col(df)

    has_country_code = "Country Code" in df.columns
    df = df.drop(columns=["Indicator Name", "Indicator Code"], errors="ignore")

    # Detect year columns
    year_cols = _detect_year_columns(df)

    if not year_cols:
        # Maybe already in long format
        if "Year" in df.columns and indicator_name in df.columns:
            base_cols = ["Country Name", "Year", indicator_name]
            if has_country_code:
                base_cols.insert(1, "Country Code")
            return df[[c for c in base_cols if c in df.columns]]
        logger.warning(f"No year columns found in {file_path}")
        return None

    # Melt from wide to long format
    id_vars = ["Country Name"]
    if has_country_code:
        id_vars.append("Country Code")

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="Year",
        value_name=indicator_name,
    )

    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce").astype("Int64")
    df_long = df_long.dropna(subset=["Year"])

    return df_long


def load_data(
    data_path: Path, target: str = "political_stability", train_end_year: int = 2017
) -> Dict[str, pd.DataFrame]:
    """
    Load raw data files and prepare train/test splits.

    Pipeline:
    1. Load & Merge: World Bank indicators (wide→long format, OUTER join)
    2. Quality Filtering: UN whitelist, >30% missing removed
    3. Imputation: Progressive window median fill (±1, ±2, ±4 years, global)
    4. Temporal Split: Train 1996-2017, Test 2018-2023
    5. Export: Save to data/processed/ for reproducibility

    Parameters
    ----------
    data_path : Path
        Path to data/raw/ directory
    target : str
        Target variable name
    train_end_year : int
        Last year for training set

    Returns
    -------
    dict
        Keys: 'X_train', 'X_test', 'y_train', 'y_test', 'df_train', 'df_test'

    Notes
    -----
    This function writes processed train/test/full datasets to data/processed/
    for reproducibility.
    """
    data_path = Path(data_path)

    # Define files to load with their indicator names
    files_to_load = [
        ("political stability", "political_stability"),  # TARGET
        ("GDP per capita", "gdp_per_capita"),
        ("UNEMPLOYMENT_TOTAL", "unemployment"),
        ("inflation consumer", "inflation"),
        ("GDP_GROWTH_%", "gdp_growth"),
        ("effectiveness", "effectiveness"),
        ("rule of law", "rule_of_law"),
        ("trade", "trade"),
    ]

    # Load each file
    dataframes = []

    for file_base, indicator in files_to_load:
        for ext in [".csv", ".numbers", ".xlsx"]:
            file_path = data_path / f"{file_base}{ext}"
            if file_path.exists():
                df = load_data_file(str(file_path), indicator)
                if df is not None and len(df) > 0:
                    if "Country Name" in df.columns and "Year" in df.columns:
                        keep_cols = ["Country Name", "Year", indicator]
                        if "Country Code" in df.columns:
                            keep_cols.insert(1, "Country Code")
                        keep_cols = [c for c in keep_cols if c in df.columns]
                        df = df[keep_cols]
                        dataframes.append(df)
                break

    # Load HDI data (special handling for different format)
    hdi_file = data_path / "hdi_data.xlsx"
    if hdi_file.exists():
        try:
            hdi_raw = pd.read_excel(hdi_file)

            if "indexCode" in hdi_raw.columns and "value" in hdi_raw.columns:
                hdi_raw = hdi_raw[hdi_raw["indexCode"] == "HDI"].copy()

                column_mapping = {
                    "country": "Country Name",
                    "year": "Year",
                    "value": "hdi",
                }
                hdi_raw = hdi_raw.rename(columns=column_mapping)

                if all(c in hdi_raw.columns for c in ["Country Name", "Year", "hdi"]):
                    hdi_df = hdi_raw[["Country Name", "Year", "hdi"]].copy()
                    hdi_df["Year"] = pd.to_numeric(
                        hdi_df["Year"], errors="coerce"
                    ).astype("Int64")
                    dataframes.append(hdi_df)
                    logger.info(
                        f"HDI data loaded: {len(hdi_df)} rows, "
                        f"years {hdi_df['Year'].min()}-{hdi_df['Year'].max()}"
                    )
        except Exception as e:
            logger.warning(f"Failed to load HDI data: {e}")

    if not dataframes:
        raise FileNotFoundError(f"No data files found in {data_path}")

    # =========================================================================
    # MERGE ALL INDICATORS (OUTER join preserves all observations)
    # =========================================================================
    merged_df = dataframes[0]

    merge_keys = ["Country Name", "Year"]
    if all("Country Code" in df.columns for df in dataframes):
        merge_keys.insert(1, "Country Code")

    for df in dataframes[1:]:
        merged_df = merged_df.merge(
            df, on=merge_keys, how="outer", suffixes=("", "_dup")
        )

    merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith("_dup")]

    if target not in merged_df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    # Convert all columns to numeric (except Country Name/Code)
    for col in merged_df.columns:
        if col not in ["Country Name", "Country Code"]:
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

    merged_df = merged_df.sort_values(["Country Name", "Year"]).reset_index(drop=True)
    logger.info(
        f"Before filtering: {len(merged_df)} rows, "
        f"{merged_df['Country Name'].nunique()} countries"
    )

    # Drop rows with missing target
    merged_df = merged_df.dropna(subset=[target])
    logger.info(f"After dropping target NaN: {len(merged_df)} rows")

    # =========================================================================
    # COUNTRY NAME NORMALIZATION
    # =========================================================================
    merged_df["Country Name"] = merged_df["Country Name"].replace(COUNTRY_NAME_MAPPING)
    logger.info(f"Name normalization: {len(COUNTRY_NAME_MAPPING)} mappings applied")

    # =========================================================================
    # UN WHITELIST FILTER
    # =========================================================================
    countries_before = merged_df["Country Name"].unique()
    countries_removed = [c for c in countries_before if c not in COUNTRIES_TO_KEEP]

    merged_df = merged_df[merged_df["Country Name"].isin(COUNTRIES_TO_KEEP)]

    logger.info(
        f"Whitelist filter: {len(UN_MEMBER_STATES)} UN members + "
        f"{len(ADDITIONAL_TERRITORIES)} territories = {len(COUNTRIES_TO_KEEP)} total"
    )
    logger.info(f"Entities removed: {len(countries_removed)}")
    if countries_removed:
        logger.debug(f"Examples removed: {', '.join(list(countries_removed)[:5])}")
    logger.info(
        f"After whitelist: {len(merged_df)} rows, "
        f"{merged_df['Country Name'].nunique()} countries"
    )

    # =========================================================================
    # DATA QUALITY FILTER (>30% missing features)
    # =========================================================================
    all_numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in all_numeric_cols if col != target]

    countries_to_keep = []
    countries_eliminated = []

    for country in merged_df["Country Name"].unique():
        country_data = merged_df[merged_df["Country Name"] == country]

        if len(feature_cols) > 0:
            total_values = len(country_data) * len(feature_cols)
            missing_values = country_data[feature_cols].isnull().sum().sum()
            missing_percentage = missing_values / total_values

            if missing_percentage <= 0.30:
                countries_to_keep.append(country)
            else:
                countries_eliminated.append((country, missing_percentage))
        else:
            countries_to_keep.append(country)

    logger.info(f"Removed countries (>30% missing): {len(countries_eliminated)}")

    merged_df = merged_df[merged_df["Country Name"].isin(countries_to_keep)]
    logger.info(
        f"After 30% filter: {len(merged_df)} rows, "
        f"{len(countries_to_keep)} countries kept"
    )

    # ============================================================================
    # OPTIMIZED PROGRESSIVE WINDOW MEDIAN IMPUTATION
    # ============================================================================
    numeric_cols = [
        col
        for col in merged_df.select_dtypes(include=[np.number]).columns
        if col != target
    ]

    logger.info(f"Columns to fill: {numeric_cols}")
    logger.info("NaN per column BEFORE filling:")
    for col in numeric_cols:
        nan_count = merged_df[col].isnull().sum()
        if nan_count > 0:
            logger.info(f"  - {col}: {nan_count} NaN")

    logger.info("Starting Progressive Window Median Fill (OPTIMIZED)")

    # Pre-compute country data dictionary (OPTIMIZATION)
    country_data_cache = {
        country: merged_df[merged_df["Country Name"] == country]
        for country in merged_df["Country Name"].unique()
    }

    for col in numeric_cols:
        if not merged_df[col].isnull().any():
            continue

        filled_window_1 = filled_window_2 = filled_window_4 = filled_global = 0

        nan_mask = merged_df[col].isnull()
        nan_rows = merged_df[nan_mask]

        for idx in nan_rows.index:
            if pd.isnull(merged_df.loc[idx, col]):
                country = merged_df.loc[idx, "Country Name"]
                year = merged_df.loc[idx, "Year"]

                # Use cached country data (OPTIMIZATION)
                country_data = country_data_cache[country]

                # Stage 1: ±1 year window
                year_mask_1 = (country_data["Year"] >= year - 1) & (
                    country_data["Year"] <= year + 1
                )
                window_1_values = country_data[year_mask_1][col].dropna()

                if len(window_1_values) >= 2:
                    merged_df.loc[idx, col] = window_1_values.median()
                    filled_window_1 += 1
                    continue

                # Stage 2: ±2 year window
                year_mask_2 = (country_data["Year"] >= year - 2) & (
                    country_data["Year"] <= year + 2
                )
                window_2_values = country_data[year_mask_2][col].dropna()

                if len(window_2_values) >= 2:
                    merged_df.loc[idx, col] = window_2_values.median()
                    filled_window_2 += 1
                    continue

                # Stage 3: ±4 year window
                year_mask_4 = (country_data["Year"] >= year - 4) & (
                    country_data["Year"] <= year + 4
                )
                window_4_values = country_data[year_mask_4][col].dropna()

                if len(window_4_values) >= 2:
                    merged_df.loc[idx, col] = window_4_values.median()
                    filled_window_4 += 1
                    continue

                # Stage 4: Global median fallback
                merged_df.loc[idx, col] = merged_df[col].median()
                filled_global += 1

        # Report statistics
        total_filled = (
            filled_window_1 + filled_window_2 + filled_window_4 + filled_global
        )
        if total_filled > 0:
            logger.info(f"  {col}:")
            if filled_window_1 > 0:
                logger.info(f"    - Window ±1 year:  {filled_window_1} values")
            if filled_window_2 > 0:
                logger.info(f"    - Window ±2 years: {filled_window_2} values")
            if filled_window_4 > 0:
                logger.info(f"    - Window ±4 years: {filled_window_4} values")
            if filled_global > 0:
                logger.info(f"    - Global median:   {filled_global} values")

    logger.info(
        f"After median fill: {merged_df[numeric_cols].isnull().sum().sum()} NaN remaining"
    )
    logger.info(
        f"FINAL: {len(merged_df)} rows, {merged_df['Country Name'].nunique()} countries, "
        f"{merged_df.isnull().sum().sum()} total NaN"
    )

    # Set multi-index (Country Name, Year) - include Country Code if present
    index_cols = ["Country Name", "Year"]
    if "Country Code" in merged_df.columns:
        index_cols.insert(1, "Country Code")

    if all(col in merged_df.columns for col in ["Country Name", "Year"]):
        merged_df = merged_df.set_index(index_cols)

    # Temporal split
    train_df, test_df = get_train_test_split(merged_df.reset_index(), train_end_year)

    if all(col in train_df.columns for col in ["Country Name", "Year"]):
        train_df = train_df.set_index(index_cols)
        test_df = test_df.set_index(index_cols)

    # Separate features (X) and target (y)
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Save processed data
    processed_dir = data_path.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_dir / "train_data.csv")
    test_df.to_csv(processed_dir / "test_data.csv")

    full_data = pd.concat([train_df, test_df])
    full_data.to_csv(processed_dir / "full_data.csv")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "df_train": train_df,
        "df_test": test_df,
    }


def get_train_test_split(
    df: pd.DataFrame, train_end_year: int = 2017
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split panel data by year (temporal split, not random).

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with Year column
    train_end_year : int
        Last year for training set (default: 2017)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training and test DataFrames
    """
    df_train = df[df["Year"] <= train_end_year].copy()
    df_test = df[df["Year"] > train_end_year].copy()

    return df_train, df_test
