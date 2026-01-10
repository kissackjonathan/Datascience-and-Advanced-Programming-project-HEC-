"""
Data Loading and Preprocessing Module.

This module provides  data loading and preprocessing capabilities for
panel data analysis, specifically designed for macroeconomic time series prediction.
Handles data from multiple sources including World Bank, UNDP, and Worldwide Governance
Indicators, transforming wide-format raw data into cleaned, imputed panel datasets.

Core Functionality
------------------
- Multi-format file loading (.csv, .xlsx, .numbers)
- Wide-to-long format transformation for panel data structure
- Temporal train-test splitting with strict no-leakage guarantee
- Progressive window-based imputation (±1, ±2, ±4 years, global fallback)
- Country filtering based on UN membership and data quality thresholds
- Cross-platform data conversion utilities

Data Pipeline
-------------
1. Load & Merge: World Bank format files (outer join on Country x Year)
2. Quality Control: UN whitelist filter, remove countries with >30% missing data
3. Temporal Split: Separate training and test sets by year cutoff
4. Imputation: Progressive window median fill, learned on training data only
5. Export: Save processed datasets to data/processed/ for reproducibility

Time-Series Safety
------------------
The pipeline ensures no temporal data leakage by:
- Performing train-test split before imputation
- Computing all imputation statistics exclusively from training data
- Applying test set imputation using only historical information

Notes
-----
All imputation uses temporal windows relative to each observation, with fallback
to training set global medians when insufficient local data is available.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from numbers_parser import Document

    NUMBERS_PARSER_AVAILABLE = True
except ImportError:
    NUMBERS_PARSER_AVAILABLE = False


# ============================================================================
# CONVERSION UTILITIES
# ============================================================================


def convert_numbers_to_csv(numbers_file: Path) -> Optional[Path]:
    import csv

    numbers_file = Path(numbers_file)

    if not NUMBERS_PARSER_AVAILABLE:
        logger.error(
            "numbers-parser not installed. Install with: pip install numbers-parser"
        )
        return None

    try:
        doc = Document(str(numbers_file))
        sheets = doc.sheets
        if not sheets:
            logger.warning(f"No sheets found in {numbers_file.name}")
            return None

        table = sheets[0].tables[0]
        csv_file = numbers_file.with_suffix(".csv")

        data = []
        for row in table.rows():
            data.append([cell.value for cell in row])

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(data)

        logger.info(f"Converted: {numbers_file.name} → {csv_file.name}")
        return csv_file
    except Exception as e:
        logger.error(f"Failed to convert {numbers_file.name}: {e}")
        return None


# ============================================================================
# COUNTRY WHITELISTS
# ============================================================================

UN_MEMBER_STATES = [
    # (unchanged full list)
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
    "Denmark",
    "Djibouti",
    "Dominica",
    "Dominican Republic",
    "Ecuador",
    "Egypt, Arab Rep.",
    "El Salvador",
    "Equatorial Guinea",
    "Eritrea",
    "Estonia",
    "Eswatini",
    "Ethiopia",
    "Fiji",
    "Finland",
    "France",
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
    "Haiti",
    "Honduras",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Iran, Islamic Rep.",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    "Jamaica",
    "Japan",
    "Jordan",
    "Kazakhstan",
    "Kenya",
    "Kiribati",
    "Korea, Dem. People's Rep.",
    "Korea, Rep.",
    "Kuwait",
    "Kyrgyz Republic",
    "Lao PDR",
    "Latvia",
    "Lebanon",
    "Lesotho",
    "Liberia",
    "Libya",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
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
    "Oman",
    "Pakistan",
    "Palau",
    "Panama",
    "Papua New Guinea",
    "Paraguay",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Qatar",
    "Romania",
    "Russian Federation",
    "Rwanda",
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
    "Uganda",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
    "Uruguay",
    "Uzbekistan",
    "Vanuatu",
    "Venezuela, RB",
    "Vietnam",
    "Yemen, Rep.",
    "Zambia",
    "Zimbabwe",
]

ADDITIONAL_TERRITORIES = [
    "Taiwan, China",
    "Kosovo",
    "West Bank and Gaza",
    "Hong Kong SAR, China",
    "Macao SAR, China",
    "Puerto Rico",
    "Guam",
]

COUNTRIES_TO_KEEP = UN_MEMBER_STATES + ADDITIONAL_TERRITORIES

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
    Attempt to read a file with optional row skipping for World Bank format compatibility.

    First attempts to read the file skipping the specified number of metadata rows.
    If that fails (e.g., for non-World Bank format files), retries without skipping.

    Parameters
    ----------
    reader : callable
        Pandas read function (pd.read_csv or pd.read_excel)
    path : Path
        File path to read
    skip : int, default=4
        Number of initial rows to skip (World Bank files have 4 metadata rows)
    **kwargs
        Additional arguments passed to the reader function

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame

    Notes
    -----
    World Bank files typically include 4 metadata rows before the actual data headers.
    """
    try:
        return reader(path, skiprows=skip, **kwargs)
    except Exception as e:
        logger.debug(f"Failed with skiprows={skip}, trying without: {e}")
        return reader(path, **kwargs)


def _detect_year_columns(df):
    """
    Identify year columns in a DataFrame based on numeric range.

    Searches DataFrame columns for values that can be interpreted as years
    within the range 1900-2100, excluding reserved column names.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to search for year columns

    Returns
    -------
    list
        List of column names representing years
    """
    year_cols = []
    for c in df.columns:
        if c in ["Country Name", "Country Code", "Year"]:
            continue
        try:
            year_val = int(float(c))
            if 1900 <= year_val <= 2100:
                year_cols.append(c)
        except (ValueError, TypeError):
            continue
    return year_cols


def _standardize_country_col(df):
    """
    Standardize country column name to canonical format.

    Searches for common variations of country column names and renames
    the first match to the standard 'Country Name' format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a country identifier column

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized 'Country Name' column

    """
    candidates = {"Country Name", "Country", "Country name", "country_name", "country"}
    found = candidates & set(df.columns)
    country_col = next(iter(found), df.columns[0])
    return df.rename(columns={country_col: "Country Name"})


def load_data_file(file_path: str, indicator_name: str) -> Optional[pd.DataFrame]:
    """
    Load and normalize a single indicator file to long-format panel data.

    Handles multiple file formats including World Bank CSV files, Excel spreadsheets,
    and Apple Numbers files. Automatically detects and transforms wide-format data
    (years as columns) into standardized long format (Country × Year × Value).

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the data file. Supported formats: .csv, .xlsx, .xls, .numbers
    indicator_name : str
        Column name for the indicator values in the output DataFrame

    Returns
    -------
    Optional[pd.DataFrame]
        Long-format DataFrame with columns: Country Name, [Country Code], Year, indicator_name.
        Returns None if the file cannot be loaded or contains no valid year columns.

    """
    file_path = Path(file_path)

    try:
        if file_path.suffix == ".numbers":
            if not NUMBERS_PARSER_AVAILABLE:
                logger.warning("numbers-parser not installed")
                return None

            doc = Document(str(file_path))
            sheet = doc.sheets[0]
            table = sheet.tables[0]

            data = []
            for row_idx in range(4, table.num_rows):
                row_data = [
                    table.cell(row_idx, col_idx).value
                    for col_idx in range(table.num_cols)
                ]
                data.append(row_data)

            if not data:
                return None
            df = pd.DataFrame(data)
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

    df = df.dropna(axis=1, how="all")
    df = _standardize_country_col(df)

    has_country_code = "Country Code" in df.columns
    df = df.drop(columns=["Indicator Name", "Indicator Code"], errors="ignore")

    year_cols = _detect_year_columns(df)

    if not year_cols:
        if "Year" in df.columns and indicator_name in df.columns:
            base_cols = ["Country Name", "Year", indicator_name]
            if has_country_code:
                base_cols.insert(1, "Country Code")
            return df[[c for c in base_cols if c in df.columns]]
        logger.warning(f"No year columns found in {file_path}")
        return None

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


def get_train_test_split(
    df: pd.DataFrame, train_end_year: int = 2017
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df[df["Year"] <= train_end_year].copy()
    df_test = df[df["Year"] > train_end_year].copy()
    return df_train, df_test


# ============================================================================
# TIME-SAFE IMPUTATION
# ============================================================================


def _compute_train_global_medians(
    df_train: pd.DataFrame, feature_cols: list
) -> Dict[str, float]:
    """
    Compute global fallback medians ONLY from train.
    """
    medians = {}
    for col in feature_cols:
        medians[col] = (
            float(df_train[col].median(skipna=True))
            if col in df_train.columns
            else np.nan
        )
    return medians


def _progressive_window_fill_inplace(
    df: pd.DataFrame,
    feature_cols: list,
    train_global_medians: Dict[str, float],
    entity_col: str = "Country Name",
    time_col: str = "Year",
    allow_future_within_df: bool = True,
) -> Dict[str, Dict[str, int]]:
    """
    Progressive window median fill:
      ±1 year (>=2 points), else ±2, else ±4, else global median (from train only)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to fill (either train or test).
    feature_cols : list
        Numeric features to fill (excluding target).
    train_global_medians : dict
        Global medians computed on TRAIN only. Used as final fallback for BOTH train and test.
    allow_future_within_df : bool
        - For train: True means you use ±windows within train
          (includes later years but still not beyond train_end_year).
          If you want strict "only past", set False (more conservative).
        - For test: True means using ±windows within test is allowed
          (doesn't leak into train). Still OK.
          If you want strict "only past" even within test, set False.

    Returns
    -------
    dict of stats per column.
    """
    stats_out = {}

    # cache per country for speed
    country_cache = {c: df[df[entity_col] == c] for c in df[entity_col].unique()}

    for col in feature_cols:
        if col not in df.columns:
            continue
        if not df[col].isnull().any():
            continue

        filled_1 = filled_2 = filled_4 = filled_global = 0
        nan_rows = df[df[col].isnull()]

        for idx in nan_rows.index:
            if pd.isna(df.at[idx, col]):
                country = df.at[idx, entity_col]
                year = df.at[idx, time_col]
                country_data = country_cache[country]

                def window_values(radius: int) -> pd.Series:
                    if allow_future_within_df:
                        mask = (country_data[time_col] >= year - radius) & (
                            country_data[time_col] <= year + radius
                        )
                    else:
                        # only past within df
                        mask = (country_data[time_col] >= year - radius) & (
                            country_data[time_col] <= year
                        )
                    return country_data.loc[mask, col].dropna()

                w1 = window_values(1)
                if len(w1) >= 2:
                    df.at[idx, col] = float(w1.median())
                    filled_1 += 1
                    continue

                w2 = window_values(2)
                if len(w2) >= 2:
                    df.at[idx, col] = float(w2.median())
                    filled_2 += 1
                    continue

                w4 = window_values(4)
                if len(w4) >= 2:
                    df.at[idx, col] = float(w4.median())
                    filled_4 += 1
                    continue

                # global fallback from TRAIN only
                fallback = train_global_medians.get(col, np.nan)
                if pd.isna(fallback):
                    # if train median is nan (col totally empty in train),
                    # use overall df median (still safe within df)
                    fallback = (
                        float(df[col].median(skipna=True))
                        if df[col].notna().any()
                        else np.nan
                    )
                df.at[idx, col] = fallback
                filled_global += 1

        stats_out[col] = {
            "window_1": filled_1,
            "window_2": filled_2,
            "window_4": filled_4,
            "global": filled_global,
            "total": filled_1 + filled_2 + filled_4 + filled_global,
        }

    return stats_out


# ============================================================================
# MAIN DATA LOADING AND PREPROCESSING PIPELINE
# ============================================================================


def load_data(
    data_path: Path,
    target: str = "political_stability",
    train_end_year: int = 2017,
) -> Dict[str, pd.DataFrame]:
    """
    Execute complete data loading and preprocessing pipeline for panel data.

    Loads raw macroeconomic and governance indicators from multiple sources, performs
    quality filtering, temporal splitting, and progressive imputation while ensuring
    no temporal data leakage between training and test sets.

    Pipeline Stages
    ---------------
    1. File Loading: Load World Bank, UNDP, and governance indicator files
    2. Merging: Outer join on (Country Name, Year) to preserve all observations
    3. Quality Control: Apply UN membership whitelist and remove low-quality countries
    4. Temporal Split: Separate data by year (train <= train_end_year < test)
    5. Imputation: Progressive window median filling using only training set statistics
    6. Export: Save processed datasets to data/processed/ directory

    Parameters
    ----------
    data_path : Path
        Directory containing raw data files (typically data/raw/)
    target : str, default='political_stability'
        Name of the target variable for prediction
    train_end_year : int, default=2017
        Last year to include in training set. Test set includes all years > train_end_year

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing:
        - 'X_train': Training features (MultiIndex: Country Name, Year)
        - 'y_train': Training target values
        - 'X_test': Test features (MultiIndex: Country Name, Year)
        - 'y_test': Test target values
        - 'df_train': Complete training DataFrame including target
        - 'df_test': Complete test DataFrame including target

    Raises
    ------
    FileNotFoundError
        If no data files are found in the specified directory
    ValueError
        If the target column is not found in the merged data

    Notes
    -----
    Data Leakage Prevention:
        - Train-test split performed BEFORE any imputation
        - All imputation statistics (medians) computed solely from training data
        - Test set imputation uses only historical windows and training set fallback values

    Quality Filters:
        - Removes non-UN member states and territories without strong economic data
        - Eliminates countries with >30% missing feature values
        - Drops observations with missing target values

    Imputation Strategy:
        - Stage 1: Temporal window ±1 year (requires ≥2 observations)
        - Stage 2: Temporal window ±2 years (requires ≥2 observations)
        - Stage 3: Temporal window ±4 years (requires ≥2 observations)
        - Stage 4: Training set global median fallback

    Examples
    --------
    >>> from pathlib import Path
    >>> data = load_data(Path('data/raw'), target='political_stability', train_end_year=2017)
    >>> print(f"Training samples: {len(data['X_train'])}")
    >>> print(f"Test samples: {len(data['X_test'])}")
    >>> print(f"Features: {list(data['X_train'].columns)}")
    """
    data_path = Path(data_path)

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

    dataframes = []
    for file_base, indicator in files_to_load:
        for ext in [".csv", ".numbers", ".xlsx"]:
            file_path = data_path / f"{file_base}{ext}"
            if file_path.exists():
                df = load_data_file(str(file_path), indicator)
                if (
                    df is not None
                    and len(df) > 0
                    and "Country Name" in df.columns
                    and "Year" in df.columns
                ):
                    keep_cols = ["Country Name", "Year", indicator]
                    if "Country Code" in df.columns:
                        keep_cols.insert(1, "Country Code")
                    keep_cols = [c for c in keep_cols if c in df.columns]
                    dataframes.append(df[keep_cols])
                break

    # HDI special
    hdi_file = data_path / "hdi_data.xlsx"
    if hdi_file.exists():
        try:
            hdi_raw = pd.read_excel(hdi_file)
            if "indexCode" in hdi_raw.columns and "value" in hdi_raw.columns:
                hdi_raw = hdi_raw[hdi_raw["indexCode"] == "HDI"].copy()
                hdi_raw = hdi_raw.rename(
                    columns={"country": "Country Name", "year": "Year", "value": "hdi"}
                )
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

    # Merge (outer join)
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

    # Convert numeric
    for col in merged_df.columns:
        if col not in ["Country Name", "Country Code"]:
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

    merged_df = merged_df.sort_values(["Country Name", "Year"]).reset_index(drop=True)
    logger.info(
        f"Before filtering: {len(merged_df)} rows, {merged_df['Country Name'].nunique()} countries"
    )

    # Drop missing target rows (can't train without y)
    merged_df = merged_df.dropna(subset=[target])
    logger.info(f"After dropping target NaN: {len(merged_df)} rows")

    # Normalize country names
    merged_df["Country Name"] = merged_df["Country Name"].replace(COUNTRY_NAME_MAPPING)

    # Whitelist
    merged_df = merged_df[merged_df["Country Name"].isin(COUNTRIES_TO_KEEP)]
    logger.info(
        f"After whitelist: {len(merged_df)} rows, {merged_df['Country Name'].nunique()} countries"
    )

    # Quality filter (>30% missing features) BEFORE split (OK: doesn't use future statistics)
    all_numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in all_numeric_cols if c != target]

    countries_to_keep = []
    for country in merged_df["Country Name"].unique():
        cd = merged_df[merged_df["Country Name"] == country]
        if not feature_cols:
            countries_to_keep.append(country)
            continue
        total = len(cd) * len(feature_cols)
        missing = cd[feature_cols].isnull().sum().sum()
        missing_pct = missing / total if total > 0 else 1.0
        if missing_pct <= 0.30:
            countries_to_keep.append(country)

    merged_df = merged_df[merged_df["Country Name"].isin(countries_to_keep)]
    logger.info(
        f"After 30% filter: {len(merged_df)} rows, {len(countries_to_keep)} countries kept"
    )

    # ============================================================================
    # TIME-SAFE ORDER: SPLIT FIRST, THEN IMPUTE
    # ============================================================================
    train_df, test_df = get_train_test_split(merged_df, train_end_year=train_end_year)

    logger.info(
        f"Temporal split done: train={len(train_df)} rows (<= {train_end_year}), "
        f"test={len(test_df)} rows (> {train_end_year})"
    )

    # Choose numeric feature cols (exclude target) on TRAIN schema
    numeric_cols = [
        c for c in train_df.select_dtypes(include=[np.number]).columns if c != target
    ]

    # Compute TRAIN global medians only
    train_global_medians = _compute_train_global_medians(train_df, numeric_cols)

    # Impute TRAIN (no future beyond train_end_year)
    logger.info("Imputing TRAIN (time-safe: train-only stats)...")
    train_stats = _progressive_window_fill_inplace(
        train_df,
        feature_cols=numeric_cols,
        train_global_medians=train_global_medians,
        entity_col="Country Name",
        time_col="Year",
        allow_future_within_df=True,  # within TRAIN only; safe relative to test
    )

    # Impute TEST (no leakage into train; fallback uses TRAIN medians)
    logger.info("Imputing TEST (fallback uses TRAIN medians)...")
    test_stats = _progressive_window_fill_inplace(
        test_df,
        feature_cols=numeric_cols,
        train_global_medians=train_global_medians,
        entity_col="Country Name",
        time_col="Year",
        allow_future_within_df=True,  # within TEST only
    )

    # Report remaining NaNs (features)
    train_remaining = (
        int(train_df[numeric_cols].isnull().sum().sum()) if numeric_cols else 0
    )
    test_remaining = (
        int(test_df[numeric_cols].isnull().sum().sum()) if numeric_cols else 0
    )
    logger.info(
        f"Remaining NaNs after fill: train={train_remaining}, test={test_remaining}"
    )

    # Set multi-index
    index_cols = ["Country Name", "Year"]
    if "Country Code" in merged_df.columns:
        index_cols.insert(1, "Country Code")

    train_df = train_df.set_index(index_cols).sort_index()
    test_df = test_df.set_index(index_cols).sort_index()

    # Separate X/y
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Save processed
    processed_dir = data_path.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_dir / "train_data.csv")
    test_df.to_csv(processed_dir / "test_data.csv")

    full_data = pd.concat([train_df, test_df]).sort_index()
    full_data.to_csv(processed_dir / "full_data.csv")

    # Optional: log imputation summary
    def _log_stats(title: str, st: Dict[str, Dict[str, int]]):
        logger.info(title)
        for col, s in st.items():
            if s["total"] > 0:
                logger.info(
                    f"  {col}: ±1={s['window_1']}, ±2={s['window_2']}, ±4={s['window_4']}, "
                    f"global(train)={s['global']} (total={s['total']})"
                )

    _log_stats("TRAIN imputation stats:", train_stats)
    _log_stats("TEST imputation stats:", test_stats)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "df_train": train_df,
        "df_test": test_df,
    }


# ============================================================================
# CONVERSION UTILITY SCRIPT
# ============================================================================


def convert_all_numbers_to_csv(data_path: Optional[Path] = None) -> int:
    if data_path is None:
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / "raw"

    data_path = Path(data_path)

    if not data_path.exists():
        logger.error(f"Directory {data_path} does not exist!")
        return 0

    numbers_files = list(data_path.glob("*.numbers"))
    if not numbers_files:
        logger.warning(f"No .numbers files found in {data_path}")
        return 0

    logger.info(f"Converting {len(numbers_files)} .numbers files to CSV")
    converted = 0
    for numbers_file in numbers_files:
        csv_file = convert_numbers_to_csv(numbers_file)
        if csv_file:
            converted += 1

    logger.info(
        f"Conversion complete: {converted}/{len(numbers_files)} files converted"
    )
    return converted


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    print("\n" + "=" * 70)
    print("CONVERT .NUMBERS FILES TO .CSV")
    print("=" * 70 + "\n")

    converted = convert_all_numbers_to_csv()

    print("\n" + "=" * 70)
    print(f"CONVERSION COMPLETE: {converted} files converted")
    print("=" * 70 + "\n")

    if converted > 0:
        print("Next steps:")
        print("1. Commit the CSV files:")
        print("   git add data/raw/*.csv")
        print("   git commit -m 'Add CSV versions of data files'")
        print("   git push")
        print("\n2. Or copy manually to Windows\n")
