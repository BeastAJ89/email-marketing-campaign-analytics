import pandas as pd
import numpy as np
from pathlib import Path

from config import RAW_EMAIL_DATA, INTERIM_DATA_DIR
from utils import get_logger, ensure_directory

logger = get_logger(__name__)

def load_data() -> pd.DataFrame:
    # Loads dataset from data/raw
    df = pd.read_csv(RAW_EMAIL_DATA)
    logger.info(f"Loaded dataset. Shape: {df.shape}")
    return df

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

def dates_parser(df: pd.DataFrame) -> pd.DataFrame:
    # Extracting date and time from mailing_date
    df["mailing_date"] = pd.to_datetime(
        df["mailing_date"],
        dayfirst=True,
        errors="coerce"
    )

    df["mailing_year"] = df["mailing_date"].dt.year
    df["mailing_month"] = df["mailing_date"].dt.month
    df["mailing_day"] = df["mailing_date"].dt.day
    df["mailing_day_of_week"] = df["mailing_date"].dt.dayofweek

    return df

def type_check(df: pd.DataFrame) -> pd.DataFrame:
    # Ensuring the correctness of numeric and categorical columns
    int_cols = [
        "open_flag", "click_flag", "conversion_flag", "unsubscribe_flag", "previous_purchases",
        "num_adults", "num_children", "age", "mailing_hour"
    ]

    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("int64")
    
    bool_cols = ["owns_home", "probable_renter"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    cat_cols = [
        "country", "region", "gender", "household_status", "presence_children", "consumer_archetypes",
        "marital_status", "income_range", "mosaic_segment", "language", "mailing_category", "device_type"
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
        
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["mailing_date"])
    df = df.dropna(subset=["first_name"])
    after = len(df)

    if before != after:
        logger.info(f"Dropped {before - after} rows due to invalid mailing_date format or missing first names.")
    
    df["previous_open_rate"].fillna(df["previous_open_rate"].median())
    df["previous_click_rate"].fillna(df["previous_click_rate"].median())

    return df

def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df["engagement_score"] = (
        0.6 * df["previous_open_rate"] +
        0.3 * df["previous_click_rate"] +
        0.1 * (df["previous_purchases"] / (df["previous_purchases"].max() + 1))
    )

    df["is_weekend"] = df["mailing_day_of_week"].isin([5,6]).astype(int)

    return df

def preprocess_raw_data(save_interim=True) -> pd.DataFrame:
    
    logger.info(">>> Starting preprocessing pipeline")

    df = load_data()
    df = normalize_column_names(df)
    df = dates_parser(df)
    df = type_check(df)
    df = clean_data(df)
    df = engineered_features(df)

    logger.info(f"Final preprocessed shape: {df.shape}")

    if save_interim:
        ensure_directory(INTERIM_DATA_DIR)
        output_path = INTERIM_DATA_DIR / "email_campaign_clean.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned dataset to {output_path}")
    
    return df

