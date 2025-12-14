from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.utils import get_logger

logger = get_logger(__name__)

def group_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    # Numeric features 
    # Features that get one-hot encoding
    # Features that get ordinal encoding

    numeric_candidate = [
        "age", "num_adults", "num_children", "previous_open_rate", "previous_click_rate", "previous_purchases",
        "engagement_score", "mailing_hour", "mailing_year", "mailing_month", "mailing_day", "mailing_day_of_week",
        "is_weekend",
    ]

    numeric_features = [c for c in numeric_candidate if c in df.columns]

    # One-hot: for low or medium cardinality

    onehot_candidate = [
        "gender", "presence_children", "owns_home", "probable_renter", "device_type", "mailing_category",
        "consumer_archetypes", "mosaic_segment", "income_range", "household_status", "marital_status",
    ]

    onehot_features = [c for c in onehot_candidate if c in df.columns]

    # Ordinal
    ordinal_candidate = [
        "country", "region", "language"
    ]

    ordinal_features = [c for c in ordinal_candidate if c in df.columns]

    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"One-hot features: {onehot_features}")
    logger.info(f"Ordinal features: {ordinal_features}")

    return {
        "numeric": numeric_features,
        "onehot": onehot_features,
        "ordinal": ordinal_features,
    }

def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, Dict[str, List[str]]]:
    # This will pass numeric features through
    # One-hot will encode selected categoricals
    # Ordinal will encode other categoricals

    feature_groups = group_features(df)

    numeric_features = feature_groups["numeric"]
    onehot_features = feature_groups["onehot"]
    ordinal_features = feature_groups["ordinal"]

    transformers = []

    if numeric_features:
        transformers.append(
            ("num", "passthrough", numeric_features)
        )
    
    if onehot_features:
        transformers.append(
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                onehot_features,
            )
        )

    if ordinal_features:
        transformers.append(
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_features,
            )
        )
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor, feature_groups

def get_feature_matrix_and_target(df: pd.DataFrame, target_col: str="open_flag",) -> Tuple[pd.DataFrame, pd.Series]:
    # This will separate features (X) and target (y) from the preprocessed dataframe

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    return X,y

def prepare_data(df: pd.DataFrame, target_col: str = "open_flag", test_size: float = 0.2, 
                 random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
                                                  ColumnTransformer, List[str], List[str]]:
    # This will prepare data for modelling
    # split into train/test -> build preprocessor -> fit preprocessor -> transform
    # -> return X_train, X_test, y_train, y_test, preprocessor, feature_names

    logger.info(f"Preparing model data with target: {target_col}")

    raw_feature_cols = df.columns.tolist()

    X, y = get_feature_matrix_and_target(df, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    logger.info(f"Train shape: {X_train.shape}, Test shape : {X_test.shape}")

    preprocessor, feature_groups = build_preprocessor(X_train)

    # Fitting only the training data
    preprocessor.fit(X_train)

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names: List[str] = []

    num_features = feature_groups["numeric"]
    feature_names.extend(num_features)

    if feature_groups["onehot"]:
        onehot = preprocessor.named_transformers_["onehot"]
        onehot_feature_names = list(onehot.get_feature_names_out(feature_groups["onehot"]))
        feature_names.extend(onehot_feature_names)
    
    feature_names.extend(feature_groups["ordinal"])

    logger.info(f"Total transformed feature dimension: {X_train_transformed.shape[1]}")

    return (
        X_train_transformed, X_test_transformed,
        y_train.to_numpy(), y_test.to_numpy(),
        preprocessor, feature_names, raw_feature_cols
    )