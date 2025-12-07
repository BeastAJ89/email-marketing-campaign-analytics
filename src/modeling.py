import numpy as np
import pandas as pd

from typing import Dict, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_curve,)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from utils import get_logger

logger = get_logger(__name__)

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    # Training a baseline model

    logger.info("---Training Logistic Regression Model")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=300) -> RandomForestClassifier:

    logger.info("---Training Random Forest Model")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train) -> Any:
    
    if not HAS_XGB:
        raise ImportError("XGBoost is not installed.")
    
    logger.info("---Training XGBoost model")
    model = XGBClassifier(
        n_estimators=300, learning_rate= 0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss", n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    # computing evaluation metrics for our classifiers

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    metrics = {
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    logger.info(f"Model evaluation: {metrics}")

    return metrics

def compute_lift(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({
        "y_true": y_true, "y_prob": y_prob
    })

    df["decile"] = pd.qcut(df["y_prob"], bins, labels=False, duplicates="drop")

    lift_table = (
        df.groupby("decile").agg(
            total=("y_true", "count"),
            positives=("y_true", "sum"),
        ).sort_index(ascending=False)
    )

    lift_table["response_rate"] = lift_table["positives"] / lift_table["total"]
    lift_table["cumulative_positives"] = lift_table["positives"].cumsum()
    lift_table["lift"] = lift_table["response_rate"] / df["y_true"].mean()

    return lift_table

def train_and_evaluate(model_name: str, X_train, y_train, X_test, y_test,) -> Tuple[Any, Dict[str, Any]]:
    # This will run training pipeline for a specific model

    logger.info(f"Running training pipeline for model: {model_name}")

    if model_name == "logreg":
        model = train_logistic_regression(X_train, y_train)
    
    elif model_name == "rf":
        model = train_random_forest(X_train, y_train)
    
    elif model_name == "xgb":
        if not HAS_XGB:
            raise ImportError("XGBoost is not installed.")
        model = train_xgboost(X_train, y_train)
    
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    metrics = evaluate_model(model, X_test, y_test)

    # Computing Lift
    y_prob = model.predict_proba(X_test)[:, 1]
    lift_table = compute_lift(y_test, y_prob)

    metrics["lift"] = lift_table

    logger.info("Training and evaluation pipeline complete.")

    return model, metrics