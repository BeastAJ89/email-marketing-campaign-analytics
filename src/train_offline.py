# src/train_offline.py

from pathlib import Path
import joblib
import pandas as pd

from preprocess import preprocess_raw_data
from features import prepare_data
from modeling import train_and_evaluate
from utils import get_logger

logger = get_logger(__name__) if "get_logger" in globals() else None

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" /"email_campaign_synthetic.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

TARGET_COL = "open_flag"

def main():
    if logger:
        logger.info(">>> Starting offline training pipeline")
        logger.info(f"Loading raw data data from : {DATA_PATH}")
    
    df_processed = preprocess_raw_data(save_interim=False)

    (X_train, X_test, y_train, y_test, preprocessor, feature_names, raw_feature_cols) = prepare_data(
        df=df_processed, target_col = TARGET_COL,
        )
    
    model, metrics = train_and_evaluate(
        "rf",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
    )

    # Saving our artifacts
    model_path = ARTIFACT_DIR / "rf_model_deploy.joblib"
    prep_path = ARTIFACT_DIR / "preprocessor.joblib"
    meta_path = ARTIFACT_DIR / "metadata.joblib"

    joblib.dump(model, model_path, compress=3)
    joblib.dump(preprocessor, prep_path)

    metadata = {
        "target_col": TARGET_COL,
        "raw_feature_cols": raw_feature_cols,
        "metrics": metrics,
        "dropdowns": {
            "country": sorted(df_processed["country"].dropna().unique().tolist()),
            "device_type": sorted(df_processed["device_type"].dropna().unique().tolist()),
            "consumer_archetypes": sorted(df_processed["consumer_archetypes"].dropna().unique().tolist()),
            "mosaic_segment": sorted(df_processed["mosaic_segment"].dropna().unique().tolist()),
            "mailing_category": sorted(df_processed["mailing_category"].dropna().unique().tolist()),
        }
    }

    joblib.dump(metadata, meta_path)

    if logger:
        logger.info(f"Saved model to : {model_path}")
        logger.info(f"Saved preprocessor to : {prep_path}")
        logger.info(f"Saved metadata to : {meta_path}")
        logger.info(">>> Offline training complete.")
    
if __name__ == "__main__":
    main()