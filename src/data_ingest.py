from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from config import RAW_EMAIL_DATA, INTERIM_DATA_DIR
from utils import get_logger, ensure_directory

logger = get_logger(__name__)

def load_raw_email_data(path: Optional[Path] = None) -> pd.DataFrame:
    # Loads the raw synthetic dataset
    csv_path = Path(path) if path is not None else RAW_EMAIL_DATA

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {csv_path}")
    
    logger.info(f"Loading raw email data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataframe with shape: {df.shape}")
    return df

def save_interim_data(df: pd.DataFrame, filename: str) -> Path:
    ensure_directory(INTERIM_DATA_DIR)
    output_path = INTERIM_DATA_DIR / filename
    logger.info(f"Saving interim data to : {output_path}")
    df.to_csv(output_path, index=False)
    return output_path