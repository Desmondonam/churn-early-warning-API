"""Read the raw Telco churn CSV."""

from pathlib import Path

import pandas as pd

from churn.config import RAW_DATA_PATH


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Return the raw dataset exactly as stored on disk (no cleaning)."""
    return pd.read_csv(path)
