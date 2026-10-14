"""Inference: load the trained pipeline + metadata and score raw customer records."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn.config import MODELS_DIR

MODEL_FILE = "churn_model.joblib"
METADATA_FILE = "metadata.json"
HIGH_RISK_PROBABILITY = 0.5  # calibrated: churn more likely than not


def model_dir_from_env() -> Path:
    return Path(os.environ.get("CHURN_MODEL_DIR", MODELS_DIR))


@dataclass
class ChurnPredictor:
    model: object
    metadata: dict

    @classmethod
    def load(cls, model_dir: Path | None = None) -> "ChurnPredictor":
        model_dir = model_dir or model_dir_from_env()
        model_path, meta_path = model_dir / MODEL_FILE, model_dir / METADATA_FILE
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {model_dir}. "
                "Run `python -m churn.models.train` then `python -m churn.models.evaluate`."
            )
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if "threshold" not in metadata:
            raise ValueError(
                "metadata.json has no threshold; run `python -m churn.models.evaluate`"
            )
        return cls(model=joblib.load(model_path), metadata=metadata)

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    def predict_proba(self, records: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(records)[:, 1]

    def risk_band(self, probability: float) -> str:
        if probability >= HIGH_RISK_PROBABILITY:
            return "high"
        return "medium" if probability >= self.threshold else "low"
