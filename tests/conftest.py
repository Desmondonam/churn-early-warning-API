import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from churn.api.main import app
from churn.data.load import load_raw
from churn.data.make_dataset import encode_target, split_xy
from churn.models.predict import METADATA_FILE, MODEL_FILE
from churn.models.train import make_pipeline

THRESHOLD = 0.3


@pytest.fixture(scope="session")
def model_dir(raw_df, tmp_path_factory):
    """Small self-contained model so API tests don't depend on models/ (or CI state)."""
    d = tmp_path_factory.mktemp("model")
    X, y = split_xy(encode_target(raw_df))
    joblib.dump(make_pipeline(LogisticRegression(max_iter=1000)).fit(X, y), d / MODEL_FILE)
    metrics = {k: 0.5 for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")}
    (d / METADATA_FILE).write_text(json.dumps({
        "model": "test_model", "trained_at": "2026-01-01T00:00:00+00:00",
        "threshold": THRESHOLD, "threshold_strategy": {"name": "max_f1"},
        "test_metrics": metrics, "input_features": [],
    }))  # fmt: skip
    return d


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("CHURN_MODEL_DIR", str(model_dir))
    with TestClient(app) as c:  # context manager runs the lifespan (model loading)
        yield c


class TestClientSession:
    """Lets ChurnApiClient use the in-process TestClient as its HTTP session.

    TestClient rejects per-request timeouts (meaningless in-process), so drop them.
    """

    def __init__(self, test_client):
        self._client = test_client

    def get(self, url, timeout=None, **kwargs):
        return self._client.get(url, **kwargs)

    def post(self, url, timeout=None, **kwargs):
        return self._client.post(url, **kwargs)


@pytest.fixture
def api_session(client):
    return TestClientSession(client)


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return load_raw()


@pytest.fixture
def customer() -> dict:
    """One raw customer record, shaped like an API request body."""
    return {
        "customerID": "0000-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
