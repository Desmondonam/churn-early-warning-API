import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from churn.api.main import app
from churn.data.make_dataset import encode_target, split_xy
from churn.models.predict import METADATA_FILE, MODEL_FILE, ChurnPredictor
from churn.models.train import make_pipeline

THRESHOLD = 0.3


@pytest.fixture(scope="module")
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


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model").json()
    assert body["model"] == "test_model"
    assert body["threshold"] == THRESHOLD


def test_predict_single(client, customer):
    r = client.post("/predict", json=customer)
    assert r.status_code == 200
    body = r.json()
    assert body["customerID"] == customer["customerID"]
    assert 0 <= body["churn_probability"] <= 1
    assert body["will_churn"] == (body["churn_probability"] >= THRESHOLD)
    assert body["risk_band"] in {"low", "medium", "high"}


def test_risky_customer_scores_above_loyal_one(client, customer):
    risky = {**customer, "tenure": 1, "Contract": "Month-to-month",
             "InternetService": "Fiber optic", "PaymentMethod": "Electronic check"}
    loyal = {**customer, "tenure": 60, "Contract": "Two year", "InternetService": "DSL",
             "OnlineSecurity": "Yes", "TechSupport": "Yes",
             "PaymentMethod": "Credit card (automatic)", "TotalCharges": 3000.0}
    p_risky = client.post("/predict", json=risky).json()["churn_probability"]
    p_loyal = client.post("/predict", json=loyal).json()["churn_probability"]
    assert p_risky > p_loyal


def test_missing_total_charges_is_allowed_for_new_customers(client, customer):
    customer.update(tenure=0)
    del customer["TotalCharges"]
    assert client.post("/predict", json=customer).status_code == 200


def test_batch(client, customer):
    customers = [customer, {**customer, "customerID": "B", "Contract": "Two year"}]
    body = client.post("/predict/batch", json={"customers": customers}).json()
    assert body["n_customers"] == 2
    assert [p["customerID"] for p in body["predictions"]] == ["0000-TEST", "B"]
    assert body["n_flagged"] == sum(p["will_churn"] for p in body["predictions"])


@pytest.mark.parametrize("change", [
    {"Contract": "Weekly"},           # unknown category
    {"tenure": -1},                   # out of range
    {"MonthlyCharges": "a lot"},      # wrong type
    {"tenur": 5},                     # typo -> unknown field
])  # fmt: skip
def test_invalid_input_is_rejected(client, customer, change):
    assert client.post("/predict", json={**customer, **change}).status_code == 422


def test_missing_required_field_is_rejected(client, customer):
    del customer["Contract"]
    r = client.post("/predict", json=customer)
    assert r.status_code == 422
    assert "Contract" in r.text


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_risk_bands(model_dir):
    p = ChurnPredictor.load(model_dir)
    assert p.risk_band(THRESHOLD - 0.01) == "low"
    assert p.risk_band(THRESHOLD) == "medium"
    assert p.risk_band(0.5) == "high"


def test_missing_artifacts_fail_with_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="churn.models.train"):
        ChurnPredictor.load(tmp_path)
