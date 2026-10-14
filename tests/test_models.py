import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.data.make_dataset import encode_target, split_xy
from churn.models.train import MODEL_PATH, make_pipeline, select


def _result(name, mean, std):
    return {"model": name, "average_precision_mean": mean, "average_precision_std": std}


def test_select_prefers_simpler_model_within_one_std():
    results = [_result("simple", 0.65, 0.01), _result("complex", 0.66, 0.02)]
    assert select(results)["model"] == "simple"


def test_select_takes_better_model_when_gap_is_real():
    results = [_result("simple", 0.60, 0.01), _result("complex", 0.66, 0.02)]
    assert select(results)["model"] == "complex"


def test_pipeline_fits_and_predicts_probabilities(raw_df, customer):
    X, y = split_xy(encode_target(raw_df.sample(1000, random_state=0)))
    model = make_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    proba = model.predict_proba(pd.DataFrame([customer]))[0, 1]
    assert 0.0 <= proba <= 1.0


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m churn.models.train` first")
def test_saved_model_ranks_risky_customer_above_loyal_one(customer):
    model = joblib.load(MODEL_PATH)
    risky = {**customer, "tenure": 1, "Contract": "Month-to-month",
             "InternetService": "Fiber optic", "PaymentMethod": "Electronic check"}
    loyal = {**customer, "tenure": 60, "Contract": "Two year", "InternetService": "DSL",
             "OnlineSecurity": "Yes", "TechSupport": "Yes",
             "PaymentMethod": "Credit card (automatic)", "TotalCharges": 3000.0}
    p = model.predict_proba(pd.DataFrame([risky, loyal]))[:, 1]
    assert p[0] > 0.5 > p[1]
