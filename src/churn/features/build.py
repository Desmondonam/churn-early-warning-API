"""Preprocessing pipeline: raw customer records in, model-ready matrix out.

The same fitted pipeline is used at training time and inside the API, so any
cleaning a raw record needs (e.g. blank TotalCharges) must happen here.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Raw input columns the pipeline expects (customerID and Churn excluded).
RAW_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]  # fmt: skip

ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
]  # fmt: skip
PROTECTIVE_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
AUTO_PAYMENTS = ["Bank transfer (automatic)", "Credit card (automatic)"]
NEW_CUSTOMER_MONTHS = 6

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "n_protective_addons"]
BINARY = ["SeniorCitizen", "is_new_customer", "is_month_to_month", "auto_payment"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    *ADDONS, "Contract", "PaperlessBilling", "PaymentMethod",
]  # fmt: skip


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean raw columns and add engineered features (stateless, based on the EDA).

    - TotalCharges: coerce to numeric; blanks are unbilled new customers -> 0
    - "No internet service" / "No phone service" -> "No" (they duplicate
      InternetService / PhoneService exactly)
    - n_protective_addons, is_new_customer, is_month_to_month, auto_payment
    """

    def fit(self, X: pd.DataFrame, y=None):
        missing = set(RAW_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X[RAW_FEATURES].copy()

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
        df[ADDONS + ["MultipleLines"]] = df[ADDONS + ["MultipleLines"]].replace(
            {"No internet service": "No", "No phone service": "No"}
        )

        df["n_protective_addons"] = (df[PROTECTIVE_ADDONS] == "Yes").sum(axis=1)
        df["is_new_customer"] = (df["tenure"] <= NEW_CUSTOMER_MONTHS).astype(int)
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
        df["auto_payment"] = df["PaymentMethod"].isin(AUTO_PAYMENTS).astype(int)
        # Order must match get_feature_names_out(): set_output relabels columns by position.
        return df[list(self.get_feature_names_out())]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(NUMERIC + BINARY + CATEGORICAL, dtype=object)


def build_preprocessor() -> Pipeline:
    """Unfitted pipeline: FeatureEngineer -> scale numerics / one-hot categoricals."""
    encode = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("bin", "passthrough", BINARY),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("engineer", FeatureEngineer()), ("encode", encode)]).set_output(
        transform="pandas"
    )
