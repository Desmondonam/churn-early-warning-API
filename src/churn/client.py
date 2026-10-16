"""HTTP client for the churn API, used by the Streamlit app.

Deliberately free of ML dependencies (no scikit-learn, no model): every prediction
comes from the deployed API, so the UI can never disagree with it.
"""

import json
from dataclasses import dataclass, field
from typing import Any, get_args

import pandas as pd
import requests

from churn.api.schemas import MAX_BATCH_SIZE, Customer

FEATURE_COLUMNS = [name for name in Customer.model_fields if name != "customerID"]


def field_options(name: str) -> list:
    """Allowed values for a categorical field, read from the API's own schema."""
    return list(get_args(Customer.model_fields[name].annotation))


class ApiError(Exception):
    """The API was unreachable or rejected the request; message is user-readable."""


def _describe(response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # pydantic validation errors
        lines = []
        for err in detail[:10]:
            loc = [str(p) for p in err.get("loc", []) if p not in ("body", "customers")]
            lines.append(f"{' > '.join(loc) or 'request'}: {err.get('msg')}")
        return "The API rejected the data:\n" + "\n".join(f"- {line}" for line in lines)
    return f"API error {response.status_code}: {detail or response.text[:300]}"


def prepare_records(df: pd.DataFrame) -> list[dict]:
    """Turn an uploaded customer table into API-ready JSON records.

    Keeps only the columns the API accepts (extra columns such as Churn are dropped),
    and turns blank TotalCharges (unbilled new customers) into null.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ApiError(f"The file is missing required columns: {', '.join(missing)}")

    cols = (["customerID"] if "customerID" in df.columns else []) + FEATURE_COLUMNS
    out = df[cols].copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    if "customerID" in out:
        out["customerID"] = out["customerID"].astype(str)
    # to_json handles NaN -> null and numpy scalar types
    return json.loads(out.to_json(orient="records"))


@dataclass
class ChurnApiClient:
    base_url: str
    session: Any = field(default_factory=requests.Session)
    timeout: float = 90.0  # Render free tier can take ~60 s to wake up
    batch_size: int = MAX_BATCH_SIZE

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = self.base_url.rstrip("/") + path
        try:
            response = getattr(self.session, method)(url, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise ApiError(
                f"Could not reach the API at {self.base_url} ({type(exc).__name__})."
            ) from exc
        if response.status_code >= 400:
            raise ApiError(_describe(response))
        return response.json()

    def health(self) -> dict:
        return self._request("get", "/health")

    def model_info(self) -> dict:
        return self._request("get", "/model")

    def predict(self, customer: dict) -> dict:
        return self._request("post", "/predict", json=customer)

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score every row (in API-sized chunks); returns df plus prediction columns."""
        records = prepare_records(df)
        predictions = []
        for start in range(0, len(records), self.batch_size):
            chunk = records[start : start + self.batch_size]
            predictions += self._request("post", "/predict/batch", json={"customers": chunk})[
                "predictions"
            ]
        scored = pd.DataFrame(predictions, index=df.index)
        return df.assign(
            churn_probability=scored["churn_probability"],
            will_churn=scored["will_churn"],
            risk_band=scored["risk_band"],
        )
