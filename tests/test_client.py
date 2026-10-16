"""The Streamlit app's API client, run against the real FastAPI app (in-process)."""

import pandas as pd
import pytest

from churn.client import ApiError, ChurnApiClient, field_options, prepare_records
from tests.conftest import THRESHOLD


@pytest.fixture
def api(api_session):
    return ChurnApiClient("http://testserver", session=api_session)


def test_health_and_model_info(api):
    assert api.health()["model_loaded"] is True
    assert api.model_info()["threshold"] == THRESHOLD


def test_predict(api, customer):
    result = api.predict(customer)
    assert set(result) == {"customerID", "churn_probability", "will_churn", "risk_band"}


def test_score_frame_on_raw_csv_rows(api, raw_df):
    """Raw rows include the Churn column and blank TotalCharges; both must be handled."""
    df = raw_df[raw_df["tenure"] == 0].head(3)
    df = pd.concat([df, raw_df.head(47)])
    scored = api.score_frame(df)
    assert len(scored) == 50
    assert scored.index.equals(df.index)
    assert scored["churn_probability"].between(0, 1).all()
    assert (scored["will_churn"] == (scored["churn_probability"] >= THRESHOLD)).all()


def test_score_frame_chunks_large_files_in_order(api, raw_df):
    api.batch_size = 40
    df = raw_df.head(100)
    chunked = api.score_frame(df)
    api.batch_size = 1000
    single = api.score_frame(df)
    pd.testing.assert_series_equal(chunked["churn_probability"], single["churn_probability"])


def test_missing_columns_are_reported(raw_df):
    with pytest.raises(ApiError, match="Contract"):
        prepare_records(raw_df.drop(columns=["Contract"]))


def test_invalid_values_give_readable_error(api, raw_df):
    df = raw_df.head(3).copy()
    df.loc[df.index[1], "Contract"] = "Weekly"
    with pytest.raises(ApiError, match="Contract") as exc:
        api.score_frame(df)
    assert "rejected" in str(exc.value)


def test_unreachable_api_gives_readable_error():
    api = ChurnApiClient("http://127.0.0.1:9", timeout=2)
    with pytest.raises(ApiError, match="Could not reach the API"):
        api.health()


def test_prepare_records_blank_total_charges_become_null(raw_df):
    records = prepare_records(raw_df[raw_df["tenure"] == 0].head(1))
    assert records[0]["TotalCharges"] is None
    assert "Churn" not in records[0]


def test_field_options_come_from_api_schema():
    assert field_options("Contract") == ["Month-to-month", "One year", "Two year"]
    assert field_options("SeniorCitizen") == [0, 1]
