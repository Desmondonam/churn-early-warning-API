import pandas as pd
import pytest

from churn.features.build import FeatureEngineer, build_preprocessor


@pytest.fixture(scope="module")
def fitted(raw_df):
    return build_preprocessor().fit(raw_df)


def test_output_is_numeric_and_complete(fitted, raw_df):
    out = fitted.transform(raw_df)
    assert len(out) == len(raw_df)
    assert out.notna().all().all()
    assert all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes)


def test_id_and_target_are_not_features(fitted):
    names = list(fitted.get_feature_names_out())
    assert not any(n.startswith(("customerID", "Churn")) for n in names)


def test_blank_total_charges_become_zero(customer):
    customer.update(tenure=0, TotalCharges=" ")
    out = FeatureEngineer().fit_transform(pd.DataFrame([customer]))
    assert out.loc[0, "TotalCharges"] == 0.0
    assert out.loc[0, "is_new_customer"] == 1


def test_redundant_levels_are_collapsed(raw_df):
    out = FeatureEngineer().fit_transform(raw_df)
    for col in ["OnlineSecurity", "StreamingTV", "MultipleLines"]:
        assert set(out[col].unique()) == {"Yes", "No"}


def test_engineered_features(customer):
    customer.update(OnlineSecurity="Yes", TechSupport="Yes", OnlineBackup="No",
                    PaymentMethod="Credit card (automatic)", Contract="Two year", tenure=40)
    row = FeatureEngineer().fit_transform(pd.DataFrame([customer])).iloc[0]
    assert row["n_protective_addons"] == 2
    assert row["auto_payment"] == 1
    assert row["is_month_to_month"] == 0
    assert row["is_new_customer"] == 0


def test_single_record_like_an_api_request(fitted, customer):
    out = fitted.transform(pd.DataFrame([customer]))
    assert out.shape == (1, len(fitted.get_feature_names_out()))


def test_unseen_category_does_not_crash(fitted, customer):
    customer["PaymentMethod"] = "Crypto"
    with pytest.warns(UserWarning, match="unknown categories"):
        out = fitted.transform(pd.DataFrame([customer]))
    assert out.filter(like="PaymentMethod_").sum(axis=1).iloc[0] == 0


def test_missing_column_is_rejected(raw_df):
    with pytest.raises(ValueError, match="tenure"):
        build_preprocessor().fit(raw_df.drop(columns=["tenure"]))
