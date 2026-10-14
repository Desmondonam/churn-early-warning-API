import pytest

from churn.config import TARGET, TEST_SIZE
from churn.data.make_dataset import encode_target, make_splits, split_xy


def test_raw_schema(raw_df):
    assert raw_df.shape == (7043, 21)
    assert raw_df["customerID"].is_unique


def test_encode_target(raw_df):
    df = encode_target(raw_df)
    assert set(df[TARGET].unique()) == {0, 1}
    assert df[TARGET].sum() == (raw_df[TARGET] == "Yes").sum()


def test_encode_target_rejects_unknown_labels(raw_df):
    bad = raw_df.head(3).copy()
    bad.loc[bad.index[0], TARGET] = "Maybe"
    with pytest.raises(ValueError):
        encode_target(bad)


def test_splits_are_stratified_and_disjoint(raw_df):
    df = encode_target(raw_df)
    train, test = make_splits(df)
    assert len(test) == pytest.approx(len(df) * TEST_SIZE, abs=1)
    assert set(train["customerID"]).isdisjoint(test["customerID"])
    assert train[TARGET].mean() == pytest.approx(test[TARGET].mean(), abs=0.01)


def test_split_xy(raw_df):
    X, y = split_xy(encode_target(raw_df))
    assert TARGET not in X.columns
    assert len(X) == len(y)
