"""Build the frozen train/test split from the raw CSV.

Feature transformations do NOT happen here; they live in the sklearn pipeline
(churn.features.build) so training and serving share the exact same code.
This module only handles dataset-level concerns: the target and the split.

Usage:
    python -m churn.data.make_dataset
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import (
    ID_COLUMN,
    PROCESSED_DIR,
    RANDOM_STATE,
    TARGET,
    TEST_PATH,
    TEST_SIZE,
    TRAIN_PATH,
)
from churn.data.load import load_raw


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map Churn 'Yes'/'No' to 1/0."""
    out = df.copy()
    out[TARGET] = out[TARGET].map({"Yes": 1, "No": 0})
    if out[TARGET].isna().any():
        raise ValueError(f"Unexpected values in {TARGET}: {df[TARGET].unique().tolist()}")
    out[TARGET] = out[TARGET].astype(int)
    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target. customerID stays in X; the pipeline drops it."""
    return df.drop(columns=[TARGET]), df[TARGET]


def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split so both sets keep the ~26.5% churn rate."""
    return train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )


def main() -> None:
    df = encode_target(load_raw())
    assert df[ID_COLUMN].is_unique
    train, test = make_splits(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    for name, part in [("train", train), ("test", test)]:
        print(f"{name}: {len(part):>5} rows, churn rate {part[TARGET].mean():.1%}")
    print(f"written to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
