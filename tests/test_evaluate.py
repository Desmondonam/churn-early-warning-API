import json

import numpy as np
import pandas as pd
import pytest

from churn.models.evaluate import (
    best_f1_threshold,
    compute_metrics,
    min_cost_threshold,
    threshold_table,
)
from churn.models.train import METADATA_PATH


@pytest.fixture
def scored():
    """Positives score higher than negatives, with some overlap."""
    rng = np.random.default_rng(0)
    y = pd.Series([0] * 300 + [1] * 100)
    proba = np.clip(np.r_[rng.normal(0.3, 0.15, 300), rng.normal(0.6, 0.15, 100)], 0, 1)
    return y, proba


def test_threshold_table_is_monotonic(scored):
    table = threshold_table(*scored)
    assert table["recall"].is_monotonic_decreasing
    assert table["flagged_rate"].is_monotonic_decreasing


def test_best_f1_threshold_matches_table_max(scored):
    table = threshold_table(*scored)
    t = best_f1_threshold(table)
    assert table.set_index("threshold").loc[t, "f1"] == table["f1"].max()


def test_expensive_misses_lower_the_threshold(scored):
    table = threshold_table(*scored)
    cheap_miss = min_cost_threshold(table, cost_fn=1, cost_fp=10)
    costly_miss = min_cost_threshold(table, cost_fn=10, cost_fp=1)
    assert costly_miss < cheap_miss


def test_compute_metrics_confusion_adds_up(scored):
    y, proba = scored
    m = compute_metrics(y, proba, 0.5)
    assert sum(m["confusion"].values()) == len(y)
    assert 0 < m["top_decile_churners_captured"] <= 1


@pytest.mark.skipif(not METADATA_PATH.exists(), reason="run train + evaluate first")
def test_metadata_has_what_the_api_needs():
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if "threshold" not in meta:
        pytest.skip("run `python -m churn.models.evaluate` first")
    assert 0 < meta["threshold"] < 1
    assert meta["input_features"]
