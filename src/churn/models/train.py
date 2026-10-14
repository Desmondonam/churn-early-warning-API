"""Compare candidate models with cross-validation on the training set and save the winner.

The test set is never touched here; it is reserved for final evaluation.

Usage:
    python -m churn.models.train
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import joblib
import pandas as pd
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from churn.config import MODELS_DIR, RANDOM_STATE, TRAIN_PATH
from churn.data.make_dataset import split_xy
from churn.features.build import RAW_FEATURES, build_preprocessor

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

PRIMARY_METRIC = "average_precision"  # PR-AUC: suited to the 26.5% minority class
SCORING = ["average_precision", "roc_auc", "recall", "precision", "f1"]
CV_FOLDS = 5


@dataclass
class Candidate:
    name: str
    estimator: ClassifierMixin
    grid: dict = field(default_factory=dict)


# Ordered from simplest to most complex; used as the tie-breaker.
# No class_weight: it inflates predicted probabilities (the API returns calibrated
# risk scores) and imbalance is handled by tuning the decision threshold instead.
CANDIDATES = [
    Candidate(
        "logistic_regression",
        LogisticRegression(max_iter=2000),
        {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]},
    ),
    Candidate(
        "random_forest",
        RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
        ),
        {"clf__min_samples_leaf": [3, 5, 10], "clf__max_features": ["sqrt", 0.3]},
    ),
    Candidate(
        "hist_gradient_boosting",
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "clf__learning_rate": [0.03, 0.1],
            "clf__max_depth": [3, 5],
            "clf__max_iter": [100, 300],
        },
    ),
]


def make_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])


def tune(candidate: Candidate, X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        make_pipeline(candidate.estimator),
        candidate.grid,
        scoring=SCORING,
        refit=PRIMARY_METRIC,
        cv=cv,
        n_jobs=-1,
    )
    return search.fit(X, y)


def summarize(name: str, search: GridSearchCV) -> dict:
    """CV mean/std of every metric for the best parameter set."""
    i = search.best_index_
    res = search.cv_results_
    return {
        "model": name,
        "best_params": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
        **{f"{m}_mean": float(res[f"mean_test_{m}"][i]) for m in SCORING},
        **{f"{m}_std": float(res[f"std_test_{m}"][i]) for m in SCORING},
    }


def select(results: list[dict]) -> dict:
    """Simplest model whose primary score is within one std of the best."""
    best = max(results, key=lambda r: r[f"{PRIMARY_METRIC}_mean"])
    cutoff = best[f"{PRIMARY_METRIC}_mean"] - best[f"{PRIMARY_METRIC}_std"]
    return next(r for r in results if r[f"{PRIMARY_METRIC}_mean"] >= cutoff)


def main() -> None:
    X, y = split_xy(pd.read_csv(TRAIN_PATH))

    searches, results = {}, []
    for cand in CANDIDATES:
        print(f"tuning {cand.name} ...")
        searches[cand.name] = tune(cand, X, y)
        results.append(summarize(cand.name, searches[cand.name]))

    table = pd.DataFrame(results).set_index("model")
    cols = [f"{m}_mean" for m in SCORING] + [f"{PRIMARY_METRIC}_std"]
    print("\nCV results (best params per model):")
    print(table[cols].round(3).to_string())

    chosen = select(results)
    print(f"\nselected: {chosen['model']} {chosen['best_params']}")

    # GridSearchCV already refit the best params on the full training set.
    model = searches[chosen["model"]].best_estimator_
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model": chosen["model"],
        "params": chosen["best_params"],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_train": len(X),
        "train_churn_rate": float(y.mean()),
        "input_features": RAW_FEATURES,
        "selection_rule": f"simplest model within 1 std of best CV {PRIMARY_METRIC}",
        "cv_results": results,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"saved {MODEL_PATH.name} and {METADATA_PATH.name} to {MODELS_DIR}")


if __name__ == "__main__":
    main()
