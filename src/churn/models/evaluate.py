"""Choose the decision threshold and evaluate the saved model once on the test set.

1. Threshold: chosen on out-of-fold (OOF) predictions from the TRAINING set, so the
   test set plays no part in any decision.
2. Test evaluation: run once with the frozen model + threshold.
3. The threshold and test metrics are written into models/metadata.json for the API.

Usage:
    python -m churn.models.evaluate                      # maximize F1 (default)
    python -m churn.models.evaluate --cost-fn 500 --cost-fp 50
        # minimize expected cost: missed churner costs 500, unneeded retention offer 50
"""

import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from churn.config import RANDOM_STATE, REPORTS_DIR, TEST_PATH, TRAIN_PATH  # noqa: E402
from churn.data.make_dataset import split_xy  # noqa: E402
from churn.models.train import CV_FOLDS, METADATA_PATH, MODEL_PATH  # noqa: E402

FIG_DIR = REPORTS_DIR / "figures"
EVAL_PATH = REPORTS_DIR / "evaluation.json"
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def threshold_table(y: pd.Series, proba: np.ndarray, grid=THRESHOLD_GRID) -> pd.DataFrame:
    """Precision / recall / F1 / share of customers flagged at each candidate threshold."""
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
            "f1": f1_score(y, pred),
            "flagged_rate": pred.mean(),
            "fn": int(((pred == 0) & (y == 1)).sum()),
            "fp": int(((pred == 1) & (y == 0)).sum()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def best_f1_threshold(table: pd.DataFrame) -> float:
    return float(table.loc[table["f1"].idxmax(), "threshold"])


def min_cost_threshold(table: pd.DataFrame, cost_fn: float, cost_fp: float) -> float:
    cost = table["fn"] * cost_fn + table["fp"] * cost_fp
    return float(table.loc[cost.idxmin(), "threshold"])


def compute_metrics(y: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    # Retention-team view: if we only contact the riskiest 10%, how many churners do we reach?
    top = pd.Series(proba).rank(ascending=False, method="first") <= int(np.ceil(0.1 * len(y)))
    top_rate = y.to_numpy()[top.to_numpy()].mean()

    return {
        "threshold": threshold,
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "flagged_rate": pred.mean(),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "top_decile_churn_rate": top_rate,
        "top_decile_lift": top_rate / y.mean(),
        "top_decile_churners_captured": y.to_numpy()[top.to_numpy()].sum() / y.sum(),
    }


def plot_threshold_tradeoff(table: pd.DataFrame, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col in ["precision", "recall", "f1", "flagged_rate"]:
        ls = "--" if col == "flagged_rate" else "-"
        ax.plot(table["threshold"], table[col], label=col, ls=ls)
    ax.axvline(threshold, c="black", lw=1)
    ax.text(threshold + 0.01, 0.95, f"chosen = {threshold:.2f}", va="top")
    ax.set(xlabel="decision threshold", ylabel="score",
           title="Threshold trade-off (out-of-fold, training set)")  # fmt: skip
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_threshold_tradeoff.png", dpi=120)
    plt.close(fig)


def plot_test_results(y: pd.Series, proba: np.ndarray, threshold: float) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    PrecisionRecallDisplay.from_predictions(y, proba, ax=axes[0], plot_chance_level=True)
    axes[0].set_title("Precision-Recall (test)")
    RocCurveDisplay.from_predictions(y, proba, ax=axes[1], plot_chance_level=True)
    axes[1].set_title("ROC (test)")

    true, pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    axes[2].plot([0, 1], [0, 1], ls=":", c="grey", label="perfect")
    axes[2].plot(pred, true, marker="o", label="model")
    axes[2].set(xlabel="predicted churn probability", ylabel="observed churn rate",
                title="Calibration (test)")  # fmt: skip
    axes[2].legend()

    ConfusionMatrixDisplay.from_predictions(
        y, (proba >= threshold).astype(int), display_labels=["stay", "churn"],
        ax=axes[3], colorbar=False, cmap="Blues",
    )  # fmt: skip
    axes[3].set_title(f"Confusion matrix @ {threshold:.2f} (test)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_test_evaluation.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cost-fn", type=float, help="cost of missing a churner")
    parser.add_argument("--cost-fp", type=float, help="cost of an unneeded retention offer")
    args = parser.parse_args()
    if (args.cost_fn is None) != (args.cost_fp is None):
        parser.error("--cost-fn and --cost-fp must be given together")

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    X_train, y_train = split_xy(pd.read_csv(TRAIN_PATH))
    X_test, y_test = split_xy(pd.read_csv(TEST_PATH))

    # 1. Threshold from out-of-fold predictions on the training set
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    table = threshold_table(y_train, oof)
    if args.cost_fn is not None:
        threshold = min_cost_threshold(table, args.cost_fn, args.cost_fp)
        strategy = {"name": "min_expected_cost", "cost_fn": args.cost_fn, "cost_fp": args.cost_fp}
    else:
        threshold = best_f1_threshold(table)
        strategy = {"name": "max_f1"}
    oof_row = table.set_index("threshold").loc[threshold]
    print(f"threshold = {threshold:.2f} ({strategy['name']}); OOF precision "
          f"{oof_row.precision:.3f}, recall {oof_row.recall:.3f}, F1 {oof_row.f1:.3f}")

    # 2. One-shot test evaluation
    proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, proba, threshold)
    default = compute_metrics(y_test, proba, 0.5)

    print("\nTest set:")
    print(f"  PR-AUC {metrics['pr_auc']:.3f} | ROC-AUC {metrics['roc_auc']:.3f} "
          f"| Brier {metrics['brier']:.3f}")
    for name, m in [(f"@ {threshold:.2f}", metrics), ("@ 0.50", default)]:
        print(f"  {name}: precision {m['precision']:.3f}, recall {m['recall']:.3f}, "
              f"F1 {m['f1']:.3f}, flags {m['flagged_rate']:.1%} of customers, {m['confusion']}")
    print(f"  top 10% riskiest: churn rate {metrics['top_decile_churn_rate']:.1%} "
          f"(lift {metrics['top_decile_lift']:.1f}x), "
          f"captures {metrics['top_decile_churners_captured']:.1%} of churners")

    # 3. Persist
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_threshold_tradeoff(table, threshold)
    plot_test_results(y_test, proba, threshold)

    report = {
        "model": metadata["model"],
        "threshold_strategy": strategy,
        "threshold": threshold,
        "oof_at_threshold": oof_row.to_dict(),
        "test": metrics,
        "test_at_0.5": default,
        "threshold_table_oof": table.round(4).to_dict(orient="records"),
    }
    EVAL_PATH.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    metadata.update(threshold=threshold, threshold_strategy=strategy, test_metrics=metrics)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {EVAL_PATH.name}, updated {METADATA_PATH.name}, figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
