"""FastAPI service exposing churn risk scores.

Run locally:
    uvicorn churn.api.main:app --reload
Docs at http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request

from churn.api.schemas import (
    BatchRequest,
    BatchResponse,
    Customer,
    Health,
    ModelInfo,
    Prediction,
)
from churn.models.predict import ChurnPredictor

logger = logging.getLogger("uvicorn.error")  # shows up in uvicorn's log output


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast at startup if artifacts are missing, rather than on the first request.
    app.state.predictor = ChurnPredictor.load()
    meta = app.state.predictor.metadata
    logger.info("loaded %s (trained %s), threshold %.2f",
                meta["model"], meta["trained_at"], app.state.predictor.threshold)  # fmt: skip
    yield


app = FastAPI(
    title="Churn Early-Warning API",
    version="0.1.0",
    description="Scores telecom customers for churn risk.",
    lifespan=lifespan,
)


def _score(predictor: ChurnPredictor, customers: list[Customer]) -> list[Prediction]:
    records = pd.DataFrame([c.model_dump() for c in customers])
    probabilities = predictor.predict_proba(records)
    return [
        Prediction(
            customerID=c.customerID,
            churn_probability=round(float(p), 4),
            will_churn=bool(p >= predictor.threshold),
            risk_band=predictor.risk_band(float(p)),
        )
        for c, p in zip(customers, probabilities, strict=True)
    ]


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    return Health(status="ok", model_loaded=hasattr(request.app.state, "predictor"))


@app.get("/model", response_model=ModelInfo)
def model_info(request: Request) -> ModelInfo:
    meta = request.app.state.predictor.metadata
    return ModelInfo(
        model=meta["model"],
        trained_at=meta["trained_at"],
        threshold=meta["threshold"],
        threshold_strategy=meta["threshold_strategy"],
        test_metrics={k: meta["test_metrics"][k]
                      for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")},
    )  # fmt: skip


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request) -> Prediction:
    return _score(request.app.state.predictor, [customer])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    predictor = request.app.state.predictor
    predictions = _score(predictor, batch.customers)
    return BatchResponse(
        threshold=predictor.threshold,
        n_customers=len(predictions),
        n_flagged=sum(p.will_churn for p in predictions),
        predictions=predictions,
    )
