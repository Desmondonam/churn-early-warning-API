"""Request/response models. Field names match the raw dataset columns."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]
MAX_BATCH_SIZE = 1000

EXAMPLE_CUSTOMER = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_CUSTOMER})

    customerID: str | None = Field(None, max_length=64, description="Echoed back in the response")
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=1200, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(
        None, ge=0, description="Omit or null for customers not yet billed (treated as 0)"
    )


class Prediction(BaseModel):
    customerID: str | None
    churn_probability: float = Field(description="Calibrated probability of churn")
    will_churn: bool = Field(description="churn_probability >= decision threshold")
    risk_band: Literal["low", "medium", "high"]


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    threshold: float
    n_customers: int
    n_flagged: int
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model: str
    trained_at: str
    threshold: float
    threshold_strategy: dict
    test_metrics: dict


class Health(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
