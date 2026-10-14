# Churn Early-Warning API

Flag customers who are likely to churn early enough to act on it, and serve those risk
scores through a REST API.

Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn):
7,043 customers, 20 features, binary target `Churn`.

## Project structure

```
churn_early_warning_api/
├── data/
│   ├── raw/            # original, immutable data (telco_churn.csv)
│   ├── interim/        # cleaned intermediate data (git-ignored)
│   └── processed/      # model-ready datasets (git-ignored)
├── notebooks/          # EDA and experiments, numbered: 01_eda.ipynb, 02_...
├── src/churn/
│   ├── config.py       # paths and constants shared by training and serving
│   ├── data/           # loading and validating raw data
│   ├── features/       # preprocessing / feature pipeline
│   ├── models/         # training, evaluation, inference
│   └── api/            # FastAPI app and request/response schemas
├── models/             # trained model artifacts (git-ignored)
├── reports/figures/    # generated plots and evaluation reports
├── tests/              # pytest suite
└── pyproject.toml      # dependencies and tool config
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell (macOS/Linux: source .venv/bin/activate)
python -m pip install --upgrade pip
pip install -e ".[dev]"             # runtime + dev dependencies, src/churn installed as editable
pytest                              # run tests
```

For production/Docker, install runtime dependencies only: `pip install .`

## Pipeline

```powershell
python -m churn.data.make_dataset    # frozen stratified train/test split -> data/processed/
python -m churn.models.train         # CV model comparison -> models/churn_model.joblib + metadata.json
python -m churn.models.evaluate      # pick threshold (OOF), test once -> reports/, metadata.json
# cost-based threshold instead of max-F1:
python -m churn.models.evaluate --cost-fn 500 --cost-fp 50
```

## API

```powershell
uvicorn churn.api.main:app --reload  # interactive docs at http://127.0.0.1:8000/docs
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness check |
| GET | `/model` | model name, training time, threshold, test metrics |
| POST | `/predict` | score one customer |
| POST | `/predict/batch` | score up to 1,000 customers |

Response: `churn_probability` (calibrated), `will_churn` (probability ≥ threshold) and
`risk_band`: `low` (< threshold), `medium` (threshold to 0.5), `high` (≥ 0.5).

The model directory can be overridden with the `CHURN_MODEL_DIR` environment variable.

## Docker

The image **trains the model during the build** (split → train → evaluate), so no model
artifacts live in git and every image ships a model trained by the exact library versions it runs.

```powershell
docker build -t churn-api .
docker run --rm -p 8000:8000 churn-api     # http://127.0.0.1:8000/docs
```

## Deployment (Render)

Defined in [render.yaml](render.yaml): Docker web service, free plan, Frankfurt region,
health check on `/health`.

- **First-time setup:** Render dashboard → **New → Blueprint** → select this GitHub repo → **Apply**.
- **Afterwards:** every push to `main` runs [CI](.github/workflows/ci.yml) (lint, tests, Docker
  build + smoke test). Render deploys only after the checks pass (`autoDeployTrigger: checksPass`).
- **Free plan:** the service sleeps after ~15 minutes idle; the first request after that takes ~30-60 s.

## Roadmap

1. **EDA**: understand the data, churn rate, and the main drivers
2. **Data & features**: cleaning (e.g. `TotalCharges` blanks), encoding, reusable pipeline
3. **Modeling**: baseline → stronger models; pick metrics for imbalanced data (recall, PR-AUC)
4. **Evaluation**: threshold selection tied to the retention use case
5. **API**: FastAPI `/predict` and `/health` endpoints
6. **Packaging & deployment**: Dockerfile, CI, cloud deployment
