# syntax=docker/dockerfile:1
#
# Three stages:
#   deps    - runtime dependencies in a venv (shared by train + runtime, so the model is
#             always loaded by the exact scikit-learn version that trained it)
#   train   - builds the split, trains, picks the threshold -> /app/models
#   runtime - slim image: venv + source + trained model, non-root, serves the API
#
# Build & run locally:
#   docker build -t churn-api .
#   docker run --rm -p 8000:8000 churn-api

ARG PYTHON_VERSION=3.12

# ---------- deps ----------
FROM python:${PYTHON_VERSION}-slim AS deps
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
# Only pyproject.toml is copied, so code changes don't invalidate the dependency layer.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > requirements.txt \
 && pip install -r requirements.txt

# ---------- train ----------
FROM deps AS train
RUN pip install "matplotlib>=3.9"   # evaluation plots only; not shipped in runtime
ENV PYTHONPATH=/app/src
COPY src ./src
COPY data/raw ./data/raw
RUN python -m churn.data.make_dataset \
 && python -m churn.models.train \
 && python -m churn.models.evaluate \
 && python -c "from churn.models.predict import ChurnPredictor; p = ChurnPredictor.load(); print('model OK, threshold', p.threshold)"

# ---------- runtime ----------
FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    CHURN_MODEL_DIR=/app/models \
    PORT=8000
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=deps /opt/venv /opt/venv
COPY --chown=app:app src ./src
COPY --from=train --chown=app:app /app/models ./models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)" || exit 1
# PORT is injected by the platform (Render sets it); defaults to 8000 locally.
CMD ["sh", "-c", "exec uvicorn churn.api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
