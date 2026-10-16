# syntax=docker/dockerfile:1
#
# Streamlit front end. Thin client over the API: no scikit-learn, no model.
# Build from the repo root:
#   docker build -f app/Dockerfile -t churn-app .
#   docker run --rm -p 8501:8501 -e API_URL=http://host.docker.internal:8000 churn-app

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    PORT=8501 \
    API_URL=http://127.0.0.1:8000
RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['optional-dependencies']['app']))" > requirements.txt \
 && pip install -r requirements.txt

# Only the HTTP client and the request schema it validates against.
COPY src/churn/__init__.py src/churn/client.py ./src/churn/
COPY src/churn/api/__init__.py src/churn/api/schemas.py ./src/churn/api/
COPY app/streamlit_app.py app/sample_customers.csv ./app/

USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health', timeout=4)" || exit 1
CMD ["sh", "-c", "exec streamlit run app/streamlit_app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false"]
