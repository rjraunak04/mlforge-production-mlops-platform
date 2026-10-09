FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFORGE_SERVING_CONFIG=configs/serving.yaml

WORKDIR /app

RUN groupadd --system mlforge && useradd --system --gid mlforge --create-home mlforge

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN python -m pip install --upgrade pip && \
    python -m pip install .

RUN mkdir -p /app/mlruns && chown -R mlforge:mlforge /app
USER mlforge

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "mlforge.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
