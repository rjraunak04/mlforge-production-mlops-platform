# Production Runbook

## 1. Quality validation

Before promotion or release, run:

```bash
ruff check src tests
ruff format --check src tests
pytest -q
python -m compileall -q src
docker build -t mlforge:local .
```

GitHub Actions performs the same Python quality gate and an independent Linux container build for pull requests and pushes to `develop` and `main`.

## 2. Model lifecycle

The governed training workflow compares candidate models on validation ROC-AUC, applies configured quality thresholds, registers an eligible candidate, and applies the strict champion/challenger promotion rule. Do not bypass this workflow by manually assigning the champion alias for a routine retraining event.

## 3. Serving

The FastAPI application reads `configs/serving.yaml`. It expects the configured MLflow registry to contain the configured model and alias. `/health` is liveness; `/ready` validates model readiness.

For container deployment, provide MLflow registry/artifact connectivity appropriate to the target environment. The repository's local SQLite configuration is for development and demonstration.

## 4. Monitoring

Monitoring configuration lives in `configs/monitoring.yaml`. A monitoring report combines feature drift, prediction drift, and optional delayed-label performance. The report emits a retraining recommendation rather than mutating registry state.

Prediction-event serialization exists as a monitoring contract. Automatic persistence from every live API request is a future integration boundary and should not be assumed.

## 5. Retraining

`configs/retraining.yaml` controls whether retraining is enabled and whether a monitoring signal is required. The orchestrator reuses the governed training workflow and writes evidence containing the decision, candidate version, quality-gate result, and promotion result.

## 6. Incident rules

- Registry unavailable: readiness should fail; do not silently serve an unverified fallback model.
- Candidate fails quality gate: reject it and retain the champion.
- Candidate does not strictly improve: retain the champion.
- Monitoring data below the configured minimum: do not treat it as sufficient production evidence.
- CI or container build fails: do not promote the release branch.
