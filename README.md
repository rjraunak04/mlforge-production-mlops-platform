# MLForge — Production MLOps Platform

MLForge is an end-to-end production ML system that trains, validates, versions, serves, monitors, and safely retrains classification models. The reference use case is customer churn; the primary deliverable is the MLOps lifecycle and governance around the model.

## Production lifecycle

```text
Raw data
  -> schema validation + leakage-safe split
  -> reproducible feature pipeline
  -> candidate training + validation comparison
  -> MLflow experiment tracking
  -> quality gates
  -> model registry (champion / challenger)
  -> FastAPI inference service
  -> production monitoring + drift detection
  -> retraining decision
  -> governed candidate training
  -> promote or reject
```

A drift signal never promotes a model directly. Retraining reuses the same validation gates and strict champion-comparison policy used by the governed training workflow.

## Engineering highlights

- Leakage-safe train/validation/test lifecycle with a locked test boundary.
- Reproducible scikit-learn preprocessing and candidate-model evaluation.
- MLflow experiment tracking and model registry with champion/challenger aliases.
- Config-driven quality gates and strict-improvement promotion decisions.
- FastAPI serving with validated request schemas, health/readiness endpoints, request IDs, and structured operational logging.
- Non-root Docker image with a container health check.
- Reference/production feature drift, prediction drift, and delayed-label performance monitoring.
- Monitoring-triggered retraining with auditable promotion/rejection evidence.
- GitHub Actions gates for Ruff, tests, source compilation, and Linux Docker builds.

## Repository map

```text
configs/                 Runtime, training, monitoring, and governance configuration
src/mlforge/data/        Loading, schema validation, and splitting
src/mlforge/features/    Feature preparation and preprocessing
src/mlforge/training/    Candidate training and evaluation
src/mlforge/tracking/    MLflow experiment tracking
src/mlforge/registry/    Model registry operations
src/mlforge/governance/  Quality and promotion gates
src/mlforge/serving/     FastAPI production inference service
src/mlforge/monitoring/  Drift and performance monitoring
src/mlforge/retraining/  Retraining decision and governed orchestration
tests/                   Unit and integration coverage
docs/                    Architecture and day-by-day engineering decisions
.github/workflows/       CI and container-build gates
```

## Quick start

Requires Python 3.12.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ruff check src tests
ruff format --check src tests
pytest -q
```

The raw IBM Telco Customer Churn CSV is intentionally not committed. Place it at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Training, MLflow, serving, monitoring, and retraining behavior is controlled by the YAML files under `configs/`.

## Container validation

```bash
docker build -t mlforge:local .
```

The image starts the FastAPI service on port 8000. The current serving configuration resolves the MLflow `champion` alias from the configured registry. A deployable environment therefore needs registry/artifact state accessible to the container; local MLflow state is intentionally not baked into the image.

## CI quality gates

Pull requests and pushes to `develop` and `main` independently validate:

```text
Ruff lint -> Ruff format -> pytest -> compileall
Docker build on Linux
```

This keeps local workstation state out of the release decision.

## Current production boundaries

MLForge demonstrates the full governed lifecycle, but it does not pretend local development infrastructure is cloud production. The repository currently uses a local SQLite-backed MLflow configuration; production deployment should supply a remote tracking/registry backend and durable artifact storage. Prediction-event persistence is available as a monitoring contract but is not automatically wired into every API request. Drift diagnostics are deterministic project-native checks rather than an Evidently dashboard.

These boundaries are explicit so the project remains reproducible and technically defensible in interviews.

## Documentation

- [Architecture](docs/architecture.md)
- [Production runbook](docs/production_runbook.md)
- [Release checklist](docs/release_checklist.md)
- [Data contract](docs/data_contract.md)
- [Data provenance](docs/data_provenance.md)
- [Leakage audit](docs/leakage_audit.md)
- [Day 06 retraining and CI/CD](docs/day06_retraining_cicd.md)

## Release strategy

Feature work is integrated into `develop`. A release candidate is promoted to `main` only after the Python CI and Linux container-build checks are green. The final release tag marks the recruiter/demo-ready baseline.

## Author

Ankur Kumar Jaiswal
