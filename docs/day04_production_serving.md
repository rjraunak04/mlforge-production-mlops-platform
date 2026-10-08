# Day 04 — Production Model Serving

## Objective

Serve the governed MLflow champion through a production-oriented FastAPI inference layer with strict request validation, registry-backed model loading, observability, and a container deployment contract.

## Serving flow

Client -> FastAPI -> Pydantic validation -> inference service -> pinned MLflow champion version -> churn probability and decision.

The API never trains models and clients cannot select arbitrary model versions. The Day 03 registry alias remains the source of truth for the production champion.

## Endpoints

- `GET /health`: process liveness.
- `GET /ready`: model readiness plus pinned model name/version.
- `POST /predict`: validated churn inference response.
- `/docs` and `/openapi.json`: generated API contract.

## Safety and observability

Requests receive an `X-Request-ID` correlation identifier. Structured request logs record method, path, status, request ID, and latency without logging raw customer feature payloads. Internal inference failures are converted to safe public errors.

## Container contract

The repository includes a Python 3.12 slim Dockerfile, non-root runtime user, Uvicorn startup command, Docker healthcheck, and a restricted build context through `.dockerignore`. MLflow databases and model artifacts are deliberately excluded from the image; a deployment environment must provide registry/artifact access at runtime.

The container contract is covered by automated tests. Local Docker runtime execution was not used for this milestone because the development machine could not allocate the required Docker Desktop storage. This is an environment constraint rather than an application test failure; image build/runtime validation remains a deployment-environment check.

## Verification evidence

Final CP-07 repository QA on the development environment:

- Ruff lint: passed.
- Ruff format check: passed across 54 files.
- Pytest: 90 passed.
- API integration coverage verifies health, readiness, prediction, and request-ID propagation.
- Container-contract tests verify the non-root runtime, healthcheck, Uvicorn binding, and exclusion of local runtime state.

Two non-blocking third-party deprecation warnings remain in FastAPI/Starlette test-client compatibility and MLflow/SQLAlchemy internals.

## Day 04 boundary

Monitoring, production drift detection, Evidently reports, and automated retraining triggers are intentionally deferred to Day 05.
