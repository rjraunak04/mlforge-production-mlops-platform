# Day 06 — Automated Retraining and CI/CD

## Objective

Close the production ML lifecycle without bypassing governance.

Monitoring evidence -> retraining decision -> governed candidate training -> validation quality gates -> model registry -> champion/challenger decision -> auditable evidence.

## Safety invariants

- Healthy monitoring evidence does not retrain.
- A monitoring signal starts candidate training; it never directly promotes a model.
- Day 03 validation quality gates remain mandatory.
- Champion replacement requires the existing strict-improvement promotion rule.
- CI does not require the private/raw churn CSV.
- CI validates lint, formatting, tests, source compilation, and the Docker build contract.
- Production deployment and scheduled execution remain Day 07/platform concerns.

## CI/CD

Pull requests and pushes to develop/main run Python quality gates. A separate GitHub Actions job builds the production inference Docker image on a Linux runner, avoiding dependence on local Docker Desktop.
