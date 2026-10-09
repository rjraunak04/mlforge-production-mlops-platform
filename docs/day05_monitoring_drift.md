# Day 05 — Production Monitoring and Drift Detection

## Objective

Extend MLForge beyond deployment by capturing production inference events, comparing live feature and prediction distributions against a training reference, evaluating delayed-label performance, and producing an auditable retraining recommendation.

## Monitoring flow

Production predictions -> JSONL event log -> monitoring batch -> reference comparison -> feature drift + prediction drift -> optional delayed-label performance -> monitoring report -> retraining recommendation.

## Checkpoints

1. Monitoring configuration and thresholds.
2. Privacy-conscious production prediction event contract.
3. Reference dataset contract aligned with training features.
4. Feature drift diagnostics for numeric and categorical inputs.
5. Prediction probability drift.
6. Delayed-label ROC-AUC, accuracy, and recall degradation checks.
7. Serializable monitoring report and retraining recommendation.

## Governance boundary

A retraining recommendation is a signal, not automatic promotion. Day 03 quality gates and champion/challenger governance remain mandatory before any replacement model can become champion.

Monitoring logs must not contain customer identifiers or secrets. The current event contract stores model inputs needed for drift analysis; deployments handling sensitive fields should apply organization-specific minimization and retention policies.

## Day 06 boundary

Automated retraining orchestration, CI/CD, scheduled monitoring execution, and promotion automation are intentionally deferred to Day 06.
