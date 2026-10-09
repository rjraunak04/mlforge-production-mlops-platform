# Engineering Decisions

This document records the main design choices behind MLForge and the trade-offs they introduce. It is intentionally short: the goal is to make the system's reasoning inspectable without turning the repository into a framework.

## 1. Treat the test set as locked

Model families are compared on validation data. The test boundary is not used to choose the winner.

**Why:** repeatedly consulting test performance during model selection converts the test set into another validation set and weakens the final estimate.

**Trade-off:** less feedback during iteration, but a more defensible evaluation boundary.

## 2. Use a simple model when it wins

The candidate benchmark includes Logistic Regression, Random Forest and HistGradientBoosting. The governed run selected Logistic Regression on validation ROC-AUC.

**Why:** production selection should follow the declared objective, not model complexity. A simpler winning model is easier to reason about and operate.

**Trade-off:** the repository does not chase marginal benchmark gains through a large hyperparameter search.

## 3. Separate eligibility from promotion

Minimum quality gates determine whether a candidate is eligible for registry evaluation. Promotion is a second decision: the eligible candidate must strictly outperform the current champion.

**Why:** a model can be acceptable in isolation without being an improvement over what is already serving.

**Trade-off:** equal-scoring retraining runs are rejected rather than creating unnecessary production churn.

## 4. Resolve an alias, then pin a version

Serving resolves the logical champion alias to a concrete registered-model version before inference.

**Why:** operators can reason in terms of a champion role while an inference process works against an immutable version.

**Trade-off:** alias changes do not magically mutate an already loaded process; rollout/reload behavior remains an operational concern.

## 5. Monitoring recommends; governance decides

Feature drift, prediction drift or delayed-label degradation can recommend retraining. Monitoring code does not directly assign the champion alias.

**Why:** distribution change is not proof that a newly trained model is better.

**Trade-off:** recovery is more conservative, but every replacement follows the same validation policy.

## 6. Keep drift diagnostics deterministic

MLForge implements project-native numeric/categorical drift and prediction-shift checks rather than adding a dashboard framework only for appearance.

**Why:** the calculations and thresholds remain small, testable and easy to explain.

**Trade-off:** the project does not provide the richer visualization/report ecosystem of tools such as Evidently.

## 7. Fail readiness instead of hiding model failure

Liveness and readiness are separate. A process can be alive while the model dependency is not ready.

**Why:** silently falling back to an unknown model would hide an operational failure.

**Trade-off:** registry/artifact outages become visible service-readiness failures and need operational handling.

## 8. Do not bake local registry state into the image

The Docker image contains application code and configuration, not the developer's local MLflow database/artifacts.

**Why:** a container should not present workstation-specific state as portable production infrastructure.

**Trade-off:** a real deployment must provide reachable tracking/registry and artifact storage.

## 9. CI validates code and container portability independently

Pull requests and pushes to integration/release branches run Python quality checks and a Linux Docker build.

**Why:** a Windows-local success is not sufficient evidence that the release artifact builds in a clean Linux environment.

**Trade-off:** CI proves buildability, not that this portfolio release is currently deployed to a live cloud endpoint.
