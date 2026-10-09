# Day 03 - Experiment Governance

## Objective
Day 03 adds MLflow experiment tracking, validation quality gates, governed model registration, and champion/challenger promotion. The locked test partition remains outside model selection and promotion decisions.

## Workflow
Training candidates are evaluated on validation data only. Parameters, validation metrics, metadata, and model artifacts are tracked in MLflow. The validation winner must pass configured quality gates before registration. An approved candidate becomes the initial champion when no champion exists; later candidates must strictly outperform the champion on the comparison metric.

## Quality gates
- ROC-AUC >= 0.80
- Average precision >= 0.55
- Recall >= 0.50

A failed gate stops the workflow before model registration.

## Real-data validation
The governed workflow was validated on the 7,043-row IBM Telco Customer Churn snapshot used by MLForge. Logistic regression, random forest, and histogram gradient boosting were tracked. Logistic regression won validation comparison with ROC-AUC 0.8453125 and passed the configured gates.

The approved model was registered as version 1 of MLForgeChurnClassifier. Because no champion existed, version 1 was promoted as the initial champion. Runtime MLflow run IDs, artifacts, and local registry database state are intentionally not committed.

## Reproducibility and artifact safety
Development tracking and registry state use local SQLite plus a local artifact root. mlruns/ and local database files are ignored by Git.

Model artifacts currently use CloudPickle because the tested MLflow SKOPS loading path rejected a required NumPy dtype in this sklearn pipeline. CloudPickle can execute code during deserialization, so artifacts must only be loaded from trusted MLForge-controlled storage.

Registry governance uses champion and challenger aliases instead of deprecated MLflow stages.

## Verification
Day 03 closure evidence:
- Ruff lint passed.
- Ruff format check passed.
- Pytest: 59 passed.
- End-to-end tracking, gating, registration, and promotion integration passed.
- Real-data governed run passed.
- The observed SQLAlchemy deprecation warning originates inside MLflow and is non-blocking.

## Boundary
Serving and containerization are Day 04 concerns. Monitoring, drift detection, and automated retraining remain later lifecycle stages.
