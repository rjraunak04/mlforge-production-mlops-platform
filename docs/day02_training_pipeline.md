# Day 2 — Reproducible Training Pipeline

Day 2 connects the validated Telco churn dataset to a leakage-safe,
configuration-driven model training and validation workflow.

## Pipeline

1. Load and validate the raw dataset.
2. Create deterministic stratified train/validation/test partitions.
3. Exclude customer identifiers and encode the binary target.
4. Fit preprocessing inside each model pipeline using training data only.
5. Train Logistic Regression, Random Forest, and Histogram Gradient Boosting.
6. Evaluate all candidates on validation data.
7. Rank models by validation ROC-AUC.
8. Keep the test partition locked; it is not used for model selection.

## Verified reference run

The local reference dataset contains 7,043 rows. The deterministic split produced:

- Train: 4,930 rows
- Validation: 1,056 rows
- Locked test: 1,057 rows

Validation results at classification threshold 0.50:

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8453 | 0.6310 | 0.6182 | 0.6459 | 0.5929 | 0.8059 |
| Histogram Gradient Boosting | 0.8188 | 0.6048 | 0.5427 | 0.5789 | 0.5107 | 0.7718 |
| Random Forest | 0.8174 | 0.5732 | 0.5993 | 0.5509 | 0.6571 | 0.7670 |

**Selected model:** Logistic Regression, based only on validation ROC-AUC.

These values are reproducibility evidence for the configured reference run, not
a claim of final test-set performance. The locked test set remains untouched for
model selection.

## Quality gates

The Day 2 closure was verified locally with:

- Ruff lint checks passing.
- Ruff formatting checks passing.
- 34 automated tests passing.
- End-to-end execution on the 7,043-row reference dataset.
- Mutually disjoint train, validation, and locked-test customer identifiers.
- Regression coverage for YAML target-label semantics.
