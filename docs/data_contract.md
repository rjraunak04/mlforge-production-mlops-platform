# MLForge Day 1 Data Contract

The Day 1 contract protects the boundary between raw source data and later ML
stages.

## Dataset-level invariants
- Exactly 21 expected columns.
- `customerID` is required, non-null, and unique.
- `Churn` is required and contains only `Yes` or `No`.
- Duplicate full rows are rejected.
- Raw input is never modified in place.

## Required columns
`customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`,
`tenure`, `PhoneService`, `MultipleLines`, `InternetService`,
`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,
`StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`,
`PaymentMethod`, `MonthlyCharges`, `TotalCharges`, and `Churn`.

## Type and domain rules
- `SeniorCitizen`: integer in {0, 1}.
- `tenure`: numeric and non-negative.
- `MonthlyCharges`: numeric and non-negative.
- `TotalCharges`: source CSV may contain blank/whitespace values. The validated
  loader converts it to numeric; blank values become missing and are retained
  for later train-only preprocessing rather than silently imputed on Day 1.
- Binary text fields use their documented source categories.
- Service fields retain source distinctions such as `No internet service` and
  `No phone service`; these are not collapsed during ingestion.

## Identifier policy
`customerID` is retained for traceability and split-integrity checks but must
not be used as a model feature.

## Target policy
`Churn` remains a human-readable Yes/No target on Day 1. Target encoding is a
training concern and is deferred.

## Split contract
The validated dataset is split 70/15/15 into train/validation/test using
stratification and random seed 42. The test split is locked for final evaluation.
No preprocessing estimator may be fitted before the split.
