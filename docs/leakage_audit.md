# Leakage Audit — Day 1

## Controls
1. Validate raw records before model development.
2. Exclude `customerID` from model features while retaining it for traceability.
3. Split before fitting imputers, encoders, scalers, feature selectors, or models.
4. Use stratified 70/15/15 train/validation/test partitions with a deterministic
   seed.
5. Keep the test partition locked from model selection and threshold decisions.
6. Assert customer identifiers do not overlap between partitions.

## Prediction-time assumption
The current candidate predictors are treated as customer/account attributes
available at the intended prediction snapshot. Day 1 does not claim causal
effects. Any later feature that is generated after churn is known must be
rejected as leakage.

## Day 1 boundary
No model training, feature-selection fitting, threshold optimization, or test-set
performance analysis is performed on Day 1.
