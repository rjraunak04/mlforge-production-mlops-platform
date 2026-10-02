# Dataset Provenance

## Dataset
IBM Telco Customer Churn sample dataset.

## MLForge usage
The dataset is a reference binary-classification workload for exercising the
MLForge production lifecycle. The MLOps system, rather than churn analysis,
is the primary project deliverable.

## Snapshot
- Raw filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Expected rows: 7,043
- Expected columns: 21
- Target: `Churn`
- Identifier: `customerID`
- Local path: `data/raw/`
- Raw data committed to Git: no
- Local file size observed: 977,501 bytes
- SHA-256: `88BE4B93FBE0CC83421AF1C503794C97C342ECA914C1576DB7C276E61D61358A`

## Source
The snapshot is the commonly used IBM Telco Customer Churn sample distributed
through Kaggle under the Telco Customer Churn dataset. The exact local snapshot
is pinned by the SHA-256 above.

## Raw-data policy
Raw source data is immutable. Cleaning, type conversion, imputation, encoding,
feature engineering, and splitting must not overwrite the source file. Derived
artifacts belong outside `data/raw/`.
