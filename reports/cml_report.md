# CML Model Training Report

## Pipeline Status

- Data pipeline: completed locally
- Feature pipeline: completed locally
- Model training: completed locally
- Profiling: completed locally

## Model Metrics

| Model | Validation RMSE | Validation MAE |
|---|---:|---:|
| Baseline | 50,753.3 | 9,277.3 |
| Random Forest | 42,479.0 | 8,577.7 |
| Gradient Boosting | 42,875.5 | 8,105.6 |
| Prophet demo model | 515.8 | 326.1 |

## Champion Model

The selected champion model is **Random Forest** with validation RMSE **42,479.0**.

## Profiling Summary

- CPU profiling completed successfully.
- Memory profiling completed successfully.
- Peak memory usage: **894.48 MiB**.
- Profiling outputs:
  - `reports/profiling/training_cpu_profile.txt`
  - `reports/profiling/training_memory_usage.txt`

This report is posted automatically by the CML workflow on pull requests.
