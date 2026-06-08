# CML Model Training Report

## Pipeline Status

- Data pipeline: completed
- Feature pipeline: completed
- Model training: completed
- CML plots: completed
- Profiling: checked

## Model Metrics from MLflow

| Model | Train RMSE | Validation RMSE | Validation MAE |
| prophet | N/A | 515.8 | 326.1 |
 | hydra_random_forest | 42,411.6 | 42,138.5 | 8,596.6 |
 | random_forest | 42,278.4 | 42,479.0 | 8,577.7 |
 | gradient_boosting | 39,050.9 | 42,898.7 | 8,106.3 |
 | baseline_mean_lag | 54,470.3 | 50,753.3 | 9,277.3 |
 
## Champion Model

- Champion model: **random_forest**
- Champion validation RMSE: **42,479.0**
- Champion validation MAE: **8,577.7**

## Champion Model Re-Evaluation

The saved champion model was loaded from `models/champion_model.pkl` and re-evaluated on the validation set.

| Metric | Value |
| Validation RMSE | 42,479.0 |
| Validation MAE | 8,577.7 |

## Profiling Summary

- CPU profiling completed successfully if profiling records are present.
- Memory profiling checked.
- Peak memory usage: **903.52 MiB**.

Profiling outputs are saved in `reports/profiling/`.

## Plots

All plots are saved in `reports/figures/`.

### Actual vs Predicted

![Actual vs Predicted](reports\figures\actual_vs_predicted.png)

### Feature Importance

![Feature Importance](reports\figures\feature_importance.png)

### Residual Plot

![Residual Plot](reports\figures\residual_plot.png)

This report is posted automatically by the CML workflow on pull requests.
