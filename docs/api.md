# API Reference

The package is importable as `mlops_se489` after running `pip install -e .`.

## `mlops_se489.config`

Project-wide path constants and typed config dataclasses.

```python
from mlops_se489.config import (
    PROJECT_ROOT,
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    Config, TrainingConfig, DataConfig, DEFAULT_CONFIG,
)
```

Use these constants instead of hard-coded relative paths — they resolve against the repo root regardless of the current working directory.

## `mlops_se489.logging_config`

```python
from mlops_se489.logging_config import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)
```

## `mlops_se489.data`

| Function | Purpose |
|---|---|
| `load_raw(filename)` | Read CSV from `data/raw/` |
| `load_processed(filename)` | Read CSV from `data/processed/` |
| `save_processed(df, filename)` | Write CSV to `data/processed/` |
| `process_data(input_dir, output_dir)` | Raw → processed pipeline |

CLI: `python -m mlops_se489.data.make_dataset [--input PATH] [--output PATH]`

## `mlops_se489.features`

```python
from mlops_se489.features import build_features

df_features = build_features(df_processed)
```

## `mlops_se489.models`

### `BaseModel` (abstract)

Abstract interface with `fit`, `predict`, `save`, `load`. Extend this for any new estimator.

### `Model`

Reference implementation scaffold. Serializes via `joblib`.

```python
from pathlib import Path
from mlops_se489.models import Model

model = Model(config={"lr": 0.01})
# model.fit(X_train, y_train)
model.save(Path("models/model.joblib"))
reloaded = Model.load(Path("models/model.joblib"))
```

## `mlops_se489.evaluation`

```python
from mlops_se489.evaluation import classification_report, regression_report

metrics = classification_report(y_true, y_pred)
# -> {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
```

## `mlops_se489.visualization`

```python
from mlops_se489.visualization import plot_training_history, plot_confusion_matrix
```

## `mlops_se489.utils`

```python
from mlops_se489.utils import set_seed, save_json, load_json

set_seed(42)
```

## Training / Prediction CLIs

```bash
python -m mlops_se489.train_model --epochs 100 --batch-size 64
python -m mlops_se489.predict_model --model-path models/model.joblib --input data/processed/test.csv
```

## Configuration

Defaults live in `mlops_se489.config.DEFAULT_CONFIG`. Override via CLI flags on the training/prediction entrypoints.

---

**Retail Demand Forecasting** · Version see `mlops_se489.__version__`
