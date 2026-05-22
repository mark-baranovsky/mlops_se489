# Retail Demand Forecasting

## Team Information

- **Team Name:** Stack Overflowers
- **Team Members:**
  - Ayan Ahmed — aahme147@depaul.edu
  - John Blaszczak — jblaszc3@depaul.edu
  - Mark Baranovsky — mbarano3@depaul.edu
- **Course:** SE 489 — Machine Learning Engineering for Production

---

## Project Overview

We built a machine learning pipeline that forecasts weekly product demand for a retail business. The dataset has over a million order records covering 2,160 products across 4 warehouses. The pipeline takes raw order data, cleans it, turns it into weekly summaries, builds time-series features, trains multiple models, picks the best one, and generates predictions for the following week.

**Key Objectives:**
- Build a clean, reproducible data pipeline from raw CSV to weekly predictions
- Compare multiple forecasting approaches and track all experiments with MLflow
- Make the pipeline easy to run and understand by anyone on the team

---

## Architecture Diagram

```
Raw CSV
   │
   ▼
01_bronze_ingestion       →  data/interim/bronze_*.parquet
   │
   ▼
02_silver_cleaning        →  data/interim/silver_*.parquet
   │
   ▼
03_gold_aggregation       →  data/processed/gold_weekly_*.parquet
   │
   ▼
04_feature_engineering    →  data/processed/gold_weekly_*_features.parquet
   │
   ▼
05_model_training         →  models/champion_model.pkl + MLflow
   │
   ▼
06_batch_prediction       →  data/processed/demand_predictions.parquet
```

---

## Phase Deliverables

- [PHASE1.md](PHASE1.md) — Project Design & Model Development
- [PHASE2.md](PHASE2.md) — Containerization & Monitoring
- [PHASE3.md](PHASE3.md) — CI/CD & Deployment

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- Git

### Installation

```bash
git clone https://github.com/aahme147/mlops_se489.git
cd mlops_se489
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
pre-commit install
```

### Get the Data

Download from Kaggle:
[Historical Product Demand](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting)

Place the CSV file at:
```
data/raw/Historical Product Demand.csv
```

### Running the Pipeline

```bash
python notebooks/00_setup.py
python notebooks/01_bronze_ingestion_product_demand.py
python notebooks/02_silver_cleaning_product_demand.py
python notebooks/03_gold_weekly_aggregation_product_demand.py
python notebooks/04_gold_feature_engineering_product_demand.py
python notebooks/05_model_training_product_demand.py
python notebooks/06_batch_prediction_product_demand.py
```

### View MLflow Runs

```bash
mlflow ui
```

Open http://127.0.0.1:5000 in your browser.

### Run Tests

```bash
make test
```

Or:

```bash
pytest tests/ -v
```

### Run Linting

```bash
make lint
```

### Docker

#### Install Docker
- Install Docker Desktop: https://docs.docker.com/get-docker/
- Start Docker Desktop before building or running images.

#### Build the image
From the repository root:

```bash
docker build -f dockerfiles/Dockerfile -t mlops_se489 .
```

#### Run the container
Mount the model artifact directory so trained models persist outside the container:

```bash
docker run -it --rm -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data mlops_se489
```

On Windows PowerShell, use:

```powershell
docker run -it --rm -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data mlops_se489
```

Or simply use docker-compose

```powershell
docker compose up
```

The container entrypoint runs the training module by default. If you want to run a specific command inside the image, use:

```bash
docker run -it --rm -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data mlops_se489 python -m mlops_se489.models.train_model
```


## Technology Stack

### Core Dependencies
- **pandas** >= 2.2.0 — Data manipulation and pipeline
- **numpy** >= 1.26.0 — Numerical computing
- **scikit-learn** >= 1.5.0 — Random Forest and Gradient Boosting models
- **prophet** — Time-series forecasting (third-party requirement)
- **mlflow** >= 2.16.0 — Experiment tracking and model logging
- **joblib** >= 1.4.0 — Model persistence

### Development Tools
- **pytest** >= 8.0 — Testing framework
- **ruff** >= 0.6.0 — Linting and formatting
- **mypy** >= 1.11 — Static type checking
- **pre-commit** >= 3.8 — Git hooks

---

## Project Structure

```
mlops_se489/
├── notebooks/               # Pipeline scripts run in order
│   ├── 00_setup.py
│   ├── 01_bronze_ingestion_product_demand.py
│   ├── 02_silver_cleaning_product_demand.py
│   ├── 03_gold_weekly_aggregation_product_demand.py
│   ├── 04_gold_feature_engineering_product_demand.py
│   ├── 05_model_training_product_demand.py
│   └── 06_batch_prediction_product_demand.py
├── data/
│   ├── raw/                 # Original CSV — never modified
│   ├── interim/             # Bronze and Silver parquet files
│   └── processed/           # Gold, features, predictions
├── models/                  # Trained model artifacts
├── tests/                   # Pytest test suite
├── src/mlops_se489/         # Reusable Python package
├── PHASE1.md
├── PHASE2.md
├── PHASE3.md
├── CONTRIBUTING.md
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Model Results

| Model | Val RMSE | Val MAE |
|---|---|---|
| Baseline (mean lag) | 50,753 | 9,277 |
| Random Forest | 42,479 | 8,578 |
| Gradient Boosting | 42,899 | 8,106 |
| Prophet (top product) | 516 | 326 |

**Champion model:** Random Forest — val RMSE 42,479

---

## Contribution Summary

- **Ayan Ahmed** — Pipeline architecture, data ingestion, cleaning, MLflow setup, GitHub
- **John Blaszczak** — Feature engineering, model training, Prophet integration, batch prediction
- **Mark Baranovsky** — Documentation, README, PHASE1.md, data quality checks

---

## References

- Dataset: [Historical Product Demand on Kaggle](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting)
- Project structure: [Cookiecutter MLOps SE489](https://github.com/Alizadeh-DePaul/cookiecutter-mlops-se489)
- Third-party package: [Prophet by Meta](https://facebook.github.io/prophet/)
- [PHASE1.md](PHASE1.md) — Phase 1 deliverables
- [PHASE2.md](PHASE2.md) — Phase 2 deliverables
- [PHASE3.md](PHASE3.md) — Phase 3 deliverables

## License

MIT
