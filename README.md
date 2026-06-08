# Retail Demand Forecasting

## Team Information

* **Team Name:** Stack Overflowers
* **Team Members:**

  * Ayan Ahmed — [aahme147@depaul.edu](mailto:aahme147@depaul.edu)
  * John Blaszczak — [jblaszc3@depaul.edu](mailto:jblaszc3@depaul.edu)
  * Mark Baranovsky — [mbarano3@depaul.edu](mailto:mbarano3@depaul.edu)
* **Course:** SE 489 — Machine Learning Engineering for Production

---

## Project Overview

We built a machine learning pipeline that forecasts weekly product demand for a retail business. The dataset contains over one million order records covering 2,160 products across 4 warehouses.

The pipeline takes raw order data, cleans it, converts it into weekly summaries, builds time-series features, trains multiple forecasting models, selects a champion model, and generates next-week demand predictions.

**Key Objectives:**

* Build a clean and reproducible data pipeline from raw CSV data to weekly predictions
* Compare multiple forecasting approaches and track experiments with MLflow
* Package pipeline logic into reusable source modules under `src/mlops_se489`
* Add automated testing, linting, formatting, type checking, and CI/CD workflows
* Support Dockerized execution and deployment preparation

---

## Architecture Diagram

```text
Raw CSV
   │
   ▼
Bronze Ingestion              →  data/interim/bronze_product_demand_raw.parquet
   │
   ▼
Silver Cleaning               →  data/interim/silver_product_demand_clean.parquet
   │
   ▼
Gold Weekly Aggregation       →  data/processed/gold_weekly_product_demand.parquet
   │
   ▼
Feature Engineering           →  data/processed/gold_weekly_product_demand_features.parquet
   │
   ▼
Model Training + MLflow       →  models/champion_model.pkl
   │
   ▼
Batch Prediction              →  data/processed/demand_predictions.parquet
```

---

## Phase Deliverables

* [PHASE1.md](PHASE1.md) — Project Design & Model Development
* [PHASE2.md](PHASE2.md) — Containerization, Logging, Profiling & Monitoring
* [PHASE3.md](PHASE3.md) — CI/CD, CML & Deployment

---

## Phase 3: CI/CD and GCP Deployment

This phase adds continuous Docker image building, Artifact Registry integration, and GCP custom training workflows.

Key Phase 3 artifacts:

* `.github/workflows/docker-publish.yaml` — GitHub Actions workflow that builds and publishes the Docker image from `dockerfiles/Dockerfile`.
* `cloudbuild.yaml` — Cloud Build configuration for pushing the same image to GCP Artifact Registry.
* `config_cpu.yaml` — Vertex AI / Agent Platform custom training specification.
* `PHASE3.md` — detailed Phase 3 documentation with file references, evidence checklist, and deployment notes.


---

## Setup Instructions

### Prerequisites

* Python 3.11+
* Git
* Docker Desktop, if running the container workflow

### Installation

```bash
git clone https://github.com/aahme147/mlops_se489.git
cd mlops_se489
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
pre-commit install
```

---

## Get the Data

Download the dataset from Kaggle:

[Historical Product Demand](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting)

Place the CSV file at:

```text
data/raw/Historical Product Demand.csv
```

---

## Running the Pipeline

The preferred pipeline commands use the `Makefile`.

```bash
make data      # ingest and clean raw CSV
make features  # build weekly features
make train     # train the champion model
make predict   # generate next-week demand predictions
```

Equivalent Python module commands:

```bash
python -m mlops_se489.data.make_dataset
python -m mlops_se489.features.build_features
python -m mlops_se489.models.train_model
python -m mlops_se489.models.predict_model
```

Pipeline outputs include:

```text
- data/interim/bronze_product_demand_raw.parquet` — bronze ingestion output
- data/interim/silver_product_demand_clean.parquet` — cleaned silver dataset
- data/processed/gold_weekly_product_demand.parquet` — aggregated weekly demand
- data/processed/gold_weekly_product_demand_features.parquet` — engineered features dataset
- models/champion_model.pkl` — trained champion model
- data/processed/demand_predictions.parquet` — batch demand predictions
```

---

## View MLflow Runs

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

---

## Testing and Code Quality

### Run Unit and Integration Tests

```bash
python -m pytest tests/ -v
```

Run tests with coverage:

```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### Run Ruff Linting

```bash
python -m ruff check src scripts tests
```

### Run Ruff Formatting Check

```bash
python -m ruff format --check src scripts tests
```

### Run Mypy Type Checking

```bash
python -m mypy src/mlops_se489 --ignore-missing-imports
```

### Run Pre-commit Hooks

```bash
pre-commit run --all-files
```

The project includes scoped pre-commit hooks for:

* Ruff linting
* Ruff formatting
* Mypy type checking
* Trailing whitespace cleanup
* End-of-file fixes
* YAML checks
* Large-file checks

---

## CI/CD Workflows

This project uses GitHub Actions for automated testing and code quality checks.

### Test Workflow

File:

```text
.github/workflows/tests.yml
```

Purpose:

* Runs the pytest test suite
* Reports coverage metrics
* Runs on pull requests and pushes
* Tests across Python 3.11 and Python 3.12

### Code Quality Workflow

File:

```text
.github/workflows/code-quality.yml
```

Purpose:

* Runs Ruff linting
* Runs Ruff formatting checks
* Runs mypy type checking
* Checks `src`, `scripts`, and `tests`

### Legacy CI Workflow

File:

```text
.github/workflows/ci.yml
```

Purpose:

* Runs combined linting, formatting, type checking, and tests
* Uses Python 3.11 and Python 3.12 matrix testing
* Avoids linting notebook exports by limiting checks to source code, scripts, and tests

### Docker Build Workflow

File:

```text
.github/workflows/docker-publish.yaml
```

Purpose:

* Builds the Docker image on PRs and branch updates
* Validates that the Docker image can be built successfully

---

## Docker

### Install Docker

Install Docker Desktop:

[Docker Installation Guide](https://docs.docker.com/get-docker/)

Start Docker Desktop before building or running images.

### Build the Image

From the repository root:

```bash
docker build -f dockerfiles/Dockerfile -t mlops_se489 .
```

### Run the Container

Mount the model and data directories so artifacts persist outside the container:

```bash
docker run -it --rm \
  -v ${PWD}/models:/app/models \
  -v ${PWD}/data:/app/data \
  mlops_se489
```

On Windows PowerShell:

```powershell
docker run -it --rm -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data mlops_se489
```

### Docker Compose

```bash
docker compose up
```

### Run a Specific Module Inside the Container

```bash
docker run -it --rm \
  -v ${PWD}/models:/app/models \
  -v ${PWD}/data:/app/data \
  mlops_se489 \
  python -m mlops_se489.models.train_model
```

---

## Profiling

A standalone profiling script is included for CPU and memory profiling.

```bash
python scripts/profile_training.py
```

Generated profiling outputs:

```text
reports/profiling/training_cpu_profile.prof
reports/profiling/training_cpu_profile.txt
reports/profiling/training_memory_usage.txt
```

---

## Technology Stack

### Core Dependencies

* **pandas** >= 2.2.0 — Data manipulation and pipeline processing
* **numpy** >= 1.26.0 — Numerical computing
* **scikit-learn** >= 1.5.0 — Random Forest and Gradient Boosting models
* **prophet** >= 1.1.5 — Time-series forecasting
* **mlflow** >= 2.16.0 — Experiment tracking and model logging
* **joblib** >= 1.4.0 — Model persistence
* **hydra-core** >= 1.3.2 — Configuration management
* **rich** >= 13.7.0 — Improved logging output
* **memory_profiler** >= 0.61.0 — Memory profiling

### Development Tools

* **pytest** >= 8.0 — Testing framework
* **pytest-cov** >= 5.0 — Test coverage reporting
* **ruff** >= 0.6.0 — Linting and formatting
* **mypy** >= 1.11 — Static type checking
* **pre-commit** >= 3.8 — Git hooks

---

## Project Structure

```text
mlops_se489/
├── .github/workflows/        # GitHub Actions workflows
│   ├── ci.yml
│   ├── code-quality.yml
│   ├── docker-publish.yaml
│   └── tests.yml
├── data/
│   ├── raw/                  # Original CSV
│   ├── interim/              # Bronze and Silver parquet files
│   └── processed/            # Gold, features, predictions
├── dockerfiles/
│   └── Dockerfile
├── docs/                     # Additional phase documentation
├── models/                   # Trained model artifacts
├── notebooks/                # Original notebook/script workflow
├── reports/
│   └── profiling/            # Profiling outputs
├── scripts/
│   └── profile_training.py
├── src/mlops_se489/          # Reusable Python package
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── utils/
│   ├── visualization/
│   ├── config.py
│   └── logging_config.py
├── tests/                    # Unit and integration tests
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
├── Makefile
├── PHASE1.md
├── PHASE2.md
├── PHASE3.md
├── pyproject.toml
└── requirements.txt
```

---

## Model Results

| Model                    | Val RMSE | Val MAE |
| ------------------------ | -------: | ------: |
| Baseline mean-lag model  |   50,753 |   9,277 |
| Random Forest            |   42,479 |   8,578 |
| Gradient Boosting        |   42,899 |   8,106 |
| Prophet top-product demo |      516 |     326 |

**Champion model:** Random Forest — validation RMSE 42,479

---

## Contribution Summary

* **Ayan Ahmed** — Source pipeline fixes, Hydra/logging cleanup, MLflow pipeline stability, GitHub Actions CI/CD testing, Ruff/mypy checks, pre-commit hooks, integration tests
* **John Blaszczak** — Feature engineering, model training, profiling cleanup, CML model training/reporting, Prophet integration, batch prediction
* **Mark Baranovsky** — Docker automation, deployment setup, README/phase documentation, final documentation, cleanup and deployment guidance

---

## References

* Dataset: [Historical Product Demand on Kaggle](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting)
* Project structure: [Cookiecutter MLOps SE489](https://github.com/Alizadeh-DePaul/cookiecutter-mlops-se489)
* Third-party package: [Prophet by Meta](https://facebook.github.io/prophet/)
* [PHASE1.md](PHASE1.md) — Phase 1 deliverables
* [PHASE2.md](PHASE2.md) — Phase 2 deliverables
* [PHASE3.md](PHASE3.md) — Phase 3 deliverables

---

## To Check Online HF Space 
* Link:https://johnblaszczak419-retail-demand-forecasting.hf.space/

##Video Demo
* https://drive.google.com/file/d/1vxB6As5pQCj1G5Ma7IzszZvb4vlgxI0g/view?usp=sharing


## License

MIT
