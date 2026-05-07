# Retail Demand Forecasting

## 1\. Team Information

* **Team Name:** Stack Overflowers
* **Team Members:**

  * Ayan Ahmed — aahme147@depaul.edu
  * John Blaszczak — jblaszc3@depaul.edu
  * Mark Baranovsky — mbarano3@depaul.edu
* **Course:** SE 489

\---

## 2\. Project Overview

We built a machine learning pipeline that forecasts weekly product demand for a retail business. The dataset has over a million order records covering 2,160 products across 4 warehouses. The pipeline takes raw order data, cleans it, turns it into weekly summaries, builds time-series features, trains a few different models, picks the best one, and then generates predictions for the following week.

The reason we picked this problem is that inventory management is genuinely hard at scale. If a warehouse orders too much it ties up cash, and if it orders too little it loses sales. A decent forecasting model that runs automatically every week can make a real difference in how a business operates.

**Main goals:**

* Build a clean, reproducible data pipeline from raw CSV to weekly predictions
* Compare multiple forecasting approaches and track them with MLflow
* Make sure the pipeline is easy to run and understand by anyone on the team

\---

## 3\. Architecture

```
Raw CSV
   │
   ▼
01\_bronze\_ingestion       →  data/interim/bronze\_\*.parquet
   │
   ▼
02\_silver\_cleaning        →  data/interim/silver\_\*.parquet
   │
   ▼
03\_gold\_aggregation       →  data/processed/gold\_weekly\_\*.parquet
   │
   ▼
04\_feature\_engineering    →  data/processed/gold\_weekly\_\*\_features.parquet
   │
   ▼
05\_model\_training         →  models/champion\_model.pkl + MLflow
   │
   ▼
06\_batch\_prediction       →  data/processed/demand\_predictions.parquet
```

\---

## 4\. Phase Deliverables

* [PHASE1.md](./PHASE1.md) — Project Design and Model Development

\---

## 5\. Setup Instructions

### Requirements

* Python 3.11
* Conda or venv

### Clone the repo

```bash
git clone https://github.com/aahme147/mlops\_se489.git
cd mlops\_se489
```

### Create the environment

**Option A — venv (recommended for this project):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".\[dev]"
```

**Option B — conda:**

```bash
conda create -n retail-demand python=3.11 -y
conda activate retail-demand
pip install -r requirements.txt
```

### Install pre-commit hooks

```bash
pre-commit install
```

### Get the data

Download from Kaggle:
[Historical Product Demand](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting)

Put the CSV file here:

```
data/raw/Historical Product Demand.csv
```

### Run the pipeline

Run notebooks in order from the project root:

```bash
python notebooks/00\_setup.py
python notebooks/01\_bronze\_ingestion\_product\_demand.py
python notebooks/02\_silver\_cleaning\_product\_demand.py
python notebooks/03\_gold\_weekly\_aggregation\_product\_demand.py
python notebooks/04\_gold\_feature\_engineering\_product\_demand.py
python notebooks/05\_model\_training\_product\_demand.py
python notebooks/06\_batch\_prediction\_product\_demand.py
```

### View MLflow runs

```bash
mlflow ui
```

Open http://127.0.0.1:5000 in your browser.

### Run tests

```bash
make test
```

Or directly:

```bash
pytest tests/ -v
```

### Run linting

```bash
make lint
```

Or directly:

```bash
ruff check .
```

\---

## 6\. Branching Strategy

We follow GitHub Flow:

```
main
 └── dev
      ├── feature/bronze-ingestion
      ├── feature/silver-cleaning
      ├── feature/model-training
      └── feature/readme
```

Never commit directly to main. Always branch off dev, do your work, open a PR, get one teammate to review, then merge into dev. When a phase is complete dev gets merged into main.

\---

## 7\. Who Did What

* **Ayan Ahmed** — pipeline architecture, bronze and silver notebooks, MLflow setup, GitHub, project structure
* **John Blaszczak** — feature engineering, model training, Prophet integration, batch prediction
* **Mark Baranovsky** — data quality checks, README, documentation, AWS pipeline setup

\---

## 8\. References

* Dataset: [Historical Product Demand on Kaggle](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting)
* Libraries: pandas, scikit-learn, MLflow, Prophet
* Project structure: [Cookiecutter MLOps SE489](https://github.com/Alizadeh-DePaul/cookiecutter-mlops-se489)
* Third-party package: [Prophet by Meta](https://facebook.github.io/prophet/)

\---

## 9\. License

MIT

