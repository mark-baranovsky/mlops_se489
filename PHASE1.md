# PHASE 1: Project Design & Model Development
 
## Overview
Phase 1 establishes the foundation for our MLOps project. This phase covers project planning, code organization, team collaboration setup, data handling, baseline model development, and documentation.
 
---
 
## 1. Project Proposal
 
- [x] **Scope & Objectives**: Forecast weekly retail product demand per warehouse-product combination to help inventory management decisions
- [x] **Detailed Description**: See section below
- [x] **Dataset Selection**: Historical Product Demand from Kaggle — over 1 million rows of real retail order data
- [x] **Dataset Description**: 1,048,575 rows, 5 columns (Product_Code, Warehouse, Product_Category, Date, Order_Demand), covering 2,160 products across 4 warehouses from 2011 to 2017
- [x] **Model Considerations**: Baseline mean-lag, Random Forest, Gradient Boosting, Prophet
- [x] **Open-Source Tools**: pandas, scikit-learn, MLflow, Prophet (Meta)
 
**Project Description (300+ words):**
 
We wanted to work on something that felt like a real business problem rather than a toy dataset. Demand forecasting fit that description well — it shows up in almost every company that deals with physical inventory, and getting it wrong has direct financial consequences.
 
The goal of this project is to build a pipeline that takes historical order data and produces reliable weekly demand forecasts for each product and warehouse combination. We wanted the whole thing to be automated, reproducible, and easy to hand off to someone else without them needing to figure out what we did.
 
The dataset we are working with has just over a million rows of order history across 2,160 products and 4 warehouse locations, spanning roughly 2011 to 2017. Each row tells us which product was ordered, from which warehouse, on which date, and in what quantity.
 
The first thing we noticed when exploring the data is that it is pretty messy. Dates come in inconsistent formats, demand values sometimes have parentheses around them to indicate returns, and there are quite a few null dates that need to be dropped. We handle all of that in the Silver cleaning notebook before anything else touches the data.
 
Once the data is clean, we aggregate it from individual orders up to weekly demand totals per warehouse-product pair. This is the Gold layer. From there we build features that a model can actually use — how much was sold last week, the week before that, the rolling average over the past month, whether demand is trending up or down, and what the same week looked like a year ago.
 
For modeling we tried four different approaches. The first is a simple baseline that just averages the last four weeks of demand. We then trained a Random Forest and a Gradient Boosting model using scikit-learn, both with median imputation for the lag features that are missing at the start of each product history. Finally we used Prophet, which is Meta's open-source time-series library, on the single most active product in the dataset. Prophet is the required third-party package for the course.
 
All four runs are tracked in MLflow. We log parameters, metrics, and model artifacts for each one, and at the end we tag the best model as the champion and save it to disk. The batch prediction notebook loads that model and generates next-week forecasts for every product-warehouse pair that was active in the most recent week.
 
**Success metrics:**
- Validation RMSE below the mean-lag baseline
- Zero nulls on critical columns after Silver cleaning
- MLflow tracking 4 clean experiment runs with logged metrics
 
---
 
## 2. Code Organization & Setup
 
- [x] **GitHub Repository**: https://github.com/aahme147/mlops_se489 — generated using professor's Cookiecutter MLOps template
- [x] **Environment Setup**: Python 3.11 via venv (.venv) with all dependencies installed
- [x] **Dependency Management**: `requirements.txt` and `pyproject.toml` with all pinned dependencies
- [x] **Project Structure**: Cookiecutter structure with `notebooks/`, `data/`, `models/`, `tests/`, `src/`
- [x] **Version Pinning**: All dependencies pinned in requirements.txt and pyproject.toml
- [x] **Installation Documentation**: Full setup instructions in README.md
 
---
 
## 3. Version Control & Collaboration
 
- [x] **Regular Commits**: Commits made after each working pipeline stage with descriptive messages
- [x] **Branching Strategy**: GitHub Flow — feature branches, PRs into main
- [x] **Pull Request Process**: PRs opened and merged by each team member
- [x] **Team Roles**: Defined below and in CONTRIBUTING.md
- [x] **Code Review Guidelines**: Documented in CONTRIBUTING.md
- [x] **Commit History**: Clean history tracking each pipeline stage per contributor
 
**Team roles:**
- Ayan Ahmed — pipeline architecture, data ingestion, cleaning, MLflow setup, GitHub
- John Blaszczak — feature engineering, model training, Prophet integration, batch prediction
- Mark Baranovsky — documentation, README, PHASE1.md, data quality checks
 
---
 
## 4. Data Handling
 
- [x] **Data Cleaning Scripts**: `02_silver_cleaning_product_demand.py` — handles nulls, type casting, date parsing, negative demand
- [x] **Normalization**: Median imputation for lag features via sklearn Pipeline in model training
- [x] **Data Augmentation**: Not applicable for time-series demand forecasting
- [x] **Data Documentation**: Data dictionary below
- [x] **Data Splits**: Time-based split — train before 2016-10-01, validation after
- [x] **Data Validation**: Null checks and sanity checks printed at end of each notebook
 
**Pipeline stages:**
 
| Notebook | Input | Output | Rows |
|---|---|---|---|
| 01_bronze | CSV | bronze.parquet | 1,048,575 |
| 02_silver | bronze.parquet | silver.parquet | 1,008,664 |
| 03_gold | silver.parquet | gold.parquet | 363,139 |
| 04_features | gold.parquet | features.parquet | 363,139 |
| 06_predict | features.parquet | predictions.parquet | 3 |
 
**Data dictionary:**
 
| Feature | Type | Description |
|---|---|---|
| product_code | string | Unique product identifier |
| warehouse | string | Warehouse location code |
| product_category | string | Product category grouping |
| order_date | date | Date of individual order |
| order_demand | int | Units ordered (negative = return) |
| week_start_date | date | Monday of the order week |
| lag_1 to lag_4 | float | Demand 1-4 weeks ago |
| lag_52 | float | Demand same week last year |
| rolling_4wk_avg | float | 4-week rolling average demand |
| rolling_4wk_std | float | 4-week rolling standard deviation |
| rolling_8wk_avg | float | 8-week rolling average demand |
| demand_momentum | float | lag_1 minus 4-week average |
| yoy_demand_change | float | Change vs same week last year |
| quarter | int | Calendar quarter 1-4 |
| is_quarter_end_week | int | 1 if week 13, 26, 39, or 52 |
| is_peak_month | int | 1 if November, December, or January |
| weekly_order_demand | float | Target variable — weekly demand to predict |
 
---
 
## 5. Model Training
 
- [x] **Training Environment**: Local Mac with venv, MLflow local tracking
- [x] **Baseline Model**: Mean of last 4 lag features — val RMSE 50,753
- [x] **Hyperparameter Configuration**: Documented below
- [x] **Evaluation Metrics**: RMSE and MAE on time-based validation set
- [x] **Model Persistence**: Champion model saved as `models/champion_model.pkl` via joblib
- [x] **Training Reproducibility**: Random seed 42 set on all models, time-based split ensures no data leakage
- [x] **Performance Baseline**: Random Forest val RMSE 42,479 vs baseline 50,753
 
**Model results:**
 
| Model | Train RMSE | Val RMSE | Val MAE |
|---|---|---|---|
| Baseline (mean lag) | 54,470 | 50,753 | 9,277 |
| Random Forest | 42,278 | 42,479 | 8,578 |
| Gradient Boosting | 39,051 | 42,899 | 8,106 |
| Prophet (top product) | — | 516 | 326 |
 
**Hyperparameters:**
 
| Model | Parameters |
|---|---|
| Random Forest | n_estimators=100, max_depth=6, random_state=42 |
| Gradient Boosting | n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42 |
| Prophet | weekly_seasonality=True, yearly_seasonality=True |
 
Champion: **Random Forest** — val RMSE 42,479
 
---
 
## 6. Documentation & Reporting
 
- [x] **README**: Comprehensive README with project overview, setup, quick start, dependencies, contributions, license
- [x] **Code Docstrings**: Descriptive print statements and variable names throughout all notebooks
- [x] **Code Style**: Ruff configured in pyproject.toml
- [x] **Type Hints**: Applied where required by mypy
- [x] **Type Checking**: mypy configured in pyproject.toml
- [x] **Makefile**: Included from Cookiecutter template with setup, train, test, lint, format targets
- [x] **CONTRIBUTING.md**: Documents branching strategy, commit style, PR process, code style, and how to run the pipeline
- [x] **API Documentation**: All notebooks documented with descriptive variable names and print statements
 
**Test coverage (11 tests passing):**
 
| Test | What it checks |
|---|---|
| test_bronze_columns_preserved | Raw columns intact after ingestion |
| test_bronze_all_strings | All Bronze columns are strings |
| test_silver_demand_parsing | Parenthesized negatives parsed correctly |
| test_silver_no_nulls_after_cleaning | No nulls on critical columns |
| test_silver_no_zero_demand | Zero demand rows removed |
| test_silver_date_parsing | Dates parsed correctly |
| test_gold_weekly_aggregation | Weekly groupby works correctly |
| test_gold_no_duplicate_keys | No duplicate warehouse-product-week keys |
| test_lag_features_no_leakage | Lag features never include current week |
| test_predictions_non_negative | All predictions floored at zero |
| test_time_based_split | Train and val sets correctly separated by date |