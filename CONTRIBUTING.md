# Contributing to Retail Demand Forecasting

## Team
- Ayan Ahmed — aahme147@depaul.edu
- John Blaszczak — jblaszc3@depaul.edu
- Mark Baranovsky — mbarano3@depaul.edu

## Branching Strategy

We follow GitHub Flow. Every piece of work goes on its own branch and comes back through a pull request.

Never commit directly to `main`. Always branch off `dev`, do your work, open a PR, get one teammate to review, then merge.

## How to start working on something

```bash
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name
```

Do your work, then:

```bash
git add .
git commit -m "short description of what you did"
git push origin feature/your-feature-name
```

Then open a pull request on GitHub from your branch into `dev`.

## Commit messages

Keep them short and clear. Start with a verb.

Good examples:
- `add silver cleaning notebook`
- `fix date parsing bug in notebook 02`
- `update requirements with prophet`

Bad examples:
- `stuff`
- `fix`
- `changes`

## Pull request process

1. Open a PR from your feature branch into `dev`
2. Write a short description of what changed and why
3. Tag at least one teammate as a reviewer
4. Address any review comments
5. Merge once approved


## Running the pipeline

Always run notebooks in order:

```bash
python notebooks/00_setup.py
python notebooks/01_bronze_ingestion_product_demand.py
python notebooks/02_silver_cleaning_product_demand.py
python notebooks/03_gold_weekly_aggregation_product_demand.py
python notebooks/04_gold_feature_engineering_product_demand.py
python notebooks/05_model_training_product_demand.py
python notebooks/06_batch_prediction_product_demand.py
```

## Running tests

```bash
make test
```

Or directly:

```bash
pytest tests/ -v
```
