---
title: Retail Demand Forecasting
emoji: 📦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# Retail Demand Forecasting

This Streamlit app predicts weekly retail order demand using the trained champion model from the MLOps pipeline.

## Files

- `app.py`: Streamlit prediction UI
- `requirements.txt`: Hugging Face Space dependencies
- `models/champion_model.pkl`: trained model copied by GitHub Actions

## Model

The app expects the trained model at:

```text
models/champion_model.pkl
```
