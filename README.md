# CS611 Assignment 2: Loan Default ML Pipeline

An Airflow pipeline that builds the loan-default datamart (bronze, silver and gold), trains and selects a model, scores each month's loan applications, and monitors model performance and stability over time.

## How to run

Prerequisites: Docker with Docker Compose, with the Docker engine running, and port `8080` free.

```sh
docker compose build
docker compose up
```

1. Open [Airflow](http://localhost:8080) and log in with `admin` / `admin`.
2. Unpause the `loan_default_pipeline` DAG. It backfills monthly runs from 2023-01-01, one month at a time, in date order.
3. Outputs appear in `datamart/` and `model_bank/` in this folder.

To stop, run `docker compose down`. To also reset Airflow's run history, run `docker compose down -v`.

The backfill covers 31 monthly runs (2023-01 to 2025-07). Airflow runs one task at a time, so the full backfill takes about 1.5–2 hours (about 3.5 minutes per month on a laptop); progress is visible in the Grid view. Training first happens in the 2024-01 run, so models, predictions and monitoring appear from about 45 minutes in.

## What each monthly run does

| Stage | Tasks | Output |
| --- | --- | --- |
| Label store | `bronze_lms` → `silver_lms` → `gold_label_store` | `datamart/gold/label_store/` — loans reaching month on book 6 this month, labelled `30dpd_6mob` |
| Feature store | bronze, silver and gold for attributes, financials and clickstream → `gold_feature_store` | `datamart/gold/feature_store/joined/` — one row per applicant, as of the application date |
| Training gate | `model_train_gate` | Trains on the first training date (2024-01-01), every 6 months after that, or after a drift alert (minimum 3 months apart); otherwise skips training |
| Training | `model_train` | Trains logistic regression and XGBoost on labels already observed at the run date, then keeps the one with the higher out-of-time Gini → `model_bank/credit_model_YYYY_MM_DD/` |
| Promotion | `model_promote` | Champion/challenger decision logged to `model_bank/model_registry.csv` |
| Inference | `model_inference` | Scores this month's applications with the current champion → `datamart/gold/model_predictions/` |
| Monitoring | `model_monitor` | PSI of this month's scores vs the model's out-of-time scores at training, plus AUC, Gini, KS and actual vs predicted default rate for the applications scored 6 months ago → `datamart/gold/model_monitoring/` |
| Charts | `model_monitor_charts` | PNG charts and a CSV of every monitoring result → `datamart/gold/model_monitoring/charts/` |

## Model governance

Settings are in `static/model_config.yaml`.

- **Label availability:** a model trained on date T uses only labels observed on or before T. A default label exists 6 months after the application.
- **Validation:** the two newest label months are the out-of-time (OOT) test set. Earlier loans split 80/20 into train and validation, stratified by label. Hyperparameters are tuned with 3-fold cross-validation on the training split.
- **Features:** an explicit allowlist. A feature is dropped at training time when more than half of that month's applicants lack it. Clickstream data stops for applicants from July 2024, so later models drop those features.
- **Promotion:** a new model replaces the champion when there is no champion, when the champion needs features current applicants lack, or when the new model's OOT AUC is at least 0.01 higher than the champion's on the same OOT data.
- **Retraining triggers:** a scheduled refresh every 6 months; score PSI ≥ 0.25; or Gini below 90% of the model's OOT Gini at training time.

## Layout

| Path | Purpose |
| --- | --- |
| `dags/` | Airflow DAG definition |
| `scripts/` | Command-line entry points that the DAG calls, one stage per snapshot date |
| `utils/` | Bronze, silver and gold transformations (from Assignment 1) |
| `static/` | Source filenames, data types, PII columns, valid ranges and loan-type taxonomy |
| `data/` | Source CSVs |
| `datamart/` | Generated bronze, silver and gold tables |
| `model_bank/` | Trained model artefacts |
