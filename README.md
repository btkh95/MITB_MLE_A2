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
| Training | `model_train` | Trains logistic regression and XGBoost on labels already observed at the run date, then keeps the one with the higher validation Gini, preferring logistic regression when the two are within 0.01 → `model_bank/credit_model_YYYY_MM_DD/` |
| Promotion | `model_promote` | Champion/challenger decision logged to `model_bank/model_registry.csv` |
| Inference | `model_inference` | Scores this month's applications with the current champion → `datamart/gold/model_predictions/` |
| Monitoring | `model_monitor` | PSI of this month's scores and CSI of each input feature vs the model's out-of-time data at training, plus AUC, Gini, KS and actual vs predicted default rate for the applications scored 6 months ago → `datamart/gold/model_monitoring/` |
| Charts | `model_monitor_charts` | PNG charts and a CSV of every monitoring result → `datamart/gold/model_monitoring/charts/` |

## Model governance

Settings are in `static/model_config.yaml`.

- **Label availability:** a model trained on date T uses only labels observed on or before T. A default label exists 6 months after the application.
- **Validation:** the two newest label months are the out-of-time (OOT) test set. Earlier loans split 80/20 into train and validation, stratified by label. Hyperparameters are tuned with 3-fold cross-validation on the training split. The model type (logistic regression or XGBoost) is chosen on validation Gini. If logistic regression is within 0.01 of XGBoost, logistic regression is kept, because a gap that small is noise on a validation set of about 500 loans and the simpler model is easier to explain; the artefact records the reason for each choice. The OOT set is never used for a choice inside a training run, so its Gini is an honest estimate and serves as the monitoring baseline.
- **Features:** an explicit allowlist. A feature is dropped at training time when more than half of that month's applicants lack it. Clickstream data stops for applicants from July 2024, so later models drop those features. The remaining features are then pruned on the training split: a feature with information value below 0.02 is dropped as weak, and of two numeric features with Spearman correlation above 0.9 the less informative one is dropped. The artefact records every feature's information value and the reason for each drop.
- **Promotion:** a new model replaces the champion when there is no champion, when the champion needs features current applicants lack, or when the new model's OOT AUC is at least 0.01 higher than the champion's on the same OOT data.
- **Retraining triggers:** a scheduled refresh every 6 months; score PSI ≥ 0.25; or Gini below 90% of the model's OOT Gini at training time, measured on the latest 3 monthly cohorts of that model pooled together (one cohort of about 500 loans is too noisy on its own).
- **Next step, not built:** shadow deployment. The runner-up model would also score each month's applicants without affecting decisions, and once their labels arrive 6 months later the two models' live Gini would be compared before deciding on promotion.
- **Feature drift:** each month, every model input is compared with the model's OOT rows using CSI (the PSI formula applied to one feature) and written to `datamart/gold/model_monitoring/feature_drift/`. CSI is a warning that explains score drift; it does not trigger retraining on its own, because with about 50 inputs one of them crossing a line by chance is likely.

## Tests

The rules that prevent leakage and the drift calculations have unit tests. Run them from the project root; they need only the packages in `requirements.txt`:

```bash
python -m unittest discover -s tests -v
```

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
| `tests/` | Unit tests for label maturity, splits, drift indices, feature rules, model choice and champion lookup |
