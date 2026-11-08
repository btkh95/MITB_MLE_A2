# CS611 Assignment 2: Loan Default ML Pipeline

An Airflow pipeline that builds the loan-default datamart (bronze, silver, gold), trains and selects a model, scores each month's applications, and monitors performance and stability over time.

## How to run

Needs Docker with Docker Compose running, and port `8080` free.

```sh
docker compose build
docker compose up
```

1. Open [Airflow](http://localhost:8080) and log in with `admin` / `admin`.
2. Unpause `loan_default_pipeline` (the toggle in the UI, or the command below). It backfills 31 monthly runs (2023-01 to 2025-07), one month at a time.
3. Outputs appear in `datamart/` and `model_bank/`.

The backfill takes about 1.5 to 2 hours. The first model is trained in the 2024-01 run, about 45 minutes in.

From a second terminal in this folder:

```sh
# start the backfill (same as the UI toggle)
docker compose exec airflow-scheduler airflow dags unpause loan_default_pipeline

# check progress: one row per monthly run and its state
docker compose exec airflow-scheduler airflow dags list-runs -d loan_default_pipeline

# pause it again
docker compose exec airflow-scheduler airflow dags pause loan_default_pipeline
```

To shut down:

```sh
docker compose down      # stop Airflow; outputs in datamart/ and model_bank/ stay
docker compose down -v   # also reset Airflow's run history, so the next start reruns every month
```

## What each monthly run does

| Stage | Tasks | Output |
| --- | --- | --- |
| Source checks | `dep_check_source_*` (one per source) | Fails the month if a source CSV is missing, empty or lacks its key columns |
| Label store | `bronze_lms` → `silver_lms` → `gold_label_store` | `datamart/gold/label_store/`: loans at month 6 on book, labelled `30dpd_6mob` |
| Feature store | bronze, silver, gold per source → `gold_feature_store` | `datamart/gold/feature_store/joined/`: one row per applicant on the application date |
| Training gate | `model_train_gate` | Trains from 2024-01, monthly at launch, at once if a source stops, or after a drift alert |
| Training | `model_train` | Trains logistic regression and XGBoost and keeps the one with the higher holdout Gini → `model_bank/credit_model_<date>_<family>_<fit id>/` |
| Promotion | `model_promote` | Champion/challenger decision → `model_bank/model_registry.csv` |
| Inference | `model_inference` | Scores applicants with the champion, plus the top 3 reasons per score → `datamart/gold/model_predictions/`; shadow models → `model_predictions/shadow/` |
| Monitoring | `model_monitor` | Stability (PSI, CSI), performance once outcomes arrive (Gini, KS, calibration), shadow comparisons → `datamart/gold/model_monitoring/` |
| Charts | `model_monitor_charts` | PNG charts and a CSV of all results → `datamart/gold/model_monitoring/charts/` |

## Model governance

Settings are in `static/model_config.yaml`.

**Data splits** (CS611 terms in brackets). A model trained on date T uses only outcomes known by T.

| Split | Data | Decides |
| --- | --- | --- |
| `train` (training) | Older label months, 80%, stratified | Fits the model; feature pruning |
| 3-fold CV in `train` (validation) | Folds of train | Hyperparameters |
| `holdout` (test): model-selection holdout | Older label months, 20% | Logistic regression or XGBoost; fits the Platt calibration curve |
| `oot` (OOT 1): out-of-time promotion set | 2 newest label months | Promotion; whether calibration is kept; the monitoring baselines |
| Monitoring cohorts (OOT 2, 3, ...) | Later scored months, once outcomes arrive | Live Gini, calibration, shadow comparison, retrain flags |

Loans are sorted by `loan_id` before splitting, so the split is the same whatever order the files are read in. OOT never chooses the model type, but it does decide promotion and calibration, so a promoted model's OOT Gini is a slightly optimistic baseline.

**Model choice.** The higher holdout Gini wins. If logistic regression is within 0.01 of XGBoost it is kept, since a smaller gap is noise on about 500 loans and the simpler model is easier to explain.

**Features.** An explicit allowlist. A feature is dropped when over half of that month's applicants lack it (clickstream stops in July 2024). The rest are pruned on train: information value below 0.02 is weak, and of two features with Spearman correlation above 0.9 the weaker one goes. The artefact records the reason for every drop.

**Promotion.** A new model replaces the champion when there is none, when the champion needs inputs applicants no longer have, or when it beats the champion's OOT AUC by 0.01 on the same loans.

**Retraining triggers.**

1. Scheduled: monthly at launch (`retrain_every_months: 1`).
2. Source coverage: at once, no cooldown, when a champion input is missing for 50+ percentage points more applicants than at training.
3. Score drift: PSI ≥ 0.25.
4. Performance: Gini, pooled over the latest 3 cohorts, below 90% of the OOT baseline.
5. Calibration: predicted vs actual default rate 5+ points apart over the same cohorts.

Triggers 3 to 5 come from last month's monitoring, count only rows about the current champion, and respect a 3-month cooldown. Each is stored as its own flag (`psi_flag`, `gini_flag`, `calibration_flag`, `coverage_flag`).

**Feature drift.** CSI (the PSI formula on one input) for every input each month, in `model_monitoring/feature_drift/`. It explains score drift but does not retrain on its own.

**Calibration.** Platt scaling is fitted on the holdout and kept only if it lowers the OOT Brier score (lower is better). It does not change the ranking, so Gini is unaffected.

**Shadow models.** The champion's runner-up and the latest rejected challenger score every applicant, but their scores are never used. Monitoring compares them with the champion on the same applicants: per model (`shadow_flag`, needs 3 months of the same shadow) and per role (`role_flag`, works with monthly retraining). Either flag calls for a promotion review, not an automatic switch.

**Explanations.** Every prediction carries `reason_1` to `reason_3`: coefficient × scaled value for logistic regression, built-in SHAP values for XGBoost.

**Data checks.** Training and inference stop on broken input (no labels, one class, duplicates, missing columns, scores outside 0 to 1), and training reloads each saved model to confirm it reproduces its scores.

**Versions and lineage.** Each run gets its own versioned folder, so reruns never overwrite earlier models. Training writes `model_bank/fits/<date>.json` after the reload check, and promotion reads it. Each artefact records package versions, the config, a fingerprint of the training data, the partitions read and the split's `loan_id`s.

## Model refresh SOP

| Role | Responsible for |
| --- | --- |
| Data engineer | Source data, pipeline runs, failed tasks |
| Data scientist (model owner) | Monthly monitoring review, investigating alerts, proposing changes |
| Model risk | Independent sign-off before a new model family or a manual change goes live |
| Business / credit lead | Cut-off, pricing and policy; sign-off on model-family changes |

**Monthly routine.** The DAG runs on the 1st. Within 3 working days the model owner reviews the charts and monitoring table and records "no action" or the action taken.

| Trigger | Pipeline does | Who acts | Who signs off |
| --- | --- | --- | --- |
| Task or data check fails | Stops that month | Data engineer | Model owner |
| Source stops arriving | Retrains at once | Data engineer, model owner | Model owner |
| Score drift or Gini drop | Retrains next month | Model owner | Model owner |
| Calibration 5+ points off | Retrains, refits calibration | Model owner | Model owner; business lead if pricing is affected |
| Shadow ahead | Flags a review | Model owner | Model risk |
| New model family, cut-off or policy change | Nothing automatic | Model owner, business lead | Model risk and business lead |

**Rollback.** Every version stays in `model_bank/`. Add a registry row promoting the previous version with reason `rollback`; inference uses it from the next run.

**Cadence review.** Monthly retraining is a launch setting (about 20 seconds a run). After 6 months, governance decides whether to move to every 6 months to cut model churn; the other triggers stay on. Once a year: a full review of features, cleaning thresholds and this SOP.

## Features

Every input describes the applicant on the application date and answers a credit question.

| Question | Inputs |
| --- | --- |
| Have they paid late before? | `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Delayed_Payments_per_Loan`, `Payment_of_Min_Amount`, `Credit_Mix` |
| Are they looking for more credit? | `Num_Credit_Inquiries`, `Changed_Credit_Limit`, `Num_Credit_Card`, `Num_Bank_Accounts` |
| Can they afford it? | income, salary, debt, EMI, `EMI_to_Salary_Ratio`, `Debt_to_Income`, `Disposable_Income`, `Credit_Utilization_Ratio`, `Interest_Rate` |
| How much credit do they hold? | `Num_of_Loan`, `listed_loan_count`, loan-type counts, `Num_Fin_Pdts`, `Loans_per_Credit_Item`, `Loan_Extent` |
| How established are they? | `Age`, `Occupation`, `Credit_History_Months`, `Spending_Behaviour`, `Payments_Size` |
| Online behaviour | `fe_1_mean` to `fe_20_mean`, `has_clickstream` (applicants up to June 2024 only) |
| Data quality | `*_invalid`, `*_missing`, `loan_count_suspicious` |

Left out on purpose: personal identifiers; `clickstream_days` (measures time since joining, not behaviour); `has_attributes` and `has_financials` (always 1); loan amount and tenure (the same for every loan); and anything from the loan file after the start date, which builds the label.

## Leakage controls

| Type | Control |
| --- | --- |
| Target leakage | Inputs come only from tables dated on the application date; clickstream uses activity strictly before it. The loan file only builds the label. |
| Train-test contamination | Imputing, scaling, encoding, pruning and tuning are fitted on train only. |
| Temporal leakage | A model uses only outcomes known when trained, and never scores months before its training date. Unit tests cover this. |
| Cleaning rules (assumption) | `static/valid_ranges.yaml` and the 30× income rule were set by looking at the whole file, then applied unchanged every month. A governance review would refit them on training months only. |

## Deployment

**Serving.** Batch: Airflow scores each month's applications into a gold table. In production, scoring at the moment of application would use an online service (for example FastAPI) loading the same model file and the same `prepare_X` code.

| Strategy | Used? |
| --- | --- |
| Champion/challenger | Yes: the promotion rule |
| Shadow | Yes: runner-up and rejected challengers |
| Canary | No: needs online serving, and outcomes take 6 months |
| A/B test | No: giving some applicants a weaker model is a fairness risk in lending |
| Blue-green | Built in: the registry points to one champion, so a switch is one row |

**Alerts.** A failed task is retried once, then shows as failed in Airflow. In production, failures and flags would also notify by email or Slack.

## Future work

- A fallback model without clickstream, ready the day a source stops.
- A governance limit on how much one data source can drive a model (clickstream: 0.143 Gini, about 3× the next input).
- Refitting cleaning thresholds on training months only.
- Rolling time-based validation for the model choice.

## Notebooks (optional)

Not needed to run the pipeline. After a full run:

- `notebooks/eda.ipynb`: bad rate by month on book and the income cleaning rule.
- `notebooks/deck_charts.ipynb`: the risk ladder and monitoring charts used in the slides.
- `notebooks/model_exploration.ipynb`: one model in depth (features, SHAP, permutation importance, calibration, all runs). It loads saved models, so use the package versions in `requirements.txt` plus `shap`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Layout

| Path | Purpose |
| --- | --- |
| `dags/` | Airflow DAG |
| `scripts/` | Entry points the DAG calls, one stage per date |
| `utils/` | Bronze, silver and gold transformations; model helpers |
| `static/` | Config: sources, data types, PII, valid ranges, model settings |
| `data/` | Source CSVs |
| `datamart/` | Generated tables, predictions and monitoring |
| `model_bank/` | Model versions and the registry |
| `notebooks/` | EDA, deck charts and model exploration |
| `tests/` | Unit tests |
