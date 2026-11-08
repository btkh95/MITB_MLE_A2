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
| Source checks | `dep_check_source_lms`, `_fe_attr`, `_fe_fin`, `_fe_click` | Before each bronze step: fails the run if the source CSV is missing, empty or lacks `Customer_ID` / `snapshot_date`; logs the rows for the month |
| Label store | `bronze_lms` → `silver_lms` → `gold_label_store` | `datamart/gold/label_store/` — loans reaching month on book 6 this month, labelled `30dpd_6mob` |
| Feature store | bronze, silver and gold for attributes, financials and clickstream → `gold_feature_store` | `datamart/gold/feature_store/joined/` — one row per applicant, as of the application date |
| Training gate | `model_train_gate` | Trains on the first training date (2024-01-01), then every month while applications arrive (launch setting; configurable, for example every 6 months later), or after a drift alert (minimum 3 months apart); otherwise skips training |
| Training | `model_train` | Trains logistic regression and XGBoost on labels already observed at the run date, then keeps the one with the higher Gini on the model-selection holdout, preferring logistic regression when the two are within 0.01. Calibrates the probabilities if that helps, and records feature importance and lineage → `model_bank/credit_model_YYYY_MM_DD_<family>/`, where the family is `logreg` or `xgboost` |
| Promotion | `model_promote` | Champion/challenger decision logged to `model_bank/model_registry.csv` |
| Inference | `model_inference` | Checks the input, scores this month's applications with the current champion and adds the top 3 reasons behind each score → `datamart/gold/model_predictions/`. Shadow models score the same applicants → `model_predictions/shadow/` |
| Monitoring | `model_monitor` | PSI of this month's scores and CSI of each input feature vs the model's out-of-time data at training, plus AUC, Gini, KS and actual vs predicted default rate for the applications scored 6 months ago, and the same live Gini for each shadow model → `datamart/gold/model_monitoring/` |
| Charts | `model_monitor_charts` | PNG charts and a CSV of every monitoring result → `datamart/gold/model_monitoring/charts/` |

## Model governance

Settings are in `static/model_config.yaml`.

- **Label availability:** a model trained on date T uses only labels observed on or before T. A default label exists 6 months after the application.
- **Data splits** (named after the course's train / validation / test / OOT split):

  | Split | Course term | Data | Decides |
  | --- | --- | --- | --- |
  | `train` | Training data | Older label months, 80% stratified by label | Fits the model; feature pruning uses train only |
  | 3-fold cross-validation inside `train` | Validation data (tuning) | Folds of train | Hyperparameters |
  | `holdout`, the **model-selection holdout** | Test data | Older label months, 20% stratified | Logistic regression or XGBoost (higher Gini; logistic regression kept if within 0.01, because a gap that small is noise on about 500 loans and the simpler model is easier to explain). The Platt calibration curve is also fitted here |
  | `oot`, the **out-of-time promotion set** | OOT 1 | The 2 newest label months at the training date | Promotion (challenger vs champion on the same loans) and whether Platt scaling is kept (lower Brier score). Also supplies the monitoring references: the Gini baseline, the PSI score reference and the CSI feature reference |
  | **Monitoring cohorts** | OOT 2, 3, ... | Each later month's scored applicants, once their outcomes arrive 6 months later | Nothing at training time: live Gini, KS, calibration gap, shadow comparison and retraining flags |

  Loans are sorted by `loan_id` before the 80/20 split, so the same loans always land in the same split whatever order the files are read in (the lineage fingerprint already ignores row order). The holdout choice never looks at OOT. Because OOT is used for promotion and the calibration decision, a promoted model's OOT Gini, which is also its monitoring baseline, is slightly optimistic. Pooling live Gini over 3 cohorts reduces sampling noise but does not remove that bias, and the 90% retrain line is an operational alert rule, not a statistical test. The artefact records the reason for each model choice.
- **Features:** an explicit allowlist. A feature is dropped at training time when more than half of that month's applicants lack it. Clickstream data stops for applicants from July 2024, so later models drop those features. The remaining features are then pruned on the training split: a feature with information value below 0.02 is dropped as weak, and of two numeric features with Spearman correlation above 0.9 the less informative one is dropped. The artefact records every feature's information value and the reason for each drop.
- **Promotion:** a new model replaces the champion when there is no champion, when the champion needs features current applicants lack, or when the new model's OOT AUC is at least 0.01 higher than the champion's on the same OOT data.
- **Retraining triggers:**
  1. a scheduled refresh every month at launch (`retrain_every_months: 1`);
  2. **source coverage**: retrain at once, with no cooldown, when any input the champion uses is missing for at least 50 percentage points more of this month's applicants than at training (for example clickstream stopping in July 2024, from 0% to 100% missing). The gate checks this before scoring, so the month a source stops is the month the model is replaced;
  3. **score drift**: score PSI ≥ 0.25;
  4. **performance**: Gini below 90% of the model's OOT Gini at training time, measured on the latest 3 monthly cohorts of that model pooled together (one cohort of about 500 loans is too noisy on its own);
  5. **calibration**: the mean predicted default rate differs from the actual rate by 5 percentage points or more over the same pooled cohorts; retraining recalibrates the probabilities.

  Triggers 3 to 5 come from last month's monitoring and respect the 3-month cooldown. Only monitoring rows about the current champion count: outcomes arrive 6 months after scoring, so a performance alert can describe a model that has already been replaced. Such alerts stay in the monitoring table for the record but do not retrain the current model. Monitoring stores each flag separately (`psi_flag`, `gini_flag`, `calibration_flag`, `coverage_flag`), and the gate logs which one caused a retrain. Monthly retraining is a launch setting: a training run takes about 20 seconds, and every new model still has to beat the champion by 0.01 OOT AUC to be promoted, so a new launch is watched closely without swapping models on noise. Once the model is stable, governance can raise the cadence to every 6 months to reduce model churn and sign-off work; the drift triggers keep working either way.
- **Feature drift:** each month, every model input is compared with the model's OOT rows using CSI (the PSI formula applied to one feature) and written to `datamart/gold/model_monitoring/feature_drift/`. CSI is a warning that explains score drift; it does not trigger retraining on its own, because with about 50 inputs one of them crossing a line by chance is likely.
- **Calibration:** after the model is chosen, Platt scaling (a one-input logistic regression on the score) is fitted on the model-selection holdout. It is kept only if it lowers the Brier score on the out-of-time promotion set (the mean squared gap between probability and outcome; lower is better). It never changes the order of applicants, so AUC and Gini are unaffected. The artefact records both Brier scores and the decision.
- **Shadow models:** the champion's runner-up (the other model type trained on the same date, named for example `credit_model_2024_01_01_xgboost_runner_up`) and any challenger that was not promoted score every month's applicants, but their scores are never used. When the labels arrive, monitoring compares their pooled live Gini with the champion's on the same applicants. A shadow ahead by 0.02 or more over 3 cohorts, while that champion is still in production, sets `shadow_flag`, which calls for a promotion review.
- **Explanations:** every prediction carries `reason_1` to `reason_3`, the inputs that pushed that applicant's risk up the most. For logistic regression this is coefficient × standardised value; for XGBoost it is the model's built-in SHAP values. The artefact also stores each input's average contribution on OOT (`feature_importance`).
- **Data checks:** training stops if there are no labelled loans, only one label class, labels other than 0/1 or duplicate loans. Inference stops if a model column is missing from the feature store, a customer appears twice, or a score is not a probability between 0 and 1. After saving, training reloads the model file and confirms it reproduces the same OOT scores. A failed check fails the Airflow task, so nothing is written from broken input.
- **Lineage:** each artefact records the Python and package versions, the training config and its hash, and a fingerprint of the exact loans and labels used, so a model can be traced and a rerun checked against it.

## Features

The model only sees an allowlist in `static/model_config.yaml`. Every input describes the applicant on the application date and is grouped by the credit question it answers.

| Question | Inputs | Why |
| --- | --- | --- |
| Have they paid late before? | `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Delayed_Payments_per_Loan`, `Payment_of_Min_Amount`, `Credit_Mix` | Past repayment behaviour is the strongest predictor of future repayment. Late payments per loan separates a few late payments spread over many loans from the same number on one or two. |
| Are they looking for more credit? | `Num_Credit_Inquiries`, `Changed_Credit_Limit`, `Num_Credit_Card`, `Num_Bank_Accounts` | Many recent applications or limit changes often come before financial stress. |
| Can they afford it? | `Annual_Income`, `Monthly_Inhand_Salary`, `Outstanding_Debt`, `Total_EMI_per_month`, `EMI_to_Salary_Ratio`, `Debt_to_Income`, `Disposable_Income`, `Credit_Utilization_Ratio`, `Interest_Rate` | Ratios compare debt with income, so they mean the same thing at every income level. The interest rate other lenders charge reflects how risky they already judge the customer. |
| How much credit do they hold? | `Num_of_Loan`, `listed_loan_count`, loan-type counts, `Num_Fin_Pdts`, `Loans_per_Credit_Item`, `Loan_Extent` | Several loans, or loans that outnumber bank and card products, point to heavy reliance on borrowing. |
| How established are they? | `Age`, `Occupation`, `Credit_History_Months`, `Spending_Behaviour`, `Payments_Size` | A longer credit history and a stable profile carry less uncertainty. |
| Online behaviour | `fe_1_mean` to `fe_20_mean`, `has_clickstream` | Average activity before the application. Only applicants up to June 2024 have it, so models trained later drop it automatically. |
| Data quality | `*_invalid`, `*_missing`, `loan_count_suspicious` | A value that had to be cleaned can itself signal risk; pruning drops the flags that carry no signal. |

Left out on purpose: names, SSNs and other personal identifiers; `clickstream_days`, which mostly measures how long ago the customer joined rather than their behaviour; `has_attributes` and `has_financials`, which are always 1; and everything in the loan file after the start date, because that is where the label comes from. Loan amount and tenure are also unused: every loan is 10,000 over 10 months, so they carry no information.

## Leakage controls

| Type | Risk in this data | Control |
| --- | --- | --- |
| Target leakage | Using information about this loan's repayment | Features come only from the attribute, financial and clickstream tables dated on the application date. The loan file is used only to build the label. Clickstream uses activity strictly before the application date. |
| Train-test contamination | Imputation, scaling, encoding or pruning fitted on holdout or OOT rows | Imputation, scaling and one-hot encoding sit inside the scikit-learn pipeline and are fitted on the training split only. Feature pruning and hyperparameter tuning also use the training split only. The model-selection holdout picks the model type; OOT decides promotion and whether calibration is kept, never the model type. |
| Temporal leakage | Training on default labels that were not yet known | A model trained on date T uses only labels observed by T, OOT is the newest label months, and a model never scores months before its training date. Unit tests cover all three. |

## Deployment

**How the model is served.** Batch scoring: once a month Airflow scores all of that month's applications and writes the scores to a gold table, which a loan decision system would read. This fits the data, which arrives in monthly snapshots, and keeps every score reproducible and auditable. In production, scoring at the point of application would use an online service (for example a small FastAPI app) that loads the same artefact and calls the same `prepare_X` code, so training and serving cannot drift apart. That service would need the applicant's features in real time from an online feature store. Streaming is not needed, because labels only arrive months later.

**How a new model goes live.**

| Strategy | What it means | Used here |
| --- | --- | --- |
| Champion/challenger | A new model replaces the current one only if it is clearly better on the same held-out data | Yes, the promotion rule |
| Shadow | A candidate scores real applicants in parallel; its scores are logged but never used | Yes, runner-up and rejected challengers |
| Canary | The new model decides for a small share of applicants first, for example 5%, then more if all is well | No: needs an online service, and outcomes take 6 months to judge |
| A/B test | Applicants are split at random between two models and outcomes compared | No: deliberately giving some applicants the weaker model raises fairness and regulatory questions in lending |
| Blue-green | Two identical environments; traffic switches from one to the other in one step | Equivalent here: the registry points to one champion, so switching is a single new row |

**Rollback.** Every model stays in `model_bank/`. To roll back, add a row to `model_registry.csv` that promotes the previous version with reason `rollback`; inference uses the latest promoted model from the next run.

**Alerts.** A failed task, including a failed data check, is retried once and then shows as failed in Airflow. In production the same failure, a `retrain_flag` or a `shadow_flag` would also send an email or Slack message through Airflow's failure callbacks.

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
| `tests/` | Unit tests for label maturity, splits, drift indices, feature rules, model choice, data checks, explanations, lineage, champion and shadow lookup |
| `notebooks/` | Model exploration notebook: transformations, feature selection, explainability (SHAP), calibration and shadow results, with figures for slides |
