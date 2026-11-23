# Bank call targeting (Project 01)

A phone team can only call some clients. This project ranks clients by how likely they are to subscribe to a term deposit, so the team can call the most promising ones first. It uses the public UCI Bank Marketing data and is built to be checked: the split follows time, every column is audited for leakage, and the test set was used once.

## Short, honest summary

- The models rank clients better than a blind guess, but not by much. On the validation month, PR-AUC is 0.145 to 0.150, which is 1.58x to 1.66x a blind guess (0.0905).
- Logistic regression, random forest and gradient boosting cannot be told apart on validation (the ranges overlap). The simplest, logistic regression, was chosen.
- Tuning did not clearly help: it moved logistic regression by +0.006, made no clear change to the random forest, and made gradient boosting worse (-0.021).
- One-time test run: PR-AUC 0.607 against a blind guess of 0.445, which is 1.36x. That is a smaller edge than on validation.
- The cut-offs chosen on validation lost to "phone everyone" on test in all three assumed cost scenarios. The yes-rate was 44.50% in the test months against 9.05% in validation, and at the assumed call costs, phoning everyone pays when the yes-rate is that high. This project did not investigate why the yes-rate was so different.
- Adding `duration` (the call length, only known after the call) lifts validation PR-AUC by +0.348 to +0.373 (middle values, three models). That is leakage, shown on purpose and never used in the real model.

## Data and credit

- Data: UCI Bank Marketing, file `bank-additional-full.csv`, 41,188 rows (12 exact duplicates are dropped before splitting). Licence: CC BY 4.0.
- Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- Paper behind the data: Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems.
- The data is not stored in this repository. `src/get_data.py` downloads it and rebuilds `data/bank.db` when that file is missing, and stops if the row count or the number of yes rows is wrong.
- The data comes from a Portuguese bank, 2008 to 2010. Nothing here says how it would work for another bank or today.

## How to run

Python 3.14.4 was used. From the project folder:

```
python -m venv .venv
source .venv/bin/activate        # on Windows Git Bash: source .venv/Scripts/activate
python -m pip install -r requirements.txt
bash run.sh
```

- `run.sh` takes about 2 minutes. On a fresh clone it first downloads the data (needs internet, about 1 MB). It runs every step that uses training and validation data and writes the results to `outputs/` (git-ignored). It never runs the one-time test run.
- Tests: `python -m pytest -q`. One test runs `run.sh` for real and is skipped unless you set `RUN_SLOW=1` (CI sets it).
- Check this README against the saved results: `python src/check_readme.py`.
- One-time test run: `python src/final.py`. It uses the test set, writes `outputs/test_results.json`, and refuses to run a second time. It is not part of `run.sh`.
- `eda.ipynb` also needs `jupyter` and `matplotlib`. They are not pinned in `requirements.txt` because `run.sh` does not use them.

All settings (split months, feature groups, model settings, tuning grids, cost scenarios) are in `config.yaml`.

## Method in brief

- **Time split.** The file has no date column, but the rows are in date order, so the year is rebuilt from the order of the month names. This relies on the dataset authors' statement that the file starts in May 2008. Splits are cut at month ends: train May 2008 to Apr 2009 (6.27% yes), validation May 2009 (9.05% yes), test Jun 2009 to Nov 2010 (44.50% yes, 4,962 clients).
- **Columns.** A column is allowed only if it is known before the call. `duration` is excluded because it is only known after the call. `campaign` is excluded from the first model because its timing is unclear. Month, day of week and the economy columns are held back and tested as ablations, because in this file they act as a clock.
- **Models.** Logistic regression, random forest and gradient boosting, each inside a pipeline (features, then model), so the feature steps are learned from the training rows only.
- **Metrics.** PR-AUC is always read next to the blind-guess score of the same set. Also precision, recall and lift in the top 10% of clients, and an "always no" row (high accuracy, finds nobody).
- **Wobble.** 200 re-drawn copies of the validation month (seed 42) give a middle-90% range for each score. A difference counts only if its range does not include 0.
- **Cut-offs.** The cut-off is a share of clients to phone, chosen on validation to maximise (subscriptions found minus cost x calls), for each cost scenario.
- **Tuning.** Time-aware: fit on the first 5 months of training, test on the 6th, then grow the window (practice exams: 200811, 200812, 200903, 200904). Small grids per model, judged on validation with the wobble ranges.

## Results

All numbers below are generated from `outputs/*.json` and `outputs/leakage_audit.csv`. `python src/check_readme.py` checks every table row against those files, so a number under the wrong label is caught. Numbers that appear only in the summary sentences are not all checked.

### 1. Models on validation (May 2009)

| Model | Train PR-AUC (x blind guess) | Validation PR-AUC (x blind guess) | Precision, top 10% | Recall, top 10% | Lift, top 10% |
|---|---|---|---|---|---|
| Logistic regression | 0.119 (1.89x) | 0.145 (1.60x) | 0.167 | 0.185 | 1.85 |
| Random forest | 0.189 (3.02x) | 0.150 (1.66x) | 0.191 | 0.212 | 2.12 |
| Gradient boosting | 0.190 (3.03x) | 0.143 (1.58x) | 0.193 | 0.214 | 2.13 |
| Always "no" | - | 0.0905 (blind guess) | 0.000 | 0.000 | - |

The blind guess on validation is 0.0905, and on training 0.0627. The random forest and boosting score about 3.02x on the data they learned from but only 1.66x on validation, which is a sign of overfitting. The training score is not an honest grade.

### 2. How much the validation scores wobble

| Model | Validation PR-AUC, middle 90% | Lift, top 10%, middle 90% | PR-AUC minus logistic regression | Verdict |
|---|---|---|---|---|
| Logistic regression | 0.146 (0.128 to 0.169) | 1.86 (1.60 to 2.11) | - | reference |
| Random forest | 0.152 (0.133 to 0.173) | 2.12 (1.83 to 2.41) | +0.006 (-0.005 to +0.018) | cannot be told apart |
| Gradient boosting | 0.144 (0.127 to 0.167) | 2.15 (1.85 to 2.43) | -0.000 (-0.015 to +0.009) | cannot be told apart |

### 3. Leakage audit (training rows only)

For every column, the yes-rate by value or bin is saved in `outputs/leakage_audit.csv`. A segment is flagged when it has at least 50 rows and a yes-rate 3x above or below the training yes-rate (6.27%), or exactly 0% or 100%. The 3x and 50 rows are assumptions, not rules from the data. Flagged segments:

| Column | Segment | Rows | Yes-rate | Times the training yes-rate |
|---|---|---|---|---|
| previous | 2 | 58 | 20.69% | 3.30 |
| poutcome | success | 102 | 38.24% | 6.10 |
| month | mar | 282 | 44.68% | 7.13 |
| month | oct | 67 | 62.69% | 10.00 |
| emp_var_rate | -1.8 | 2,739 | 20.74% | 3.31 |
| cons_price_idx | 92.843 | 282 | 44.68% | 7.13 |
| cons_price_idx | 93.798 | 67 | 62.69% | 10.00 |
| cons_conf_idx | -50.0 | 282 | 44.68% | 7.13 |
| cons_conf_idx | -40.4 | 67 | 62.69% | 10.00 |
| nr_employed | 5099.1 | 2,739 | 20.74% | 3.31 |
| duration | (-0.001, 86.0] | 6,128 | 0.16% | 0.03 |
| duration | (86.0, 140.0] | 6,062 | 1.29% | 0.21 |
| duration | (140.0, 214.0] | 6,087 | 1.91% | 0.30 |
| duration | (362.0, 4918.0] | 6,079 | 24.84% | 3.96 |

- Several different columns show the same row counts and yes-rates (for example 282 rows at 44.68% under `month`, `cons_price_idx` and `cons_conf_idx`). That is what a clock looks like: these values identify a stretch of time, not a kind of client. This is an inference from the pattern, not something proven.
- `duration` is flagged in 4 of 5 bins and is excluded from every real model.
- `campaign` is not flagged, but its problem is timing, which this audit cannot see.
- The written reasons for each flagged column are in `decision_log.md`.

### 4. Ablations (change in validation PR-AUC against the baseline, middle 90%)

Baseline PR-AUC: logistic regression 0.146, random forest 0.152, gradient boosting 0.144.

| Setup | Logistic regression | Random forest | Gradient boosting |
|---|---|---|---|
| without_history | -0.040 (-0.059 to -0.027), worse | -0.037 (-0.058 to -0.023), worse | -0.029 (-0.041 to -0.018), worse |
| plus_calendar | -0.007 (-0.015 to -0.000), worse | -0.006 (-0.015 to +0.005), cannot tell | -0.036 (-0.051 to -0.022), worse |
| plus_economy | -0.003 (-0.009 to +0.003), cannot tell | +0.004 (-0.004 to +0.013), cannot tell | -0.004 (-0.019 to +0.011), cannot tell |
| plus_campaign | +0.006 (+0.003 to +0.009), better | +0.005 (+0.001 to +0.011), better | +0.003 (-0.001 to +0.006), cannot tell |
| plus_duration_LEAKAGE_DEMO | +0.360 (+0.324 to +0.400), better | +0.373 (+0.328 to +0.411), better | +0.348 (+0.315 to +0.376), better |

- `plus_duration_LEAKAGE_DEMO` is a deliberate demonstration of leakage and is never used.
- With 15 comparisons, a small win (such as `plus_campaign`) could be luck. Nothing was added to the final model because of it.

### 5. Time-aware tuning, before and after

| Model | Best settings tried | Practice exams, default | Practice exams, tuned | Folds improved | Validation PR-AUC, default to tuned | Change on validation |
|---|---|---|---|---|---|---|
| Logistic regression | {C: 0.01} | 0.237 +/- 0.194 | 0.242 +/- 0.202 | 3 of 4 | 0.145 to 0.151 | +0.006 (+0.002 to +0.010), better |
| Random forest | {min_samples_leaf: 20, max_depth: 12} | 0.247 +/- 0.217 | 0.250 +/- 0.208 | 3 of 4 | 0.150 to 0.150 | -0.000 (-0.004 to +0.003), cannot tell |
| Gradient boosting | {max_iter: 50, max_leaf_nodes: 15, min_samples_leaf: 100} | 0.243 +/- 0.215 | 0.244 +/- 0.202 | 2 of 4 | 0.143 to 0.122 | -0.021 (-0.033 to -0.012), worse |

Tuning did not clearly help. The practice-exam scores swing a lot between months (see the +/- values). A likely reason is that months differ a lot in yes-rate, but this was not checked.

### 6. Cost scenarios (invented assumptions)

One subscription is worth 1 unit and one call costs the amount shown. These costs are invented to show how the advice changes. They are not the bank's costs. "Net" is units gained per 100 clients. These are for the final model (logistic regression, C = 0.01). The share phoned was chosen on validation. The test columns use the same share on the test months.

| Scenario (assumed cost per call) | Share phoned (chosen on validation) | Validation: net per 100 clients | Validation: call everyone | Test: net per 100 clients | Test: call everyone |
|---|---|---|---|---|---|
| cheap (0.03) | 97% | 6.08 | 6.05 | 40.90 | 41.50 |
| moderate (0.09) | 40% | 1.34 | 0.05 | 20.24 | 35.50 |
| expensive (0.15) | 5% | 0.49 | -5.95 | 2.92 | 29.50 |

### 7. The one-time test run

| Measure | Value |
|---|---|
| Model | logistic_regression, C = 0.01, max_iter = 1000, fitted on training rows only |
| Test clients | 4,962 |
| Yes-rate: train / validation / test | 6.27% / 9.05% / 44.50% |
| Test PR-AUC (middle 90%) | 0.607 (0.589 to 0.628) |
| Blind guess PR-AUC on the same set | 0.445 |
| Test PR-AUC as a multiple of a blind guess | 1.36x |
| Top 10% (497 calls): precision / recall / lift | 0.706 / 0.159 / 1.59 |
| Always "no": accuracy / precision / recall | 0.555 / 0.000 / 0.000 |

## Numbers that could mislead

- **The test numbers look high only because the test yes-rate is 44.50%.** A PR-AUC of 0.607 and a top-10% precision of 0.706 cannot be compared with the validation numbers directly. Read them as multiples of a blind guess: 1.36x for PR-AUC and 1.59x lift.
- **No selection inflation on test.** Nothing was chosen on the test set: the model, its settings and the cut-offs were fixed on training and validation first, and the test run refuses a second go.
- **The wobble ranges are narrower than the real uncertainty.** They only measure luck within the one validation month. They do not measure change from month to month, which the test run shows is large.
- **Validation is a single month.** The three models being tied could change on another month.
- **Many comparisons.** Over 15 ablation comparisons and the tuning grids, a small gain can be luck.
- **Training scores are inflated** for the forest and boosting (overfitting). Only validation and test scores are honest.
- **The cost scenarios are assumptions.** The call costs are invented. No profit claim is made.
- **The year is rebuilt** from the row order, as described above.

## Assumptions and limits

- Costs, the 3x flag and the 50-row minimum are assumptions, labelled as such.
- Only one dataset, from one bank and one period.
- `decision_log.md` records the choices, and `recommendation.md` gives the plain-language advice and what the bank should check before relying on the model.

## Documents

- `decision_log.md`: the questions, the key decisions and the reasons, including each flagged column.
- `model_note.md`: how the model was trained, evaluated and tuned.
- `recommendation.md`: one page for a non-technical reader.
- `learning_log.md`: what was learned.
- `prompt_chat_history.md`: the raw AI chat history.

## AI tools used

- Claude (Anthropic) was used as a trainer and code assistant. It wrote the code, the tests and this README, working from my instructions and reading my results. I ran everything, made my own commits, branches and pull requests, and wrote `decision_log.md`, `model_note.md`, `recommendation.md` and `learning_log.md` in my own words.
- An automated GitHub reviewer (the OpenAI Codex connector, `chatgpt-codex-connector`) posts a review summary on each pull request.
- The full chat history is in `prompt_chat_history.md`.
