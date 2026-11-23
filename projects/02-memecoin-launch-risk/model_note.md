# Model note: how the model was trained, evaluated, tuned and chosen

> Drafted with Claude from my project outputs, then reviewed by me. For a data scientist. Numbers come from `outputs/*.json`, `config.yaml` and the README tables. Learning project; no claim about trading use, returns or profit.

## Task and data

- **Task:** for a coin at the moment it migrates, score the chance it crashes in the next 20 minutes. Crash = `min_ratio < 0.3` (the lowest price in those 20 minutes below 30% of the migration price). The label and the scope filter are the MELT authors' definitions.
- **Data:** MELT (arXiv 2602.13480), CC BY-NC 4.0. `feature.pkl` and `label.csv` are hash-checked and never committed. The pickle is opened through a restricted loader.
- **Scope:** at least 100 holders and 60 s of valid swaps. Of 41,470 coins with a result, 21,635 are kept (crash share 63.2%). 233 feature rows with no label were set aside.

## Split

Coins are ordered by migration time and cut 60/20/20 (seed 42):

| Pile | Coins | Crash share |
|---|---|---|
| Train | 12,981 | 61.8% |
| Validation | 4,327 | 71.3% |
| Test | 4,327 | 59.2% |

Eight coins whose 20-minute window runs past the next pile's start were kept and disclosed.

## Features

- 81 columns from groups 2 and 3, whose timing the config records as confirmed from the authors' code.
- Group 1 (calendar, SOL price) and group 4 (cluster columns) are held back and tested as ablations.
- `group1_migrate_year` and `group1_migrate_month` are never used: the leakage audit flags them `clock_like`.
- Leakage audit (train only): 122 columns, 36 with unverified timing. Flags: `clock_like` 2; `extreme_bin`, `mostly_zero`, `constant`, `strong_alone` 0. Migration time alone ranks crashes with AUC 0.5433.

## Models

Logistic regression, random forest and gradient boosting (scikit-learn), seed 42. Tuning grids are small and live in `config.yaml`.

## Evaluation

- **Metric:** PR-AUC on the crash side and on the survivor side (label 0 as the class to find). Each is read as a multiple of the lazy guess of its own pile (validation: 0.713 and 0.287). Also precision in the top 10% of scores.
- **Wobble:** 200 re-drawn copies of a pile give a middle-90% range per score and per difference. A difference counts only if its range excludes 0. This measures sampling noise inside one pile, not change between periods.
- **Untuned, validation (crash / survivor PR-AUC):** logistic regression 0.808 / 0.429; random forest 0.823 / 0.478; gradient boosting 0.826 / 0.481.
- **Overfitting:** the random forest scores 0.830 on train and 0.478 on validation on the survivor side. The training score is not an honest grade.

## Tuning

- Time-ordered, on the train pile only. Train is cut into 4 blocks; each trial learns from earlier blocks and is graded on the next. Coins whose result was not yet known when a graded block starts are dropped (a 1,200 s label purge).
- The tuning score is the average, across both sides, of PR-AUC as a multiple of the lazy guess.
- Best settings: logistic regression `C: 0.1` (4 trials, 1.412); random forest `max_depth: None, min_samples_leaf: 20` (6 trials, 1.451); gradient boosting `learning_rate: 0.03, max_iter: 100, max_leaf_nodes: 15` (8 trials, 1.440).
- **Tuned, validation (crash / survivor):** logistic regression 0.809 / 0.434; random forest 0.823 / 0.478; gradient boosting 0.814 / 0.480.
- Tuning changed little. The largest move is gradient boosting's crash side, which got worse (0.826 to 0.814). The tuned random forest and gradient boosting wobble ranges overlap almost completely.

## Column tests

On the tuned gradient boosting (reference 81 columns, 0.814 / 0.480):

| Setup | Columns | Crash | Survivor | Note |
|---|---|---|---|---|
| plus_group1 | 85 | 0.816 | 0.486 | cannot be told apart |
| plus_group4 | 116 | 0.821 (+0.007) | 0.485 | crash better; unverified timing |
| both | 120 | 0.825 (+0.010) | 0.485 | crash better; unverified timing |

Survivor changes are never separable from zero. Nothing was added to the model.

## Choice and freezing

Rule: the highest average of crash-side and survivor-side validation PR-AUC among tuned models, with a tie to the earlier candidate. Averages: logistic regression 0.621, random forest 0.650, gradient boosting 0.647. The random forest wins by 0.0033. Cut-offs per invented cost scenario are the lowest-total-cost score on validation, with a tie to the higher cut-off. The choice and cut-offs were written to `frozen_choice.yaml` (commit `e81bc82`) before the test pile was opened.

## One-time test

The frozen random forest, fitted on train only, was graded once on the test pile.

- **Ranking:** random forest crash 0.737 (1.24x), survivor 0.615 (1.51x); logistic regression 0.732 (1.24x) and 0.589 (1.44x); lazy guess 0.592 and 0.408. Random forest minus logistic regression: crash +0.006 (-0.006 to +0.018), cannot be told apart; survivor +0.024 (+0.013 to +0.038), better.
- **Frozen cut-offs (invented costs):** equal costs: 81.0% flagged, 0.3464 per coin against 0.4077 for flagging everything. Crasher costs 3x: 98.9% flagged, 0.4003 against 0.4077, nearly "flag everything". Skipped-good costs 3x: 49.0% flagged, 0.6284 against 0.5923 for flagging nothing, so worse than doing nothing.

## What could be inflated or misleading

- Validation was reused for the column tests, tuning judgement, model choice and cut-offs, so validation scores are somewhat optimistic. The test pile only graded frozen choices.
- The crash side looks good because most coins crash.
- The cut-off failure on test cannot be split into base-rate shift (71.3% to 59.2%) and a cut-off tuned too closely to validation. The test pile cannot be used to find out.
- The wobble is narrower than the real uncertainty because it does not cover change between periods.
- Many comparisons were made, so a small gain can be luck.
- 36 columns have unverified timing. The group-4 gains are treated as possible leakage.
- The costs are invented. The test lock file only guards my own machine; a fresh clone has no lock.
