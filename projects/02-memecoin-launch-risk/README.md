# Memecoin launch risk (Project 02)

A learning project. For a Solana memecoin at the moment it "migrates" in the public MELT dataset, it scores how likely the coin is to crash in the next 20 minutes. A crash here means the lowest price in those 20 minutes falls below 30% of the price at migration. The score is a **risk flag, not a trade signal**, and this project makes **no claim about trading, returns or profit**. It is built to be checked: the split follows time, every column is audited for leakage, the model and cut-offs were frozen in a committed file before the test pile was opened, and the test pile was opened once.

## Short, honest summary

- The flag ranks coins better than a lazy guess, but modestly. For the chosen random forest on validation, crash-side PR-AUC is 0.823 against a lazy guess of 0.713 (1.15x). The survivor side is 0.478 against 0.287 (1.66x). Most coins crash, so the crash side has little room above the lazy guess. The survivor side is where the model has more to say.
- Logistic regression, random forest and gradient boosting finish close together. A written rule (the highest average of the two sides on validation) picked the random forest. Its lead over gradient boosting is 0.0033, and the wobble ranges of the two overlap almost completely (section 4), so this is close to a coin flip. Tuning changed little (section 4).
- Adding the held-back column groups lifts the crash side by at most +0.010, and the groups that help carry unverified timing. Nothing was added to the model (section 5).
- On the one-time test, ranking quality held up relative to the lazy guess: crash side 0.737 (1.24x), survivor side 0.615 (1.51x). The random forest's edge over logistic regression is small: on the crash side the two cannot be told apart, and on the survivor side it is +0.024 (+0.013 to +0.038).
- The cut-offs held up unevenly. The equal-cost cut-off still beat both lazy options on test (0.3464 against 0.4077 for flagging everything), but it flags 81.0% of coins. The "holding a crasher costs 3x" cut-off flags 98.9% and is close to "flag everything". The "skipping a good coin costs 3x" cut-off, chosen on validation, lost on test (0.6284) to flagging nothing (0.5923). The crash share fell from 71.3% in validation to 59.2% in test. I cannot separate how much of that failure comes from the shift and how much from a cut-off tuned to validation, and the test pile cannot be used to find out.
- All cost numbers are invented. They show how the best cut-off moves when the two mistakes are not equally bad. They are not measured from any trading.

## Data and credit

- Data: MELT, "MELT: A Behavioral Trace Dataset for High-Risk Memecoin Launch Detection" (arXiv 2602.13480), https://github.com/git-disl/MELT. Licence: CC BY-NC 4.0 (non-commercial; credit the authors). This project is non-commercial and credits them here.
- Two files are used. `feature.pkl` (517,642,519 bytes, sha256 `1b27d71f911aa269b1953c834d67cc6d87d31dffc5e8f58f8847b6e560b41f71`) and `label.csv` (3,861,124 bytes, sha256 `dcfcce188fb352466b76ab8345f79633766c93e96d29729282ab114133fcc09a`). The code refuses to run if either hash differs.
- The data is not stored in this repository. `src/get_data.py` downloads `label.csv`. `feature.pkl` must be saved by hand from the authors' Google Drive link in the MELT README (the link shows a virus-scan page, so it cannot be scripted). The pickle is opened through a restricted loader (`src/safe_pickle.py`), because pickle files can run code.
- The crash label (lowest price in the 20 minutes after migration below 30% of the migration price) and the scope filter are the authors' definitions. I did not re-derive them.

## How to run

Python 3.14.4 was used. From this project folder:

```
python -m venv .venv
.venv\Scripts\activate            (Windows;  on Linux or macOS:  source .venv/bin/activate)
python -m pip install -r requirements.txt
python src/get_data.py             (downloads label.csv, checks both files; save feature.pkl into data/ first)
python src/pipeline.py             (or:  bash run.sh)
```

- `python src/pipeline.py` checks the data files, prepares the piles, runs the leakage audit, trains the three models, measures the wobble, re-runs the tuning, runs the column tests and then verifies that the frozen choice reproduces from scratch. It took 95 seconds on my machine. It uses only the training and validation piles and never opens the test pile.
- On real data the tuning re-run writes to `outputs/tuned_params_rerun.yaml` and the verify step compares it with the committed `tuned_params.yaml` and `frozen_choice.yaml`. A difference stops the run and is named. Nothing committed is overwritten.
- `--skip-tune` skips the slow tuning re-run. `--sample` (or `bash run.sh --sample`) runs every step on made-up data in a temporary folder, with nothing downloaded. Its numbers mean nothing; it only shows that the steps run and hang together.
- Tests: `python -m pytest -q`. One test runs `run.sh` for real on made-up data and is skipped unless you set `RUN_SLOW=1` (CI sets it). CI runs on Ubuntu with Python 3.14 and never downloads the data.
- The one-time test run was `python src/final_test.py --confirm-open-test`. It was run once, after the choices were frozen. It is not part of the pipeline, and a test checks that the pipeline never imports it. A lock file in `data/` makes a second run on my machine refuse. That lock is a guard against my own mistakes. It cannot stop someone who clones the repository fresh, which has no lock.
- Check this README against the saved results: `python src/check_readme.py`. It needs the real `outputs/` folder, which is not committed, so it runs on my machine only. CI runs the checker's own tests on made-up results. CI cannot check the numbers below, and that is a limit of this repository.

All settings (split shares, column groups, model settings, tuning grids, cost scenarios) are in `config.yaml`.

## Method in brief

- **Time split.** Coins are ordered by migration time (creation time plus the seconds the coin took to migrate), oldest first, and cut 60% / 20% / 20% into train, validation and test. Earlier coins never see later ones.
- **Scope.** Only coins with at least 100 holders and at least 60 seconds of valid swaps, the authors' own filter. A further 233 coins had no result in the label file and were set aside.
- **Columns.** The model sees 81 columns from groups 2 and 3, whose timing the config records as confirmed from the authors' code. Group 1 (calendar and SOL price) and group 4 (cluster columns) are held back and tested separately. Two calendar columns, `group1_migrate_year` and `group1_migrate_month`, are never used because the audit flags them as clock-like.
- **Models.** Logistic regression, random forest and gradient boosting, from scikit-learn, with seed 42.
- **Two sides.** A flag means "likely to crash". Every ranking score is shown for both sides: the crash side (finding crashers) and the survivor side (finding coins that survive). Each is read next to the lazy guess of the same pile: the share of crashers for the crash side, the share of survivors for the survivor side.
- **Two mistakes.** From a trader's point of view, *skipped good* means flagging a coin that would have survived, and *held crasher* means not flagging a coin that crashes. The cut-off is the score at or above which a coin is flagged.
- **Wobble.** 200 re-drawn copies of a pile (seed 42) give a middle-90% range for each score. A difference counts only if its range does not include 0. The wobble measures sampling luck inside one pile. It does not measure change from one period to the next.
- **Tuning.** Time-ordered, on the train pile only: train is cut into 4 blocks, each trial learns from the earlier blocks and is graded on the next, and coins whose result was not yet known when a graded block starts are dropped. Small grids per model.
- **Choice rule.** The model with the highest average of the crash-side and survivor-side PR-AUC on validation, using the tuned settings. A tie goes to the earlier candidate.
- **Cut-offs.** For each invented cost scenario, the cut-off with the lowest total cost on validation. A tie goes to the higher cut-off.
- **Freezing.** The model, its settings and the three cut-offs were written to `frozen_choice.yaml` and committed before the test pile was opened. The test step reads them from that file and chooses nothing.

## Results

Every table below is printed from `outputs/*.json` (and, for the first one, from the facts in `config.yaml` that `prepare.py` reproduces on the real data before it saves anything). `python src/check_readme.py` checks every table row as a whole, so a number under the wrong label is caught. Apart from the gap quoted in the summary, the numbers in the sentences are not machine-checked.

### 1. The data and the three piles

| Measure | Value |
|---|---|
| Coins with a result | 41,470 |
| Coins set aside (no result in the label file) | 233 |
| Coins kept by the scope filter | 21,635 |
| Crash share of the kept coins | 63.2% |
| Train pile: first and last migration (UTC) | 2024-12-09 22:33 to 2025-01-24 14:23 |
| Validation pile: first and last migration (UTC) | 2025-01-24 14:24 to 2025-02-06 07:13 |
| Test pile: first and last migration (UTC) | 2025-02-06 07:13 to 2025-02-28 23:43 |

| Pile | Coins | Crash share (lazy guess for the crash side) | Survivor share (lazy guess for the survivor side) |
|---|---|---|---|
| Train | 12,981 | 61.8% | 38.2% |
| Validation | 4,327 | 71.3% | 28.7% |
| Test | 4,327 | 59.2% | 40.8% |

The lazy guess for each side is the share of that kind of coin in the same pile. The crash share moves from pile to pile (61.8%, 71.3%, 59.2%), so scores are always read as multiples of their own pile's lazy guess.

### 2. The three models before tuning, on validation

| Model | Validation, crash side: PR-AUC (x lazy guess) | Validation, survivor side: PR-AUC (x lazy guess) | Train, crash side | Train, survivor side | Validation precision, top 10%, crash side | Validation precision, top 10%, survivor side |
|---|---|---|---|---|---|---|
| Logistic regression | 0.808 (1.13x) | 0.429 (1.49x) | 0.799 (1.29x) | 0.627 (1.64x) | 0.852 | 0.538 |
| Random forest | 0.823 (1.15x) | 0.478 (1.66x) | 0.925 (1.50x) | 0.830 (2.17x) | 0.901 | 0.566 |
| Gradient boosting | 0.826 (1.16x) | 0.481 (1.68x) | 0.909 (1.47x) | 0.806 (2.11x) | 0.878 | 0.624 |
| Lazy guess | 0.713 | 0.287 | 0.618 | 0.382 | - | - |

The random forest and gradient boosting score far higher on training than on validation (for example, the random forest's survivor side is 0.830 on train and 0.478 on validation). That is overfitting. The training score is not an honest grade.

### 3. How much the validation scores wobble (untuned)

| Model | Crash PR-AUC, middle 90% | Crash: difference vs logistic regression | Survivor PR-AUC, middle 90% | Survivor: difference vs logistic regression |
|---|---|---|---|---|
| Logistic regression, untuned | 0.808 (0.794 to 0.823) | reference | 0.430 (0.406 to 0.453) | reference |
| Random forest, untuned | 0.822 (0.807 to 0.839) | +0.014 (+0.006 to +0.023), better | 0.478 (0.454 to 0.508) | +0.049 (+0.034 to +0.063), better |
| Gradient boosting, untuned | 0.826 (0.811 to 0.841) | +0.018 (+0.011 to +0.025), better | 0.481 (0.458 to 0.508) | +0.052 (+0.037 to +0.069), better |

### 4. Time-ordered tuning, before and after

| Model | Best settings tried | Time-ordered tuning score: best (x lazy guess, average of both sides) | Validation crash PR-AUC: untuned to tuned | Validation survivor PR-AUC: untuned to tuned |
|---|---|---|---|---|
| Logistic regression | {C: 0.1} | 1.412 (worst trial 1.408, 4 trials) | 0.808 to 0.809 | 0.429 to 0.434 |
| Random forest | {max_depth: None, min_samples_leaf: 20} | 1.451 (worst trial 1.433, 6 trials) | 0.823 to 0.823 | 0.478 to 0.478 |
| Gradient boosting | {learning_rate: 0.03, max_iter: 100, max_leaf_nodes: 15} | 1.440 (worst trial 1.408, 8 trials) | 0.826 to 0.814 | 0.481 to 0.480 |

Tuning changed little. The largest move is gradient boosting's crash side, which got worse (0.826 to 0.814). The wobble of the tuned models:

| Model | Crash PR-AUC, middle 90% | Crash: difference vs logistic regression | Survivor PR-AUC, middle 90% | Survivor: difference vs logistic regression |
|---|---|---|---|---|
| Logistic regression, tuned | 0.809 (0.794 to 0.822) | reference | 0.437 (0.410 to 0.461) | reference |
| Random forest, tuned | 0.822 (0.807 to 0.839) | +0.014 (+0.005 to +0.023), better | 0.478 (0.454 to 0.508) | +0.044 (+0.029 to +0.057), better |
| Gradient boosting, tuned | 0.815 (0.799 to 0.831) | +0.005 (-0.002 to +0.015), cannot be told apart | 0.479 (0.456 to 0.509) | +0.045 (+0.030 to +0.059), better |

### 5. Column tests (group 1 and group 4)

Each extra group is added on top of the 81 columns, using the tuned gradient boosting. The columns `group1_migrate_year` and `group1_migrate_month` are never added.

| Setup | Columns | Crash PR-AUC | Crash: change vs reference | Survivor PR-AUC | Survivor: change vs reference | Caution |
|---|---|---|---|---|---|---|
| reference | 81 | 0.814 | - | 0.480 | - | - |
| plus_group1 | 85 | 0.816 | +0.001 (-0.001 to +0.004), cannot be told apart | 0.486 | +0.006 (-0.002 to +0.014), cannot be told apart | - |
| plus_group4 | 116 | 0.821 | +0.007 (+0.003 to +0.011), better | 0.485 | +0.006 (-0.001 to +0.012), cannot be told apart | possible leakage or unverified timing |
| plus_group1_and_group4 | 120 | 0.825 | +0.010 (+0.006 to +0.015), better | 0.485 | +0.004 (-0.003 to +0.015), cannot be told apart | possible leakage or unverified timing |

- The two group-4 setups lift the crash side (+0.007 and +0.010, the ranges exclude 0). Survivor-side changes cannot be told apart from 0 anywhere.
- The audit lists group 4 as having unverified timing (see below), so these gains count as possible leakage, not a win. Nothing was added to the model.
- With this many comparisons across the project (column tests, model comparisons, tuning), a small gain can be luck.

### 6. Leakage audit (training pile only)

| Measure | Value | Columns |
|---|---|---|
| Columns audited | 122 | - |
| Columns with unverified timing | 36 | - |
| Migration time alone, ranking AUC (0.5 = no better than chance) | 0.5433 | - |
| Flagged extreme_bin | 0 | - |
| Flagged mostly_zero | 0 | - |
| Flagged constant | 0 | - |
| Flagged strong_alone | 0 | - |
| Flagged clock_like | 2 | group1_migrate_year, group1_migrate_month |

Each column is checked for a band of values with a 0% or 100% crash rate, for mostly-zero or constant values, for ranking crashes very well or very badly alone, and for moving with migration time. Two columns are flagged clock-like and are never used. The audit also scores migration time alone: an AUC of 0.5433, a little above chance, meaning that when a coin migrated carries a small amount of information by itself. The audit lists 36 of the 122 columns as having timing I could not verify. The written reasons for every flagged column are in `decision_log.md`.

### 7. The model choice, on validation

| Tuned model | Validation crash PR-AUC | Validation survivor PR-AUC | Average (the choice rule) |
|---|---|---|---|
| Logistic regression | 0.809 | 0.434 | 0.621 |
| Random forest | 0.823 | 0.478 | 0.650 |
| Gradient boosting | 0.814 | 0.480 | 0.647 |

The random forest has the highest average. Its gap over the runner-up is 0.0033. The choice follows the rule, not a judgement that the random forest is clearly better.

### 8. Cut-offs chosen on validation (invented costs)

The costs are unit-free and invented: for each scenario, what holding a crasher and what skipping a good coin is assumed to cost. They show how the best cut-off moves. They are not measured from any trading.

| Scenario (invented costs) | Cut-off | Share of coins flagged | Skipped good coins | Held crashers | Cost per coin: this cut-off | Cost per coin: flag everything | Cost per coin: flag nothing |
|---|---|---|---|---|---|---|---|
| equal (holding a crasher 1, skipping a good coin 1) | 0.3958 | 87.5% | 929 | 229 | 0.2676 | 0.2873 | 0.7127 |
| crasher_costs_3x (holding a crasher 3, skipping a good coin 1) | 0.1591 | 99.6% | 1,228 | 2 | 0.2852 | 0.2873 | 2.1382 |
| skipped_good_costs_3x (holding a crasher 1, skipping a good coin 3) | 0.5919 | 57.7% | 473 | 1,059 | 0.5727 | 0.8618 | 0.7127 |

Flag everything and flag nothing are the two lazy options, and the cost of the chosen cut-off is shown next to both. In the "holding a crasher costs 3x" scenario the best cut-off flags nearly every coin, so it is close to the lazy option "flag everything".

### 9. The one-time test run: ranking quality

The test pile was opened once, with the frozen model fitted on the train pile only. The reference model is logistic regression.

| Row | Crash side: PR-AUC (x lazy guess) | Survivor side: PR-AUC (x lazy guess) |
|---|---|---|
| Random forest on validation | 0.823 (1.15x) | 0.478 (1.66x) |
| Random forest on test | 0.737 (1.24x) | 0.615 (1.51x) |
| Logistic regression on test (reference) | 0.732 (1.24x) | 0.589 (1.44x) |
| Lazy guess on test | 0.592 | 0.408 |

| Model | Crash PR-AUC, middle 90% | Crash: difference vs logistic regression | Survivor PR-AUC, middle 90% | Survivor: difference vs logistic regression |
|---|---|---|---|---|
| Random forest, test pile | 0.738 (0.721 to 0.754) | +0.006 (-0.006 to +0.018), cannot be told apart | 0.614 (0.594 to 0.637) | +0.024 (+0.013 to +0.038), better |
| Logistic regression, test pile | 0.732 (0.716 to 0.750) | reference | 0.588 (0.570 to 0.611) | reference |

The wobble covers sampling noise inside the test pile. It does not cover change between periods.

### 10. The one-time test run: the frozen cut-offs

| Scenario (invented costs) | Cut-off (frozen) | Share flagged | Skipped good coins | Held crashers | Cost per coin: frozen cut-off | Cost per coin: flag everything | Cost per coin: flag nothing | Same cut-off on validation | Saving per coin vs the cheaper lazy option (middle 90%) |
|---|---|---|---|---|---|---|---|---|---|
| equal | 0.3958 | 81.0% | 1,221 | 278 | 0.3464 | 0.4077 | 0.5923 | 0.2676 | +0.0604 (+0.0497 to +0.0712), cheaper in 100% |
| crasher_costs_3x | 0.1591 | 98.9% | 1,720 | 4 | 0.4003 | 0.4077 | 1.7770 | 0.2852 | +0.0074 (+0.0039 to +0.0106), cheaper in 100% |
| skipped_good_costs_3x | 0.5919 | 49.0% | 569 | 1,012 | 0.6284 | 1.2230 | 0.5923 | 0.5727 | -0.0359 (-0.0664 to -0.0092), cheaper in 2% |

| Scenario | Crash precision | Crash recall | Survivor precision | Survivor recall |
|---|---|---|---|---|
| equal | 65.2% | 89.1% | 66.1% | 30.8% |
| crasher_costs_3x | 59.8% | 99.8% | 91.7% | 2.5% |
| skipped_good_costs_3x | 73.2% | 60.5% | 54.1% | 67.7% |

- Equal costs: the frozen cut-off was cheaper than both lazy options in nearly every re-drawn exam (100% after rounding). It flags most coins and keeps only 30.8% of the survivors.
- "Holding a crasher costs 3x": cheaper than flagging everything by a very small amount, because it nearly is flagging everything.
- "Skipping a good coin costs 3x": costlier than flagging nothing (0.6284 against 0.5923). The validation cost of 0.5727 did not carry over.

## Numbers that could mislead

- **The crash side looks good because most coins crash.** The lazy guess is already 0.59 to 0.71 there. Read the multiples (1.15x on validation, 1.24x on test), not the raw scores.
- **The validation pile was used for choices.** The model choice, the tuning judgement and the cut-offs were all picked on validation, so validation scores are somewhat optimistic. The test pile was used only to grade the frozen choices, so the test numbers carry no selection from the test pile.
- **The cut-off result is not proven to be a shift.** The cut-offs did worse on test, and the crash share also fell from 71.3% to 59.2%. A shift is one possible reason. A cut-off tuned too closely to validation is another. This project cannot separate them.
- **The model choice is not a clear win.** The gap between the top two models is 0.0033, and their wobble ranges overlap.
- **The wobble is narrower than the real uncertainty.** It measures luck inside one pile, not change from one period to the next, which the test run shows is real.
- **Training scores are inflated** for the random forest and gradient boosting.
- **Many comparisons.** A small gain in the column tests or tuning can be luck.
- **The cost scenarios are assumptions.** No profit claim is made, and nothing here says how a flag would do in live trading.
- **The audit cannot prove timing.** Thirty-six columns have timing I could not verify. The group-4 gains above are treated as possible leakage for that reason.

## Assumptions and limits

- The crash label and the scope filter come from the dataset's authors. A coin outside the filter is not modelled: of 41,470 coins with a result, 21,635 are kept.
- One dataset and one period, migrations from 9 December 2024 to 28 February 2025 (UTC). Nothing here says how the flag would work in another period, and it was not tried on new coins.
- A crash is defined by the first 20 minutes after migration. It says nothing about what happens later.
- The costs, and the three cost scenarios, are invented.
- Validation was reused for several choices, as described above.
- `decision_log.md` records the choices and the reasons, and `recommendation.md` gives plain-language advice on how the flag should and should not be used, and what to check before relying on it.

## Documents

- `decision_log.md`: the questions, the key decisions and the reasons, including each flagged column.
- `model_note.md`: how the model was trained, evaluated, tuned and chosen, for a data scientist.
- `recommendation.md`: one page for a non-technical reader.
- `learning_log.md`: what was learned.

## AI tools used

- Claude (Anthropic) was used as a trainer and code assistant. It wrote the code, the tests and this README, working from my instructions and reading my results. I ran everything on my machine and made my own commits, branches and pull requests. `decision_log.md`, `model_note.md`, `recommendation.md` and `learning_log.md` were drafted with Claude from my project outputs and reviewed by me.
- An automated GitHub reviewer (the OpenAI Codex connector, `chatgpt-codex-connector`) may post a review summary on a pull request. Where it did, each finding was checked against the code.
- The raw chat history is not included in this project.
