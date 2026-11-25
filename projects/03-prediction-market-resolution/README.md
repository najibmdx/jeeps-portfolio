# Project 03: Prediction Market Resolution

Learning project. No claim about trading use, returns or profit.

**Status: Stage 10. The project is built end to end and the one-time test has been run. The dataset was built, split by time and checked for leakage and shift; three baselines and an untuned model ladder were scored on validation; the ladder was tuned lightly and one model was chosen on validation and written to `frozen_choice.yaml`; the chosen model was checked with column groups left out; then the test split was opened once. Nothing was retuned or refitted afterwards.**

## The question

Polymarket markets trade at a price that already works as a forecast. Given what a market's price and trading looked like in its first seven days, how well can a model forecast whether it resolves Yes, compared with the market's own price? Forecasts are probabilities and are scored by log loss.

## Result in short

On the test split the frozen model cannot be told apart from the market's own price, in the contested slice (the headline) and on all markets. A forecast that ignores the price (the train Yes rate for every market) is clearly worse. Validation gives the same reading. In plain words: in this test window, the extra columns added nothing measurable to what the price already says. The contested slice is the markets whose last price before the decision time was between 0.05 and 0.95. Every number is in Results below, and "Numbers that could mislead" says how far to trust them.

## Data

`vgregoire/polymarket-users` on Hugging Face (Akey, Gregoire, Harvie and Martineau, 2026), licence CC-BY-4.0 for the authors' processing. Fields that come from the Polymarket API stay under Polymarket's terms. The data is not stored in this repo. Four tables are used: events, markets, predictions and daily price bars.

## How to run

On made-up data (no download needed):

    python src/pipeline.py --sample

On real data, download the four files into `data/` (about 310 MB, each checked against a size and sha256), check the layout, then audit them against the numbers in `config.yaml`:

    python src/download.py
    python src/pipeline.py
    python src/audit.py

`pipeline.py` checks the layout, builds the dataset into `data/derived/`, then splits it by decision time into train, validation and test files in the same folder. Markets of one event stay together, and train and validation markets whose outcome was not yet known when the next period began are dropped. Nothing reads the test file.

`python src/leakage.py` then prints the leakage audit: a table of every column with its source and whether it is known at the decision time, and four screens. Three use the train part only (groups where the outcome never varies, features that move with the calendar, and the final-bar leak, cited from the audit). The fourth compares the features of train with those of validation, without any outcome, to show which inputs look different in the later period. A flag is a finding to look at, not proof of a leak, and nothing is dropped automatically. The test file is never read.

`python src/baselines.py` scores the three baselines on the validation part only: the raw price at the decision time taken as the probability of Yes, the same price passed through a two-number correction (intercept and slope on the log-odds) fitted on train, and the train Yes rate for every market. Log loss decides, Brier is shown beside it, for all markets and for the contested slice. Every forecast is clipped into 0.001 to 0.999 before scoring so that a price of exactly 0 or 1 cannot give an infinite score. A calibration table shows, for fixed price bands, how often markets really resolved Yes. The ranges come from re-drawing the validation set 200 times by resampling whole events; they show sampling noise only, not the change between periods. The test file is never read.

`python src/models.py` fits the untuned model ladder on train and scores it on validation next to the baselines: the price-only correction, a logistic regression on all inputs, a random forest and a gradient-boosting model. Every model uses library defaults and seed 42 (only the forest size and the logistic iteration cap are set). The inputs are the market features known at the decision time, one-hot category and no `has_fee` (about none of the train and validation markets have a fee, so nothing can be learned from it). The report shows train and validation scores side by side so overfitting is visible, and compares every model with the raw price on the same resampled validation sets. Nothing is tuned and nothing is chosen at this stage. The test file is never read.

`python src/tune.py` tunes the ladder lightly. Each tunable model gets the same budget of 6 settings (the regularisation strength of the logistic regression, the leaf size of the forest, the learning rate and leaf size of boosting), and every grid includes the library default, so the best setting cannot be worse than the untuned model on validation. Every setting is fitted on train and scored on validation, for all markets and for the contested slice, next to its train score. The choice is the lowest validation log loss on all markets, compared at 4 decimals; a tie goes to the simpler setting, then to the earlier model in the ladder. The winner's margin over the runner-up is shown next to its wobble, and the wobble never decides. Only the four ladder models can win; the raw price is shown beside them as the yardstick. The chosen model stays fitted on train only. Its validation score is slightly flattering, because it was picked on those markets; the one-time test is the honest number. The test file is never read.

`python src/freeze.py` then writes `frozen_choice.yaml` (model, settings, seed, validation scores, margin over the runner-up, date) from that report. It holds no test number, it is committed before the test split is opened, and it refuses to overwrite an existing choice or to run after the test has been opened.

`python src/ablate.py` asks which groups of columns the frozen model needs. It takes the model and settings from `frozen_choice.yaml` and refits it on train with some columns left out: the price path alone, the price path plus the activity columns, plus the category, plus the sibling count, everything except the sibling count, and everything. Nothing is retuned, so the rows differ only in their columns. Each row is compared with the price-only correction and with the others in pairs, on the same resampled validation sets, and the full row must reproduce the validation score written in `frozen_choice.yaml` or the run stops. It scores validation only, never reads the test file, and cannot change the frozen choice. The sibling count rises over time, so a gain from it could partly be the calendar.

`python src/final_test.py --confirm-open-test` is the one-time test. Without the flag it only prints its plan and reads nothing. With it, the script first checks that `frozen_choice.yaml` is committed and unchanged, that the test split has the agreed 8,991 markets, and that the frozen model, refitted on train only, reproduces its validation score. Only then does it write a lock file, read the test split once, and score the frozen model against the market price, the price-only correction and the base rate, on all markets and on the contested slice, with the same resampled ranges as validation. The reading of the result is fixed in `config.yaml` before the run, and the validation numbers are printed beside it. Nothing is retuned afterwards, and a second run is refused. It was run once, on 25 Nov 2026; the saved report is `outputs/final_test_k7.json`, the only file in `outputs/` that is committed. `frozen_choice.yaml` still says the test split was not opened, because it was written and committed before the run and the run checks that it is unchanged; the date of the run is in `final_test_k7.json`.

`python src/check_readme.py` checks every number in the Results section against the saved files in `outputs/`, row by row, and also checks that each saved reading ("worse", "cannot be told apart") agrees with its own range. `python src/check_readme.py --update` rewrites the Results section from those files, so no number in it is typed by hand. Files that are missing are skipped with a note: `outputs/` is not committed, apart from `final_test_k7.json`. The check itself is tested on made-up results.

## Results

Learning project. No claim about trading use, returns or profit. All numbers below are written by `python src/check_readme.py --update` from the saved files in `outputs/`. Log loss is lower-is-better. A "range" is the 90% range from re-drawing the split 200 times by resampling whole events. A difference is "worse" or "better" only when its range lies wholly on one side of zero; otherwise it is "cannot be told apart". The validation tables come from fits on train only. The one-time test is the last three tables and the two sentences before them.

<!-- results:begin (written by src/check_readme.py --update; do not edit by hand) -->

**Markets and events per split**

| Split | Markets | Events |
|---|---|---|
| Train | 23,897 | 7,156 |
| Validation | 7,374 | 1,213 |
| Test | 8,991 | 1,336 |

**The three windows side by side**

| Split | Markets | Decision dates (UTC) | Median days from decision to close | Closing within a day | Resolved Yes | Contested |
|---|---|---|---|---|---|---|
| Train | 23,897 | 2023-02-23 to 2025-12-22 | 3.39 | 39.9% | 20.5% | 44.6% |
| Validation | 7,374 | 2025-12-23 to 2026-02-08 | 0.86 | 66.1% | 16.8% | 26.8% |
| Test | 8,991 | 2026-02-09 to 2026-03-30 | 0.86 | 62.5% | 18.2% | 27.0% |

**Baselines, validation**

| Slice | Forecast | Log loss | Range (90%) | Brier | Log loss minus raw price | Reading |
|---|---|---|---|---|---|---|
| All markets | Raw market price | 0.1531 | 0.1426 to 0.1651 | 0.0465 | - | - |
| All markets | Price-only correction | 0.1534 | 0.1428 to 0.1656 | 0.0466 | +0.0003 (0.0000 to +0.0007) | worse |
| All markets | Base rate (train Yes rate) | 0.4569 | 0.4388 to 0.4728 | 0.1410 | +0.3038 (+0.2851 to +0.3216) | worse |
| Contested | Raw market price | 0.4711 | 0.4444 to 0.4961 | 0.1530 | - | - |
| Contested | Price-only correction | 0.4718 | 0.4444 to 0.4976 | 0.1533 | +0.0007 (-0.0003 to +0.0018) | cannot be told apart |
| Contested | Base rate (train Yes rate) | 0.7238 | 0.6943 to 0.7612 | 0.2573 | +0.2527 (+0.2204 to +0.2830) | worse |

**Untuned model ladder (log loss; every model fitted on train)**

| Model | Train, all | Validation, all | Validation minus raw price, all | Reading, all | Validation, contested | Validation minus raw price, contested | Reading, contested |
|---|---|---|---|---|---|---|---|
| Raw market price | 0.2335 | 0.1531 | - | - | 0.4711 | - | - |
| Base rate (train Yes rate) | 0.5078 | 0.4569 | +0.3038 (+0.2851 to +0.3216) | worse | 0.7238 | +0.2527 (+0.2204 to +0.2830) | worse |
| Price-only correction | 0.2333 | 0.1534 | +0.0003 (0.0000 to +0.0007) | worse | 0.4718 | +0.0007 (-0.0003 to +0.0018) | cannot be told apart |
| Logistic regression, all inputs | 0.2329 | 0.1533 | +0.0002 (-0.0003 to +0.0008) | cannot be told apart | 0.4722 | +0.0012 (-0.0005 to +0.0030) | cannot be told apart |
| Random forest | 0.0612 | 0.1597 | +0.0066 (+0.0040 to +0.0095) | worse | 0.4867 | +0.0157 (+0.0085 to +0.0228) | worse |
| Gradient boosting | 0.1983 | 0.1572 | +0.0040 (+0.0022 to +0.0060) | worse | 0.4804 | +0.0094 (+0.0039 to +0.0160) | worse |

**Light tuning (validation log loss of each model's best setting)**

| Model | Settings tried | Chosen setting | Validation, all | Library default, all | Validation, contested |
|---|---|---|---|---|---|
| Price-only correction | 1 | none (nothing to tune) | 0.1534 | 0.1534 | 0.4718 |
| Logistic regression, all inputs | 6 | C=1.0 | 0.1533 | 0.1533 | 0.4722 |
| Random forest | 6 | min_samples_leaf=50 | 0.1550 | 0.1597 | 0.4738 |
| Gradient boosting | 6 | learning_rate=0.1, min_samples_leaf=400 | 0.1550 | 0.1572 | 0.4762 |

**The choice made on validation**

| Measure | Value |
|---|---|
| Chosen on validation | Logistic regression, all inputs, C=1.0 |
| Runner-up | Price-only correction |
| Winner, validation log loss, all markets | 0.1533 |
| Runner-up minus winner, all markets | +0.0001 (-0.0006 to +0.0007), cannot be told apart |
| Winner minus raw price, all markets | +0.0002 (-0.0003 to +0.0008), cannot be told apart |
| Winner minus raw price, contested | +0.0012 (-0.0005 to +0.0030), cannot be told apart |

**Column groups left out of the frozen model (validation log loss)**

| Columns used | Inputs | Validation, all | Minus raw price, all | Reading, all | Validation, contested | Minus raw price, contested | Reading, contested |
|---|---|---|---|---|---|---|---|
| Raw market price | - | 0.1531 | - | - | 0.4711 | - | - |
| Price-only correction | - | 0.1534 | +0.0003 (0.0000 to +0.0007) | worse | 0.4718 | +0.0007 (-0.0003 to +0.0018) | cannot be told apart |
| Price path only | 6 | 0.1533 | +0.0002 (-0.0003 to +0.0007) | cannot be told apart | 0.4725 | +0.0014 (-0.0001 to +0.0031) | cannot be told apart |
| Price path + activity | 10 | 0.1536 | +0.0004 (0.0000 to +0.0010) | cannot be told apart | 0.4732 | +0.0022 (+0.0006 to +0.0041) | worse |
| Price path + category | 7 | 0.1533 | +0.0002 (-0.0003 to +0.0008) | cannot be told apart | 0.4725 | +0.0015 (-0.0002 to +0.0034) | cannot be told apart |
| Price path + sibling count | 7 | 0.1531 | 0.0000 (-0.0005 to +0.0006) | cannot be told apart | 0.4716 | +0.0006 (-0.0012 to +0.0025) | cannot be told apart |
| All except sibling count | 11 | 0.1535 | +0.0004 (-0.0002 to +0.0010) | cannot be told apart | 0.4730 | +0.0020 (+0.0001 to +0.0039) | worse |
| All columns (the frozen model) | 12 | 0.1533 | +0.0002 (-0.0003 to +0.0008) | cannot be told apart | 0.4722 | +0.0012 (-0.0005 to +0.0030) | cannot be told apart |

Pairwise comparisons in the ablation: 28 (both slices); flagged as different: 2 (A better: 0, A worse: 2).

**Ablation pairs flagged as different (A minus B; negative means A has the lower log loss)**

| Slice | A | B | A minus B, log loss | Reading |
|---|---|---|---|---|
| All markets | Price path + activity | Price path only | +0.0003 (+0.0001 to +0.0005) | worse |
| Contested | Price path + activity | Price path only | +0.0007 (+0.0001 to +0.0014) | worse |

The test split was opened once, at 2026-11-25 02:30 UTC, on 8,991 markets in 1,336 events.

Reading, fixed before the run: **No measurable difference from the market price in this test window.** Validation and test agree.

**Headline: frozen model against the market price**

| Log loss difference (positive means the frozen model is worse) | Test window | Validation window |
|---|---|---|
| Contested (the headline slice): frozen model minus raw market price | +0.0003 (-0.0013 to +0.0021), cannot be told apart | +0.0012 (-0.0005 to +0.0030), cannot be told apart |
| Contested (the headline slice): frozen model minus price-only correction | +0.0002 (-0.0014 to +0.0020), cannot be told apart | +0.0004 (-0.0015 to +0.0030), cannot be told apart |
| All markets: frozen model minus raw market price | -0.0001 (-0.0006 to +0.0004), cannot be told apart | +0.0002 (-0.0003 to +0.0008), cannot be told apart |
| All markets: frozen model minus price-only correction | +0.0001 (-0.0005 to +0.0006), cannot be told apart | -0.0001 (-0.0007 to +0.0006), cannot be told apart |

**One-time test: all four forecasts**

| Slice | Forecast | Log loss | Range (90%) | Brier | Log loss minus raw price | Reading |
|---|---|---|---|---|---|---|
| All markets | Raw market price | 0.1504 | 0.1401 to 0.1601 | 0.0469 | - | - |
| All markets | Price-only correction | 0.1503 | 0.1398 to 0.1599 | 0.0468 | -0.0002 (-0.0004 to 0.0000) | cannot be told apart |
| All markets | Base rate (train Yes rate) | 0.4759 | 0.4618 to 0.4885 | 0.1493 | +0.3255 (+0.3115 to +0.3405) | worse |
| All markets | Frozen model (logistic regression, all inputs, C=1.0) | 0.1503 | 0.1400 to 0.1602 | 0.0469 | -0.0001 (-0.0006 to +0.0004) | cannot be told apart |
| Contested | Raw market price | 0.4969 | 0.4781 to 0.5152 | 0.1632 | - | - |
| Contested | Price-only correction | 0.4969 | 0.4778 to 0.5159 | 0.1630 | 0.0000 (-0.0006 to +0.0007) | cannot be told apart |
| Contested | Base rate (train Yes rate) | 0.6933 | 0.6703 to 0.7151 | 0.2440 | +0.1965 (+0.1713 to +0.2234) | worse |
| Contested | Frozen model (logistic regression, all inputs, C=1.0) | 0.4971 | 0.4780 to 0.5155 | 0.1632 | +0.0003 (-0.0013 to +0.0021) | cannot be told apart |

**Calibration of the frozen model on the test split, all markets**

| Forecast band | Markets | Mean forecast | Share that resolved Yes |
|---|---|---|---|
| 0 to 0.05 | 5,847 | 0.006 | 0.004 |
| 0.05 to 0.1 | 401 | 0.073 | 0.065 |
| 0.1 to 0.2 | 512 | 0.147 | 0.191 |
| 0.2 to 0.3 | 342 | 0.246 | 0.254 |
| 0.3 to 0.5 | 420 | 0.393 | 0.343 |
| 0.5 to 0.7 | 311 | 0.591 | 0.543 |
| 0.7 to 0.95 | 378 | 0.835 | 0.823 |
| 0.95 and above | 780 | 0.989 | 0.994 |

<!-- results:end -->

## Numbers that could mislead

- **Many comparisons.** The ablation alone makes the number of pairwise comparisons shown above, and every row is also compared with the price. At a 90% range, about one comparison in ten is flagged by chance even when nothing differs, so a few flags among this many are expected. The flagged pairs say that adding the activity columns scored worse than the price path alone; no comparison in the ablation shows a column group helping.
- **One window.** The test split is one stretch of weeks (see the window table). A different period could read differently. A result for one window is not a result about prediction markets in general.
- **The ranges cover sampling noise only.** They come from resampling whole events inside one split. They do not cover a change of behaviour between periods, which is the larger worry here.
- **The windows differ.** Train looks different from validation and test: its markets close later and more of them are contested (see the window table). Validation and test look alike. The model was fitted on the population that differs most from the one it was scored on.
- **The test leans to quick-closing markets.** Markets still open when the data ends are absent, so the test split leans towards markets that close quickly. A model could behave differently on long-lived markets.
- **"Cannot be told apart" is not "equal".** It means this data cannot separate the two forecasts. A small gain or loss inside the range is possible in either direction.
- **Rounding.** Numbers are shown to four decimals. A range that reads 0.0000 at one end can sit just above or just below zero, and the reading uses the unrounded range.
- **Validation numbers of the chosen model are slightly flattering.** The model was picked on those markets. The one-time test is the honest number.
- **Calibration has no ranges.** The calibration table has eight bands, some with few markets. A visible gap in one band (for example 0.1 to 0.2) is one of eight and has the same sampling noise as everything else.
- **Fee markets.** A fee flag was left out of the inputs because almost no train or validation market has one. Almost all fee markets in the data sit in the test period, so a fee effect could not be learned or checked.

## Limits

- The inputs are the first seven days of daily price bars and trading counts, plus category and the number of sibling markets in the event. No text, no order book, no information about who trades.
- One decision time (seven days after a market opens), one forecast target (does Yes win), one scoring rule (log loss, with Brier beside it).
- Four models, six or fewer settings each, tuned lightly. A better model, or a better setting, may exist; this project does not claim otherwise.
- The market price is taken as a probability as it stands. Fees, spreads, costs and execution are not modelled, and nothing here says anything about trading. No claim about trading use, returns or profit.
- The test split was opened once and the script refuses a second run (it writes a lock file), so the test result cannot be repeated or averaged. The model was not refitted on train plus validation, so it was fitted on the oldest data only.
- The data is a third-party dataset under its own licence (see Data). The project reads it; it does not clean up or correct it.

Tests: `python -m pytest -q`
