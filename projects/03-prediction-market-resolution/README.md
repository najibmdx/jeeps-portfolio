# Project 03: Prediction Market Resolution

Learning project. No claim about trading use, returns or profit.

**Status: Stage 9 (dataset built, split by time, checked for leakage and shift; three baselines and an untuned model ladder scored on validation; the ladder tuned lightly and one model chosen on validation; the chosen model checked with column groups left out; the one-time test step is built). The chosen model is written to `frozen_choice.yaml` before the test split is opened. The test split has not been read and there are no final results to report.**

## The question

Polymarket markets trade at a price that already works as a forecast. Given what a market's price and trading looked like in its first seven days, how well can a model forecast whether it resolves Yes, compared with the market's own price? Forecasts are probabilities and are scored by log loss.

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

`python src/final_test.py --confirm-open-test` is the one-time test. Without the flag it only prints its plan and reads nothing. With it, the script first checks that `frozen_choice.yaml` is committed and unchanged, that the test split has the agreed 8,991 markets, and that the frozen model, refitted on train only, reproduces its validation score. Only then does it write a lock file, read the test split once, and score the frozen model against the market price, the price-only correction and the base rate, on all markets and on the contested slice, with the same resampled ranges as validation. The reading of the result is fixed in `config.yaml` before the run, and the validation numbers are printed beside it. Nothing is retuned afterwards, and a second run is refused.

Tests: `python -m pytest -q`
