# Project 03: Prediction Market Resolution

Learning project. No claim about trading use, returns or profit.

**Status: Stage 5 (dataset built, split by time, checked for leakage and shift, and three baselines scored on validation). No model is built yet and there are no results to report.**

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

Tests: `python -m pytest -q`
