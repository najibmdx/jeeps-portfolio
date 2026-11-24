# Project 03: Prediction Market Resolution

Learning project. No claim about trading use, returns or profit.

**Status: Stage 0 (skeleton) only. Nothing is modelled yet and there are no results.**

## The question

Polymarket markets trade at a price that already works as a forecast. Given what a market's price and trading looked like in its first seven days, how well can a model forecast whether it resolves Yes, compared with the market's own price? Forecasts are probabilities and are scored by log loss.

## Data

`vgregoire/polymarket-users` on Hugging Face (Akey, Gregoire, Harvie and Martineau, 2026), licence CC-BY-4.0 for the authors' processing. Fields that come from the Polymarket API stay under Polymarket's terms. The data is not stored in this repo. Four tables are used: events, markets, predictions and daily price bars.

## How to run

On made-up data (no download needed):

    python src/pipeline.py --sample

On real data, put the four files in `data/` first, then:

    python src/pipeline.py

Tests: `python -m pytest -q`
