# Model note: Project C, Prediction Market Resolution

Drafted with Claude, reviewed by me. Numbers come from my own result files in `outputs/`. The test was run once, on 25 Nov 2026.

For a data scientist. This note says how the model was trained, scored, tuned and chosen, and what could be inflated.

## Task and data
- Task: forecast whether a Polymarket Yes/No market resolves Yes, using only its first 7 days. Decision time t is the open date plus 7 days. Only daily bars strictly before t are used.
- Data: the `vgregoire/polymarket-users` dataset. Binary markets with exactly one winner, still open at t, with at least 20 trades before t. That gives 43,881 markets from 10,752 events. The label matches `outcome_yes` everywhere, and the build stops if it does not.
- Inputs: 12 columns. Price path (last close, first open, high, low, mean close, change from first to last), activity (bars, days since last bar, trades before t, trades on the last bar) and context (category, sibling count at t). `has_fee` is dropped because only 6 of 23,897 train markets have it.
- Left out on purpose: `close_time`, event volume, event open interest and `events.n_questions`, which are not known at t. `expiration_time` is held back because its timing is unverified.

## Split
- By t, in time order, at the 60% and 80% marks. Boundaries 2025-12-23 and 2026-02-09.
- Each event goes wholly to the split of its earliest market. Markets that close after the next period starts are purged.
- Final sizes: train 23,897, validation 7,374, test 8,991 markets.
- The windows differ. Train has a median of 3.39 days to close and 44.6% contested markets. Validation and test have a median near 0.86 days and about 27% contested.

## Scoring
- Metric: log loss, with Brier score and a calibration table beside it. Every forecast is clipped into [0.001, 0.999].
- Yardstick: the raw price at t. Other baselines are a price-only correction fitted on train and the train base rate.
- Slices: all markets, and "contested" (last price from 0.05 up to but not including 0.95).
- Uncertainty: 200 resamples of whole events, the same draws for every model, so differences against the price are paired. The 90% range decides the verdict. A range wholly above zero is "worse", wholly below is "better", otherwise "cannot be told apart".

## Models and tuning
- Ladder: logistic regression on price only, logistic regression on all inputs, random forest, gradient boosting. Category is one-hot. The logistic regression sees price columns as log-odds.
- Tuning: a small fixed grid per model, with six settings each, on validation only. The library default is in every grid, listed simplest first. A tie goes to the simpler setting.
- The untuned forest overfit heavily (validation gap +0.0985 on all markets). Tuning cut its validation log loss from 0.1597 to 0.1550. Boosting went from 0.1572 to 0.1550. Its best setting sits on the edge of its grid. I did not widen the grid, because that would be tuning on the result.

## Choice and freeze
- Rule: the lowest validation log loss on all markets, compared at four decimals.
- Winner: `lr_all_features`, C=1.0, at 0.1533 on validation. The runner-up, `lr_price_only`, scored 0.1534. The margin is 0.0001 (range -0.0006 to +0.0007), so they cannot be told apart.
- The choice was written to `frozen_choice.yaml` and committed before the test was opened. There was no refit on train plus validation.

## Test (opened once, 25 Nov 2026, 8,991 markets)
- All markets, log loss: price 0.1504, price-only correction 0.1503, frozen model 0.1503, base rate 0.4759 (clearly worse).
- Frozen model minus price: contested +0.0003 (-0.0013 to +0.0021), all markets -0.0001 (-0.0006 to +0.0004). Both cannot be told apart from zero. Validation on contested was +0.0012 (-0.0005 to +0.0030).
- Reading: no measurable difference from the market price in this test window.
- The lock file `data/test_opened.lock` was never deleted and nothing was retuned afterwards.

## Ablations (validation only)
- Six column sets refitted with the frozen settings. There were 28 comparisons, 2 flagged as worse and 0 as better.
- The activity columns hurt slightly (+0.0003 on all markets and +0.0007 on contested). Category adds nothing. Sibling count leans helpful, but every range reaches zero. It also rises with t, so it may only reflect the calendar.
- Nothing from the ablations changed the frozen model.

## What could be inflated or wrong
- The winner's validation score is slightly flattering, because it was picked on those markets.
- The test is a single window (9 Feb to 30 Mar 2026). It leans to quick-closing markets, because markets still open at the data end are absent.
- Train covers a different window from validation and test, so overfit gaps are read against the price's own gap, not against zero.
- The price at t can be up to 7 days stale. Category is assumed fixed at open. Volume units are unconfirmed, so I used trade counts.
- 28 ablation comparisons at a 90% range means a few flags by chance are expected.
- No claim about trading, returns or profit.
