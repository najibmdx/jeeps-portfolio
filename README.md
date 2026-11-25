# Jeeps portfolio

Three small machine-learning projects on public data. Each asks one question and is built to be checked: the split follows time, every column is audited for leakage, the model is chosen before the test data is opened, and the test data is opened once. Results are reported as they come out, including when the model does not beat the simple alternative.

None of these projects makes a claim about trading, returns or profit.

## The projects

| Project | Question | Headline result | Data |
|---|---|---|---|
| [01 Bank call targeting](projects/01-bank-call-targeting) | Which bank clients should a phone team call first? | The models rank clients better than a blind guess, but not by much (validation PR-AUC 0.145 to 0.150 against a blind guess of 0.0905). On the one-time test, the cut-offs chosen on validation lost to "phone everyone" in all three assumed-cost scenarios, because the test yes-rate was 44.50% against 9.05% in validation. | UCI Bank Marketing |
| [02 Memecoin launch risk](projects/02-memecoin-launch-risk) | Will a Solana memecoin crash in the 20 minutes after it migrates? | The random forest ranks coins modestly above a lazy guess (on test, crash side 1.24x and survivor side 1.51x). Of three cut-offs, one lost to flagging nothing on test. | MELT dataset |
| [03 Prediction market resolution](projects/03-prediction-market-resolution) | Can a model beat a Polymarket market's own price as a forecast of how it resolves? | No measurable difference from the price on a one-time test of 8,991 markets. On the contested slice the frozen model minus the price is +0.0003 in log loss (range -0.0013 to +0.0021). | Polymarket data on Hugging Face |

## Where to start

Project 03 is the most recent. Its README has the full method and results, and `recommendation.md` is a one-page summary for a non-technical reader. Each project folder has its own README, with how to run it, its results and a section on numbers that could mislead.

## How each project is built

- **Time split.** Train, validation and test follow time, so a model never learns from the future.
- **Leakage audit.** Every column is checked for whether it is known at prediction time.
- **Model chosen before the test.** The model and its settings are written to a committed file before the test data is opened.
- **Test opened once.** The final test runs once, and the script refuses a second run.
- **Ranges, not single numbers.** Scores come with a 90% range from re-drawing the data 200 times, and a difference counts only when its range does not include zero.
- **Checked README.** Scripts compare the README tables with the saved results, so a typed number cannot drift from the real one.
- **Tests and CI.** Each project has its own tests, which run on every pull request.

## Data and credit

Each project credits its own data source and licence in its README. No data is stored in this repository. Python 3.14 was used.
