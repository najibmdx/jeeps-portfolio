# Model note: Project 01, Bank Call Targeting

Numbers come from my own result files in `outputs/` and from the README.

For a data scientist. This note says how the model was trained, scored, tuned and chosen, and what could be inflated.

## Task and data
- Task: rank bank clients by how likely they are to subscribe to a term deposit, using only what is known before the call.
- Data: UCI Bank Marketing, `bank-additional-full.csv`, 41,188 rows, of which 12 exact duplicates are dropped before splitting. One Portuguese bank, 2008 to 2010.
- The file has no date column. The rows are in date order, so I rebuilt the year from the order of the month names. This relies on the dataset authors' statement that the file starts in May 2008.

## Columns
- A column is allowed only if it is known before the call.
- `duration` is excluded, because it is only known after the call. `campaign` is excluded from the first model because its timing is unclear.
- Month, day of week and the economy columns are held back and tested as ablations, because in this file they act as a clock.
- Every column was audited for leakage on training rows only. A segment is flagged at 50 or more rows and a yes-rate 3x above or below the training yes-rate (6.27%), or exactly 0% or 100%. Both numbers are my assumptions.

## Split
- By time, cut at month ends. Train May 2008 to Apr 2009 (6.27% yes), validation May 2009 (9.05% yes), test Jun 2009 to Nov 2010 (44.50% yes, 4,962 clients).

## Models and scoring
- Logistic regression, random forest and gradient boosting, each inside a pipeline, so the feature steps are learned from the training rows only.
- Main metric: PR-AUC, always read next to the blind-guess score of the same set (0.0905 on validation, 0.0627 on train, 0.445 on test). Also precision, recall and lift in the top 10% of clients, and an "always no" row.
- Wobble: 200 re-drawn copies of the validation month (seed 42) give a middle-90% range. A difference counts only if its range does not include zero.

## Results on validation
- PR-AUC: logistic regression 0.145 (1.60x blind guess), random forest 0.150 (1.66x), gradient boosting 0.143 (1.58x). The three cannot be told apart.
- The forest and boosting score about 3.02x on train but 1.66x on validation, which is overfitting.

## Tuning
- Time-aware: fit on the first 5 months of training, test on the 6th, then grow the window (practice exams 200811, 200812, 200903, 200904). Small grids per model, judged on validation with the wobble ranges.
- Tuning did not clearly help. Logistic regression moved by +0.006 (+0.002 to +0.010), the random forest showed no clear change, and gradient boosting got worse (-0.021).
- The practice-exam scores swing a lot between months. A likely reason is that months differ in yes-rate, but I did not check.

## Choice
- Logistic regression (C = 0.01, max_iter 1000), the simplest of three models that cannot be told apart. It was fitted on training rows only.

## Ablations
- Without history (`pdays`, `previous`, `poutcome`): worse for all three models (-0.029 to -0.040).
- Plus calendar: worse for logistic regression and gradient boosting.
- Plus economy: no clear difference.
- Plus `campaign`: a small gain (+0.003 to +0.006), not added.
- Plus `duration`: +0.348 to +0.373, a deliberate leakage demonstration, never used.
- With 15 comparisons, a small win could be luck.

## Cut-offs
- For each assumed cost scenario, the share of clients to phone was chosen on validation to maximise subscriptions found minus cost times calls: 97% (cheap, 0.03), 40% (moderate, 0.09) and 5% (expensive, 0.15).

## One-time test
- PR-AUC 0.607 (0.589 to 0.628) against a blind guess of 0.445, which is 1.36x. Top 10% (497 calls): precision 0.706, recall 0.159, lift 1.59.
- The cut-offs lost to "phone everyone" in all three scenarios. Net per 100 clients on test: cheap 40.90 against 41.50, moderate 20.24 against 35.50, expensive 2.92 against 29.50.
- The test run refuses a second go.

## What could be inflated or wrong
- The test numbers look high only because the test yes-rate is 44.50%. Read them as multiples of a blind guess.
- Validation is a single month, so the three models being tied could change on another month.
- Training scores for the forest and boosting are inflated by overfitting.
- The wobble ranges measure luck inside one month, not change from month to month, which the test shows is large.
- The cost scenarios are invented. No profit claim.
- The year is rebuilt from the row order.
- I did not investigate why the yes-rate jumped from 9.05% to 44.50%.
