# Learning log: Project 01, Bank Call Targeting

These are things this project taught me, with the evidence from the project.

## 1. A column that is only known after the event is leakage
`duration` (the call length) lifted validation PR-AUC by +0.348 to +0.373. That looked like skill but was leakage, because the call length is only known after the call. I excluded it from every real model and kept one labelled demonstration.

## 2. Split by time, because the model should only learn from the past
A random split would let later months leak into training. I split by time: train May 2008 to Apr 2009, validation May 2009, test Jun 2009 to Nov 2010.

## 3. Compare with a blind guess
A PR-AUC means little on its own. On validation the blind guess was 0.0905 and on test 0.445, so the test score of 0.607 is only 1.36x the guess. I read every score next to its own blind guess.

## 4. Tuning does not always help
Tuning moved logistic regression by +0.006, changed the random forest by nothing clear, and made gradient boosting worse (-0.021). I reported that as it came out.

## 5. A rule chosen on one month can fail on another
The cut-offs chosen on validation lost to "phone everyone" on test in all three cost scenarios. The yes-rate was 9.05% in validation and 44.50% in test. I did not find out why, and I wrote that down instead of guessing.

## 6. Many comparisons can produce luck
With 15 ablation comparisons, a small gain (such as adding `campaign`) could be chance. Nothing was added to the final model because of it.

## 7. Label what is invented
The call costs are invented. I labelled them as assumptions everywhere, and I challenged an unlabelled invented number from the assistant. Made-up numbers are always labelled.

## What I would do next
Investigate the yes-rate jump, get real call costs and capacity, and re-test on more months.
