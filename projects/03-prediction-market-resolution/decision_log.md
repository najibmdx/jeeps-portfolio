# Decision Log: Project C — Prediction Market Resolution

## 1. Clarifying questions

Before building the model, I identified three questions I could not resolve from the available data. I handled each as follows:

1. **Can a market’s category change after opening?** The category comes from a classifier, but I could not verify whether it changes. I proceeded on the assumption that it is fixed at opening. This remains a limitation.
2. **When is `expiration_time` set?** I could not establish whether it is set when a market opens or updated later. I excluded it because later updates could introduce information unavailable at prediction time.
3. **What does volume measure?** The unit was unconfirmed, so I excluded volume. Instead, I required at least 20 trades before day 7.

## 2. Problem statement

My objective was to predict whether a Polymarket Yes/No market resolves Yes, using information available during its first seven days.

I evaluated predictions using log loss and compared them with the market’s own price at day 7. The question was whether my model could improve on that price. This was a learning project; I make no claim about trading performance or profitability.

The frozen model did not demonstrate an improvement on the one-time test of 8,991 markets:

| Evaluation group | Model log loss minus market-price log loss | Reported interval |
|---|---:|---:|
| Contested markets | +0.0003 | −0.0013 to +0.0021 |
| All markets | −0.0001 | −0.0006 to +0.0004 |

Lower log loss is better. Both intervals include zero, so neither result establishes a difference from the market price.

## 3. Key decisions

### Decision 1: Split chronologically, keep events together and remove overlaps

**What I decided**

I used chronological training, validation and test splits, with boundaries on 23 December 2025 and 9 February 2026. I assigned each event to the split containing its earliest market and kept the event together. I also removed markets that closed after the next period began.

The final split sizes were:

| Split | Markets |
|---|---:|
| Training | 23,897 |
| Validation | 7,374 |
| Test | 8,991 |

**Why**

I wanted the evaluation to reflect predicting future markets from earlier data. Markets belonging to the same event can share information, so distributing them across splits could make the evaluation misleading.

**Trade-off**

The splits have different characteristics. The median time to close is 3.39 days in training and 0.86 days in validation and test. Contested markets account for 44.6% of training, compared with approximately 27% in the later splits.

The test also favours markets that close quickly, because markets still open at the end of the dataset are absent.

### Decision 2: Use market price as the benchmark and report contested markets separately

**What I decided**

I compared every model with the raw market price at day 7 using log loss. I reported results for all markets and for a contested subset, defined as markets with a last price of at least 0.05 and below 0.95.

**Why**

The market price was already well calibrated, making it a demanding benchmark. In validation, 4,899 of 7,374 markets—about 66%—were priced below 5%, and those forecasts were almost always correct. These markets lower the overall loss and can obscure performance on less certain outcomes.

**Trade-off**

The contested subset is smaller, so its results are noisier. I retained both views rather than relying on either one alone.

### Decision 3: Exclude `has_fee`

**What I decided**

I excluded `has_fee` from every model.

**Why**

Only six of the 23,897 training markets had the flag. That provided too little evidence for the model to learn a reliable relationship.

**Trade-off**

The 1,349 fee markets fall almost entirely in the test period. My results therefore do not establish whether the flag would have been useful with adequate training coverage.

### Decision 4: Freeze the model before opening the test

**What I decided**

I selected `lr_all_features` with C=1.0 using validation results and saved the choice in `frozen_choice.yaml`. I committed that file before opening the test and did not refit on combined training and validation data.

I ran the test once, on 25 November 2026 at 10:30 SGT. I kept `data/test_opened.lock` in place and made no subsequent tuning changes.

**Why**

The test needed to remain independent of model selection. Changing the model after seeing its test performance would undermine that check.

**Trade-off**

The conclusion rests on one test window, from 9 February to 30 March 2026.

The committed `frozen_choice.yaml` still contains `test_split: not opened`, reflecting its state before the run. The README explains this.

### Decision 5: Follow the selection rule despite a small margin

**What I decided**

My selection rule was the lowest validation log loss across all markets, compared at four decimal places. A tie would favour the simpler setting.

`lr_all_features` with C=1.0 scored 0.1533, compared with 0.1534 for `lr_price_only`. Under that rule, there was no tie.

**Why**

A predefined rule prevented me from choosing a model based on personal preference after seeing the results.

**Trade-off**

The difference was only 0.0001, with a reported interval of −0.0006 to +0.0007. That does not establish a meaningful difference between the models.

The selected model’s validation performance may also be slightly optimistic because those same results determined its selection.

### Decision 6: Use ablations to understand the model without changing the frozen choice

**What I decided**

I ran feature-group ablations using validation data. Across 28 comparisons, two were flagged as worse and none as better. I did not expand the search or change the frozen model in response.

**Why**

I wanted to investigate which inputs contributed useful information while keeping the final model choice fixed. Using these results to revise the model would introduce another round of selection on the same validation data.

**Trade-off**

The results did not support a clear claim that any feature group improved performance.

Adding activity features increased log loss relative to price-path features alone by 0.0003 across all markets and 0.0007 on contested markets. Both comparisons were flagged as worse.

Sibling count showed a possible benefit, but every interval included zero. It also increased over time, so the apparent benefit could reflect calendar effects.

## 4. Use of the AI assistant

I used Claude to write code and tests in small stages. I merged each stage after CI and Codex review. The following examples show where I intervened in the implementation and verification process.

### Case A: Reversed verdict labels

The initial README checker had the meanings of “better” and “worse” reversed.

I checked it against my actual result files. The check comparing saved verdicts with their reported intervals failed and exposed the error. The checker was corrected so that an interval entirely above zero means “worse”, and one entirely below zero means “better”.

### Case B: Missing table-header checks

Codex’s review of PR #39 identified that the checker did not validate table headers.

I requested the correction, which was implemented in PR #40. The subsequent Codex review reported no findings.

### Case C: Preserving the test lock

A test failed locally because the real test lock file already existed.

I kept the lock in place. The test was changed to compare the file’s bytes before and after execution, checking that it remained unchanged.

## 5. Next steps

With one additional week, I would:

- Evaluate the model on a later window to check whether the result holds beyond the current test’s concentration of quick-closing markets.
- Complete the `expiration_time` ablation and k=3 rerun, which I omitted because of time constraints.
- Re-examine sibling count while controlling for time, to distinguish a possible feature effect from a calendar effect.
