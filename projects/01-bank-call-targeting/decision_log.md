# Decision log: Project 01, Bank Call Targeting

Drafted with Claude, reviewed by me. Numbers come from my own result files in `outputs/` and from the README.

## 1. Questions I could not ask the bank

I had no one to ask, so these are the questions I would have put to the bank, what I assumed instead, and how a different answer would change what I built.

**Cost of a call.**
- Assumed: three invented cost scenarios (cheap 0.03, moderate 0.09, expensive 0.15, with a subscription worth 1). They are labelled as assumptions in `config.yaml` and the README. I set 0.09 close to the validation yes-rate (9.05%), which is the break-even point for phoning a client picked at random.
- If the real cost were different: it changes how many clients to phone. Cheap calls mean phoning many clients, maybe everyone. Expensive calls mean phoning only the top few. In my results the share phoned was 97%, 40% and 5% for the three scenarios.

**Call capacity (how many clients the team can phone).**
- Assumed: nothing about capacity. The code has no capacity limit. The share to phone is chosen from the cost alone. I reported precision, recall and lift in the top 10% of clients as a simple reading of a limited team, but I did not use it to choose the cut-off.
- If the bank gave me a real capacity: I would cap the share at that capacity. This matters most when calls are cheap, because the share chosen for the cheap scenario (97%) is far more than any real team could phone. The model itself stays the same.

**When each column is known.**
- Assumed: `duration` is only known after the call, so I excluded it. I assumed the client columns (`age`, `job`, `marital`, `education`, `default`, `housing`, `loan`), `contact`, and the history columns (`pdays`, `previous`, `poutcome`) are known before the call. For the doubtful ones (`month`, `day_of_week`, the economy columns and `campaign`) I kept them out of the first model and tested them as ablations.
- If the bank told me a column is known only during or after the call: I must drop it. If it is known before the call, I can keep it.

**How long each call lasts.**
- This links to `duration`: call length is only known at the end of the call, which is why it leaks.
- It also changes the cost: longer calls cost more per client. My cost scenarios use one flat cost per call, so I did not model this.

## 2. The problem I decided to solve

The call team can phone only some clients. The model gives each client a score before the call, so the team can phone the most promising clients first, and the cost of a call decides how far down the ranked list to go. The prediction is made before the call, so anything only known during or after the call is not allowed.

**Key assumptions.**
- The rows are in time order. The file has no date column, so I rebuilt the year from the row order, relying on the dataset authors' statement that the file starts in May 2008.
- Calls are independent: phoning one client does not change another client's answer.
- The call costs are invented, and I did not model call capacity.

**Other framings I considered and did not choose.**
- Keeping `duration` as a normal feature. I did not choose it because it is not known before the call (it is leakage).
- Predicting profit directly. I did not choose it because the call costs are invented. My models predict yes or no, and the costs only come in when I choose how many clients to phone. I make no profit claim.

**What I left out.**
- Why clients say yes. I rank clients, I do not explain causes.
- A live system: no deployment, monitoring or retraining plan.
- Any data beyond the public UCI file.

## 3. Three key decisions

**Decision 1: time split, not random split.**
- Options: a random split of rows, or a split by time.
- Choice: split by time. Train on May 2008 to Apr 2009, validate on May 2009, test on Jun 2009 to Nov 2010.
- Why: it copies real use. In real use the model predicts future clients from the past. A random split would let later months leak into training.

**Decision 2: exclude `duration`.**
- Options: keep it, or exclude it. I also ran one clearly labelled comparison with it, to see what leakage looks like.
- Choice: exclude it from every real model.
- Why: it is not known before the call, so it breaks my rule that a column is allowed only if it is known when the prediction is made. The labelled run lifted validation PR-AUC by +0.348 to +0.373 across the three models, which is leakage, not skill.

**Decision 3: choose how many to phone per cost scenario, on validation.**
- Options: a fixed score threshold (for example 0.5), a fixed top 10%, or a share of clients chosen on validation for each cost scenario.
- Choice: a share of clients per cost scenario, the one with the highest net value on validation (subscriptions found minus cost per call times calls made), searched in steps of 1%. The test set was used once.
- Why: the best number of calls depends on the cost. I used a share and not a score threshold because the yes-rate drifts between periods, so a score threshold picked on validation would behave very differently on test.
- What happened: on test, all three chosen cut-offs lost to "phone everyone". The yes-rate was 9.05% in validation but 44.50% in test. At my assumed call costs, phoning a random client pays whenever the yes-rate is above the cost per call, so a cut-off that skips clients loses. I did not investigate why the yes-rate was so different.

## 4. Every column the leakage audit flagged

The audit (`outputs/leakage_audit.csv`) flags a segment when it has at least 50 training rows and a yes-rate 3x above or below the training yes-rate (6.27%), or exactly 0% or 100%.

**`previous` (value 2) and `poutcome` (success).**
- What the audit showed: yes-rates of 20.69% (58 rows) and 38.24% (102 rows), 3.3x and 6.1x the training yes-rate.
- What I did: kept them, together with `pdays`. They come from earlier campaigns, so they are known before this call. Removing the three history columns made validation worse (-0.029 to -0.040 across the three models).

**`month` (March and October), `cons_price_idx`, `cons_conf_idx`, `emp_var_rate`, `nr_employed`.**
- What the audit showed: several of these show the same row counts and yes-rates. March, `cons_price_idx` 92.843 and `cons_conf_idx` -50.0 all show 282 rows at 44.68%. October, `cons_price_idx` 93.798 and `cons_conf_idx` -40.4 all show 67 rows at 62.69%. `emp_var_rate` -1.8 and `nr_employed` 5099.1 both show 2,739 rows at 20.74%. This looks like a clock: the values mark a stretch of time, not a kind of client.
- What I did: held them back from the first model and tested them as ablations. Adding the calendar columns made validation worse for logistic regression and gradient boosting. Adding the economy columns made no clear difference.

**`duration`.**
- What the audit showed: flagged in 4 of 5 bins (from 0.16% to 24.84% yes-rate).
- What I did: excluded it, and ran one labelled demonstration of the leakage that is never used in a real model.

**Not flagged, but doubtful: `campaign`.**
- It counts contacts in this campaign and may include the call being predicted, so its timing is unclear. I excluded it from the first model and tested it as an ablation. It gave a small gain (+0.003 to +0.006), but I did not add it to the final model.

**What I could not prove.**
- That the repeated row counts mean these columns act as a clock. That is an inference from the pattern.
- The 3x and 50-row flag rule are my assumptions, not rules from the data.

## 5. Use of the AI assistant

I used Claude as a trainer and code assistant. These are three cases where I changed or rejected what it did.

**1. I rejected a change to my plan.**
- It steered me toward finishing only Project A by 27 Nov.
- I kept my own plan: Project A and Project B, then submit.

**2. I challenged numbers it did not label.**
- It used an invented PR-AUC of 0.25 without saying it was invented.
- I challenged it. Since then every made-up number is labelled as invented, for example the cost scenarios.

**3. I changed how it explains things.**
- It sent long multi-step lists and dense explanations, and I said I was lost.
- I set a rule: one action per message, short and plain, with commands in separate blocks.

## 6. Limits and next steps

**What my result shows.**
- The models rank clients better than a blind guess, but only modestly: 1.58x to 1.66x on validation and 1.36x on the one-time test run.
- `duration` is leakage.

**What it does not show.**
- Why the yes-rate jumped from 9.05% (validation) to 44.50% (test).
- That it works for another bank or period. It is one bank and one period (2008 to 2010).
- Real profit. The call costs are invented.
- Which model is best. The three models cannot be told apart on one validation month.

The model is not ready for the bank to use as it is.

**If I had one more week.**
1. Investigate the yes-rate jump. The cut-offs chosen on validation lost to phoning everyone on test in all three scenarios.
2. Get real costs and call capacity from the bank, because the cheap scenario would have me phone 97% of clients, more than a real team could.
3. Check month by month whether the ranking holds. Validation is a single month, and my wobble ranges do not measure change from month to month.
4. Re-test the three models on more months, because their ranges overlap on one validation month.
