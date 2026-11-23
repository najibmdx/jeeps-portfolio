# Decision log

> Drafted with Claude from my project outputs and chats, then reviewed by me. Numbers come from `outputs/*.json` and the README.

## 1. Clarifying questions

I could not ask the data owner anything, so these are the questions I would have asked and what each answer would change.

**How is a crash defined?**
I used the authors' rule: a crash is `min_ratio` below 0.3 in the 20 minutes after migration. 63.2% of the 21,635 kept coins crash. If the owner said a trader means something else, I would first count how the crash share changes under the other rule, because every score is compared with a lazy guess for that share.

**What does the scope filter hide?**
Only coins with at least 100 holders and at least 60 s of valid swaps are kept: 21,635 of the 41,470 coins that have a result. If the owner said the dropped coins behave very differently, I would ask how many real launches pass the filter. If few do, the model covers little, and I would say so first.

**When does each column become known?**
I could not check the timing of the cluster columns (group 4). Whether the owner said before or after migration, I would keep them out of the model until the timing is proven, and report their test result with the flag "unverified timing".

## 2. Problem statement

For a coin that has just migrated, I flag it as likely to crash when I expect its price to fall below 30% of its migration price at any point in the next 20 minutes (`min_ratio` below 0.3, the MELT authors' rule), using only information known at migration.

This measures a price crash, not proof of a scam. It is a risk flag, not a trade signal. It covers only coins with at least 100 holders and 60 s of valid swaps.

## 3. Key decisions

### Decision 1: split by migration time, 60/20/20
- **What I did:** ordered coins by migration time and cut the oldest 60% as train (12,981 coins, crash share 61.8%), the next 20% as validation (4,327, 71.3%) and the newest 20% as test (4,327, 59.2%). Seed 42.
- **Why:** the model would be used on later launches, so earlier coins must never see later ones. The test pile stays cleanly in the future.
- **What it costs:** the crash mix shifts between piles (61.8%, 71.3%, 59.2%), so the piles are not alike. On test, the cut-offs I chose on validation did not transfer evenly, and I cannot separate the shift from over-tuning. The test pile may not be reopened to find out.

### Decision 2: 81 columns from groups 2 and 3 only
- **What I did:** used 81 columns whose timing is confirmed from the authors' code. I held back group 1 (calendar and SOL price) and group 4 (cluster columns) and tested them as ablations.
- **Why:** held-back groups are tested, not guessed, so I can show what each one adds.
- **What it costs or how I checked:** plus_group4 raised the crash side from 0.814 to 0.821 (+0.007, range +0.003 to +0.011); both groups gave 0.825 (+0.010). Survivor-side changes were never separable from zero. The audit lists 36 columns with unverified timing, so I treated the gain as possible leakage and added nothing. The cost is that I may have left a real crash-side gain unused. I cannot tell without the authors confirming the timing.

### Decision 3a: `group1_migrate_year` never used
- **What the audit showed:** flagged `clock_like`. It is one of only two flags among 122 columns. The other flag types (`extreme_bin`, `mostly_zero`, `constant`, `strong_alone`) flagged nothing.
- **What I did:** left it out of every model and every column test.
- **Why:** it says when the coin migrated, not what the coin did.
- **What it costs or how I check:** I have no number for what it would have added. Migration time alone ranks crashes with AUC 0.5433 on the train pile, a little above chance, so time carries some signal by itself.

### Decision 3b: `group1_migrate_month` never used
- **What the audit showed:** flagged `clock_like`, the same as 3a.
- **What I did:** left it out of every model and every column test.
- **Why:** same reason as 3a. A model could learn the date, not the coin.
- **What it costs or how I check:** same as 3a. I never ran it, so I have no number for its effect.

### Decision 4: score the crash side and the survivor side
- **What I did:** report PR-AUC for finding crashers and for finding survivors, each against the lazy guess of its own pile (validation: 0.713 on the crash side, 0.287 on the survivor side).
- **Why:** most coins crash (63.2%), so the crash side looks good by default.
- **What it costs:** two sets of numbers to read. The raw survivor score for the random forest (0.478) looks low next to the crash score (0.823) because survivors are only 28.7% of validation. Against its own lazy guess it is 1.66x, while the crash side is 1.15x.

### Decision 5: model choice rule, frozen before the test
- **What I did:** the tuned model with the highest average of crash-side and survivor-side PR-AUC on validation wins, and a tie goes to the earlier candidate. Averages: logistic regression 0.621, random forest 0.650, gradient boosting 0.647. The random forest won (`max_depth: None`, `min_samples_leaf: 20`). I wrote the choice and the cut-offs to `frozen_choice.yaml` (commit `e81bc82`) before the test pile was opened.
- **Why:** both sides count equally, and a rule written before I looked at the test stops me picking a model after seeing its test score.
- **What it costs or how I check:** the random forest leads gradient boosting by only 0.0033, and their wobble ranges overlap almost completely, so the pick is close to a coin flip. Validation was reused for tuning and choice, so its scores are somewhat optimistic. The one-time test is the check: the random forest and logistic regression cannot be told apart on the crash side, and on the survivor side the random forest is +0.024 better (+0.013 to +0.038).

### Decision 6: set aside 233 coins with no result
- **What I did:** dropped the 233 feature rows that have no row in the label file, and said so in the README.
- **Why:** with no label, there is nothing to learn from or to grade.
- **What it costs:** I do not know whether those coins differ from the rest. I only know the count.

### Decision 7: keep 8 boundary coins
- **What I did:** kept 8 coins whose 20-minute window runs past the start of the next pile (6 in train, 2 in validation).
- **Why:** 8 of 21,635 coins is a negligible effect, and their outcomes are in the label file. I disclose them instead of dropping them.
- **What it costs:** a tiny overlap between those coins' outcome windows and the next pile.

## 4. Use of the AI assistant

I used Claude (Anthropic) as a trainer and code assistant. It wrote the code, the tests and the README; I ran everything on my machine and made my own git steps. These are three cases where I directed, challenged or changed what it did.

**1. I challenged an answer that mixed measured results with assumptions.**
Claude's answer about whether the model could predict up or down mixed what we had measured (59% to 71% of coins crash; the cost-based cut-offs failed on test) with common trading patterns it had not tested on memecoins ("edges decay fast", "costs wipe out small edges"). I pushed back. Claude split its answer into "grounded in your data" and "my assumptions, not tested", and said the only supported answer is that Project B tests one 20-minute crash question on one dataset.
*Why it mattered:* it is why the README and this log make no claim about trading, returns or profit.

**2. I kept 8 boundary coins for my own reason, and Claude corrected the reason.**
I wanted to keep the 8 coins whose outcome window crosses a pile boundary, because I did not know their final result. Claude pointed out their outcomes are in the label file, so that was not the reason. I kept the decision and changed the reason: the effect is negligible (8 of 21,635), and I disclose it.

**3. I set the viewpoint: everything is read as a memecoin trader.**
I told Claude that all of Project B must be taken from a memecoin trader's view. That is why the two mistakes are named *skipped good* (flagging a coin that survived) and *held crasher* (not flagging a coin that crashed), why the costs are labelled as invented, and why the label is described as a price crash from a price a trader cannot buy at.

## 5. Next steps (one more week)

1. **Try a label a trader could actually use.** My label measures a drop from the migration price, which a trader cannot buy at. I would define a label from a price a trader could get and run it through the same pipeline. I make no claim that it would show anything.
2. **Get real costs for the two mistakes.** All three cost scenarios are invented and unit-free.
3. **Find out why the "skipping a good coin costs 3x" cut-off lost on test.** It cost 0.6284 per coin against 0.5923 for flagging nothing. I would need a fresh, untouched period, because the test pile is used up.
4. **Ask the authors when the 36 unverified columns become known.** Only then can the +0.007 and +0.010 crash-side gains be counted as real or as leakage.
