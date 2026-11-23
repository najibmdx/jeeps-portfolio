# Learning log

> Drafted with Claude from the record of my sessions (diagnostic results, handovers, chats), then reviewed by me. Where I have no record of something, it says so.

## What I can now explain

- **A score is read against a lazy guess.** A PR-AUC means little alone. A random guesser scores the share of positives in the same pile. In Project B the lazy guess is 0.713 on the crash side and 0.287 on the survivor side on validation, so the random forest's 0.823 is only 1.15x on the crash side.
- **Split by time.** Earlier coins must never see later ones, so I cut by migration time (60/20/20). A shuffled split would let later coins teach the model about earlier ones.
- **The "known at the time" rule.** A column is allowed only if it is known when the prediction is made. The audit flagged `group1_migrate_year` and `group1_migrate_month` as clock-like, so I never used them. 36 columns have timing I could not verify, so I tested them instead of trusting them.
- **Wobble.** Re-drawing a pile 200 times shows how much a score moves by luck. A difference counts only if its range excludes 0. It measures luck inside one pile, not change between periods.
- **Overfitting.** The random forest scores 0.830 on train and 0.478 on validation on the survivor side. The training score is not an honest grade.
- **Freeze before you test.** I wrote the model choice and the cut-offs to `frozen_choice.yaml` and committed it before opening the test pile, and I opened the test pile once.

## What I got wrong, and the right answer

- **Baseline (diagnostic, 21 Nov).** I called 88.6% a random guesser's PR-AUC. The right answer is the positive rate, 12.4%.
- **Why cut-offs lose on a new period (Project A).** I said "the model overfit". The right reason: the yes-rate jumped from 9.05% on validation to 44.50% on test, above every call cost, so cut-offs chosen on validation lost to "call everyone". Project B showed the same pattern: the crash share fell from 71.3% to 59.2%, and the "skipping a good coin costs 3x" cut-off lost to flagging nothing. This time I could not separate the shift from over-tuning.
- **Tuning (diagnostic).** I judged a tuning gain without comparing it to the spread of the folds. In Project B, tuning changed little, and gradient boosting's crash side got worse (0.826 to 0.814).
- **Git workflow.** On 21 Nov I answered "don't know". It is now routine: branch, small commits, pull request, merge, sync. I made mistakes: I merged before waiting for the go twice (Stage 7 PR #24 and Stage 8a PR #25; both turned out green), and in Project A I made commits on `main` before creating the branch and had to move `main` back.

## Working with AI

- It can be wrong, and it can sound sure. I caught an invented, unlabelled PR-AUC number, and an answer that mixed measured results with untested trading claims. Now I ask what is measured and what is an assumption, and every invented number is labelled.
- It wrote the code and the README. I ran everything, made the git steps and checked the numbers with `check_readme.py`.

## Still open

- Explaining PR-AUC against its baseline, the wobble, and overfitting in my own words without notes. My record shows these were explained, not re-tested.
- Why the cut-offs failed on test (shift or over-tuning). The test pile cannot answer it.
- Mock interview scores: none recorded in my notes.

## What I would do differently

- Check at the start which columns have verified timing, so fewer sit in the "unverified" group.
- Ask for real costs before building cost scenarios, instead of inventing three.
- Wait for the go and green CI before every merge.
