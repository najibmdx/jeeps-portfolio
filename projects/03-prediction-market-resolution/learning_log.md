# Learning log: Project C, Prediction Market Resolution

Drafted with Claude, reviewed by me. These are things this project taught me, with the evidence from the project.

## 1. A null result is a real result
I built four models, tuned them and tested the best one once. It could not be told apart from the market price (contested slice +0.0003, range -0.0013 to +0.0021). I reported that as it came out. I learned that "no measurable difference" is a finding I can defend, and that a good-looking number I could not defend would be worse.

## 2. Pick the model before the test, and never look twice
I froze the choice in `frozen_choice.yaml` before the test was opened, and opened the test once. I learned why: if I choose after seeing the test, the test no longer checks anything. The same idea held for the ablations. I did not swap a better-looking column set into the frozen model, because that would be choosing twice on the same validation markets.

## 3. Split by time, because the model should only learn from the past
A random split would let the model learn from markets from the same weeks as the test ones, and the test score would look better than it really is. I split in time order and kept each event whole, so sibling markets could not leak across splits. This is the reason I split the way I did. I should be able to explain it in my own words, and I am still practising that.

## 4. The easy cases hide the hard ones
On validation, 66% of markets are priced under 5% and almost always right. They make every score look good. I learned to report a "contested" slice (last price from 0.05 up to but not including 0.95), because that is where a model could add anything.

## 5. Check that a number really is what I think it is
- My first README checker had "worse" and "better" swapped. Running it against my real result files caught the mistake.
- Codex found that table headers were not checked. I fixed that in a second pull request.
- A test failed because the real lock file exists. I fixed the test and did not delete the lock.
I learned to test against real output, not only made-up data, and to read review comments as information.

## 6. Small things that mattered
- Dropping `has_fee`: only 6 of 23,897 train markets had it, so it could not be learned.
- Wobble: a small gap means little unless I know how much it moves. I used 200 resamples of whole events to get a 90% range.
- Generated tables: the README results are written by a script from the result files, so a typed number cannot drift from the real one.

## What I would do next
Run the same test on a later window. My one test window leaned toward quick-closing markets, and a later window would show whether the result holds.
