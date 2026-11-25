# Recommendation: Project 01, Bank Call Targeting

This is for a call-centre manager who is not technical. It makes no profit claim. The call costs below are invented.

## What the model does
It gives each client a score before the call, so the team can phone the clients most likely to say yes first.

## What I found
- It ranks clients better than a blind guess, but only modestly. On the validation month it was about 1.6 times better than guessing. On the final test it was about 1.4 times better.
- The three models I tried could not be told apart, so I chose the simplest.
- On the final test, all three rules for "how many clients to phone" did worse than simply phoning everyone. The reason is that in the test months far more clients said yes (44.5%) than in the month I used to choose the rule (9%). I did not find out why.

**My advice: do not use the "how many to phone" rule from this project. Use the model only to order the list, and decide how far down the list to go from your real call costs.**

## How to use the score, and how not to
- Use it to put the call list in order, most promising first.
- Do not use it to decide how many clients to phone. That number depends on what a call really costs, and on the share of clients who say yes at the time.
- Do not read it as proof that it works for another bank or another year. It was built on one Portuguese bank from 2008 to 2010.

## Where it still goes wrong
- When the share of clients saying yes changes a lot, as it did in the test months, a rule chosen earlier can phone too few clients and miss sales.
- It cannot see anything that is only known during or after the call, such as how long the call lasts, and I left that out on purpose.
- My costs are invented (0.03, 0.09 and 0.15 per call, with a subscription worth 1). Any figures you use are your own.

## How the advice changes if priorities change
- If calls are cheap, phone most of the list, perhaps everyone. In my cheap scenario the best share was 97%.
- If calls are expensive, phone only the top of the list. In my expensive scenario the best share was 5%.
- If your team can only make a fixed number of calls, cap the list at that number. The model only decides the order.

## One check before relying on it
Check last month's share of clients who said yes against the share in the data you built the rule on. If they are far apart, rebuild the rule before using it.
