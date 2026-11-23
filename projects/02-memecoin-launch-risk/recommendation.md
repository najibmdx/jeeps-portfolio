# Recommendation: how to use the launch-risk flag

> Drafted with Claude from my project outputs, then reviewed by me. This is a learning project. It makes no claim about trading use, returns or profit.

## What it does

When a Solana memecoin has just migrated from Pump.fun to a DEX, the flag scores how likely it is to crash in the next 20 minutes, where "crash" means its price drops below 30% of its migration price. It is a risk warning, not a buy or sell signal.

## How to use the score

**Do:**
- Use it to rank coins from riskier to safer. The ranking is the useful part. On coins it had never seen, it did better than a lazy guess, but only modestly: about 1.2x better at spotting crashers and about 1.5x better at spotting coins that survive.
- Treat a high score as "check this coin harder", not as "this will crash".

**Do not:**
- Do not read the score as a probability. It is a ranking.
- Do not use it to decide how much to risk, or on its own to decide whether to trade.
- Do not use it outside the coins it was built on: coins with at least 100 holders and 60 seconds of trading, from December 2024 to February 2025.

## The two mistakes it still makes

Most coins crash (59% to 71% in different periods), so a flag that says "crash" for nearly everything looks good. The two mistakes are:

- **Skipped good coin:** the flag says "likely to crash" but the coin survives. You pass on a coin that was fine.
- **Held crasher:** the flag stays quiet but the coin crashes. You are still in when it falls.

On the one-time test of 4,327 coins, using the equal-cost cut-off, the flag skipped 1,221 good coins and missed 278 crashers. It caught 89.1% of the crashers, but it also flagged 81.0% of all coins, so it kept only 30.8% of the survivors.

## What each mistake costs (assumptions)

The costs below are **invented**. I do not know what these mistakes cost in real trading. They only show how the advice moves when one mistake hurts more than the other.

| If this is true (invented costs) | Advice |
|---|---|
| Both mistakes cost the same | Use the flag to cut out the riskiest coins. On test it beat both "flag everything" and "flag nothing". It still flags most coins. |
| Holding a crasher costs 3 times more | The best setting flags almost everything (98.9% on test). The flag then adds very little over "flag everything". |
| Skipping a good coin costs 3 times more | Do not rely on the cut-off. It looked fine on older coins, but on test it did worse than flagging nothing. |

## How the advice changes if priorities change

If missing a crash hurts much more than passing on a good coin, flag almost everything and accept that the flag helps little. If passing on good coins hurts more, the flag is not reliable enough to lean on.

## One check before relying on it

Look at the share of recent coins that crashed. The flag was built when that share was 59% to 71%. When it moved between periods, the cut-offs stopped working evenly. If the recent share is far outside that range, do not trust the cut-offs.

## The main limit

I tested it on one dataset and one three-month window. Nothing here shows how it behaves on new coins or in another period, and I cannot tell whether the cut-offs failed on test because the crash share changed or because they were tuned too closely to older coins.
