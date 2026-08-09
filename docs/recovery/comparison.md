# Effect of experiment calibration

Paired over the **8** replications healthy in both arms (seeds 0, 1, 2, 3, 4, 6, 7, 11). The arms otherwise exclude different seeds, and comparing their summary tables directly would confound the effect with a change in which replications were counted.

Each cell reads *uncalibrated → calibrated*.

| channel group           | n  | median abs ROAS error | mean bias       | coverage 95% | 95% width   |
|-------------------------|----|-----------------------|-----------------|--------------|-------------|
| tested (search, social) | 16 | 19.0% → 9.2%          | -12.2% → -4.8%  | 100% → 100%  | 2.10 → 1.42 |
| untested                | 24 | 49.2% → 45.6%         | +52.1% → +42.9% | 88% → 88%    | 3.17 → 2.97 |
| all channels            | 40 | 30.9% → 23.0%         | +26.4% → +23.8% | 92% → 92%    | 2.42 → 1.60 |

Calibration sharply improves the channels that were tested and barely moves the ones that were not. Pinning one channel's contribution does constrain what is left for the others to explain, but that spillover is weak: in practice you have to test the channel you want a trustworthy number for.
