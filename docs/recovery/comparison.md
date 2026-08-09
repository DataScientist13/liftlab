# Effect of experiment calibration

Paired over the **9** replications healthy in both arms (seeds 0, 1, 2, 3, 4, 6, 7, 9, 11). The arms otherwise exclude different seeds, and comparing their summary tables directly would confound the effect with a change in which replications were counted.

Each cell reads *uncalibrated → calibrated*.

| channel group           | n  | median abs ROAS error | mean bias       | coverage 95% | 95% width   |
|-------------------------|----|-----------------------|-----------------|--------------|-------------|
| tested (search, social) | 18 | 12.8% → 8.5%          | -1.2% → -0.8%   | 100% → 100%  | 2.33 → 1.37 |
| untested                | 27 | 41.5% → 39.8%         | +10.2% → +10.2% | 81% → 85%    | 3.00 → 2.73 |
| all channels            | 45 | 26.2% → 21.0%         | +5.6% → +5.8%   | 89% → 91%    | 2.52 → 1.59 |

Calibration sharply improves the channels that were tested and barely moves the ones that were not. Pinning one channel's contribution does constrain what is left for the others to explain, but that spillover is weak: in practice you have to test the channel you want a trustworthy number for.
