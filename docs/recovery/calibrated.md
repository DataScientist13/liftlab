# Parameter-recovery benchmark

Profile **calibrated** — 12 replications of 1095 days across 5 channels, 500 warmup / 500 draws x 2 chains.

Reproduce with:

```bash
just recover --profile calibrated
```

## Results

| parameter       | n  | bias (mean est.) | bias (median est.) | median abs rel. bias | coverage 95% | coverage 50% | median 95% width |
|-----------------|----|------------------|--------------------|----------------------|--------------|--------------|------------------|
| beta            | 45 | +17.0%           | +6.9%              | 19.9%                | 98%          | 56%          | 0.21             |
| decay           | 36 | +10.4%           | +9.4%              | 15.2%                | 97%          | 61%          | 0.27             |
| half_saturation | 45 | +36.2%           | +14.3%             | 29.9%                | 100%         | 69%          | 4.59             |
| roas            | 45 | +5.8%            | +1.2%              | 21.0%                | 91%          | 44%          | 1.59             |
| slope           | 45 | +1.6%            | -2.4%              | 18.4%                | 100%         | 49%          | 1.58             |

`coverage 95%` is the fraction of replications whose 95% credible interval contained
the true value. Nominal is 95%: materially below means the model is overconfident,
materially above means it is needlessly vague. Coverage matters more than bias — a
biased estimate with an honest interval is usable, an overconfident one is not.

## Effect of calibration, by channel group

Experiments were simulated for **search, social** — a 42-day window with a 20% relative standard error.

| channel group     | n  | mean rel. bias | median abs rel. bias | coverage 95% |
|-------------------|----|----------------|----------------------|--------------|
| untested channels | 27 | +10.2%         | 39.8%                | 85%          |
| tested channels   | 18 | -0.8%          | 8.5%                 | 100%         |

The experiment is simulated as unbiased: it measures the channel's true incremental
revenue with Gaussian noise. A real geo test with contaminated controls would be
biased, and the bridge would propagate that bias faithfully. This measures what
calibration buys when the experiment is sound, not what happens when it is not.

## Sampler health

Replications excluded: **3** of 12 — R-hat > 1.05, bulk ESS < 100, or divergence rate > 2%.

Worst divergence rate across replications: **6.9%**. Divergences are treated as a hard exclusion criterion rather than a warning: a divergent chain systematically fails to explore part of the posterior, so its intervals are the wrong width and its coverage is not evidence about the method.

Total sampling time 9 min.
