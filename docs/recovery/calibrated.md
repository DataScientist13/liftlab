# Parameter-recovery benchmark

Profile **calibrated** — 12 replications of 1095 days across 5 channels, 500 warmup / 500 draws x 2 chains.

Reproduce with:

```bash
just recover --profile calibrated
```

## Results

| parameter       | n  | mean rel. bias | median abs rel. bias | coverage 95% | coverage 50% | median 95% width |
|-----------------|----|----------------|----------------------|--------------|--------------|------------------|
| beta            | 45 | +37.5%         | 21.4%                | 96%          | 56%          | 0.18             |
| decay           | 36 | +14.2%         | 15.9%                | 97%          | 47%          | 0.27             |
| half_saturation | 45 | +52.4%         | 18.7%                | 100%         | 69%          | 3.58             |
| roas            | 45 | +23.1%         | 22.6%                | 93%          | 49%          | 1.65             |
| slope           | 45 | +1.8%          | 19.1%                | 100%         | 53%          | 1.50             |

`coverage 95%` is the fraction of replications whose 95% credible interval contained
the true value. Nominal is 95%: materially below means the model is overconfident,
materially above means it is needlessly vague. Coverage matters more than bias — a
biased estimate with an honest interval is usable, an overconfident one is not.

## Effect of calibration, by channel group

Experiments were simulated for **search, social** — a 42-day window with a 20% relative standard error.

| channel group     | n  | mean rel. bias | median abs rel. bias | coverage 95% |
|-------------------|----|----------------|----------------------|--------------|
| untested channels | 27 | +41.6%         | 43.6%                | 89%          |
| tested channels   | 18 | -4.6%          | 9.2%                 | 100%         |

The experiment is simulated as unbiased: it measures the channel's true incremental
revenue with Gaussian noise. A real geo test with contaminated controls would be
biased, and the bridge would propagate that bias faithfully. This measures what
calibration buys when the experiment is sound, not what happens when it is not.

## Sampler health

Replications excluded: **3** of 12 — R-hat > 1.05, bulk ESS < 100, or divergence rate > 2%.

Worst divergence rate across replications: **9.0%**. Divergences are treated as a hard exclusion criterion rather than a warning: a divergent chain systematically fails to explore part of the posterior, so its intervals are the wrong width and its coverage is not evidence about the method.

Total sampling time 25 min.
