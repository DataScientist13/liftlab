# Parameter-recovery benchmark

Profile **full** — 12 replications of 1095 days across 5 channels, 500 warmup / 500 draws x 2 chains.

Reproduce with:

```bash
just recover --profile full
```

## Results

| parameter       | n  | mean rel. bias | median abs rel. bias | coverage 95% | coverage 50% | median 95% width |
|-----------------|----|----------------|----------------------|--------------|--------------|------------------|
| beta            | 40 | +40.1%         | 22.7%                | 98%          | 42%          | 0.18             |
| decay           | 32 | +14.4%         | 14.3%                | 97%          | 47%          | 0.27             |
| half_saturation | 40 | +53.1%         | 24.5%                | 98%          | 65%          | 3.28             |
| roas            | 40 | +26.4%         | 30.9%                | 92%          | 48%          | 2.42             |
| slope           | 40 | +6.5%          | 19.9%                | 100%         | 50%          | 1.53             |

`coverage 95%` is the fraction of replications whose 95% credible interval contained
the true value. Nominal is 95%: materially below means the model is overconfident,
materially above means it is needlessly vague. Coverage matters more than bias — a
biased estimate with an honest interval is usable, an overconfident one is not.

## Sampler health

Replications excluded: **4** of 12 — R-hat > 1.05, bulk ESS < 100, or divergence rate > 2%.

Worst divergence rate across replications: **8.8%**. Divergences are treated as a hard exclusion criterion rather than a warning: a divergent chain systematically fails to explore part of the posterior, so its intervals are the wrong width and its coverage is not evidence about the method.

Total sampling time 23 min.
