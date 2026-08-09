# Parameter-recovery benchmark

Profile **full** — 12 replications of 1095 days across 5 channels, 500 warmup / 500 draws x 2 chains.

Reproduce with:

```bash
just recover --profile full
```

## Results

| parameter       | n  | bias (mean est.) | bias (median est.) | median abs rel. bias | coverage 95% | coverage 50% | median 95% width |
|-----------------|----|------------------|--------------------|----------------------|--------------|--------------|------------------|
| beta            | 50 | +18.2%           | +7.9%              | 20.7%                | 98%          | 58%          | 0.23             |
| decay           | 40 | +7.1%            | +6.0%              | 13.0%                | 98%          | 62%          | 0.28             |
| half_saturation | 50 | +38.2%           | +17.2%             | 33.8%                | 98%          | 70%          | 4.68             |
| roas            | 50 | +5.2%            | -1.1%              | 25.5%                | 90%          | 42%          | 2.57             |
| slope           | 50 | +4.8%            | +0.2%              | 19.8%                | 100%         | 46%          | 1.65             |

`coverage 95%` is the fraction of replications whose 95% credible interval contained
the true value. Nominal is 95%: materially below means the model is overconfident,
materially above means it is needlessly vague. Coverage matters more than bias — a
biased estimate with an honest interval is usable, an overconfident one is not.

## Sampler health

Replications excluded: **2** of 12 — R-hat > 1.05, bulk ESS < 100, or divergence rate > 2%.

Worst divergence rate across replications: **9.6%**. Divergences are treated as a hard exclusion criterion rather than a warning: a divergent chain systematically fails to explore part of the posterior, so its intervals are the wrong width and its coverage is not evidence about the method.

Total sampling time 11 min.
