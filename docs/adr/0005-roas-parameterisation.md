# ADR 0005 — Put the prior on ROAS and derive the asymptote

**Status:** accepted
**Date:** 2026-08-09

## Context

The published benchmark showed channel ROAS point estimates biased **+26.4%** (posterior mean
against truth), with `beta` at +40% and `half_saturation` at +53%, despite honest interval
coverage. An upward offset of a quarter on the headline quantity is not a usable measurement
system, and fixing it was made a precondition for further work.

## Diagnosis

Three causes, in decreasing order of contribution:

1. **The prior sat on the least identified parameter.** The level was sampled as `beta`, the Hill
   asymptote, with a hierarchical prior. Below saturation the likelihood constrains only the
   product `beta * k**-s` (ADR 0004), so along that flat direction the *prior* on
   `half_saturation` decided the answer — and its median (1.5× median spend) sat above the true
   values for four of five channels. Overestimated `k` forces overestimated `beta`, and both
   propagate into ROAS.
2. **Skew inflation.** The benchmark scored the posterior mean of right-skewed posteriors. The
   weakly identified channels had enormous right tails (email's ROAS interval reached 27), and the
   mean of such a distribution sits far above its centre.
3. **Aggregation over unequal channels.** The aggregate bias was dominated by the low-spend
   channels where the data is weakest; search and social were actually *under*-estimated.

## Decision

Sample each channel's **realised ROAS** — total incremental revenue over total spend — and derive
the Hill asymptote from it:

```
roas_c ~ LogNormal(log 1.5, 0.7)
beta_c = roas_c * total_spend_c / (revenue_scale * Σ_t response_ct * open_t)
```

`k` and `slope` keep their existing priors and now carry only curve *shape*; the level is pinned by
a quantity the data identifies well. Additionally, the benchmark now reports bias for **both** the
posterior-mean and posterior-median estimators, so skew artifact and genuine bias are visible
separately.

Two points of provenance. First, informative ROAS priors were in the project specification from the
start; this ADR implements a listed feature, not a rescue. Second, this differs from rejected
ADR 0004 in what is being sampled: ADR 0004 moved the *saturation* parameterisation and changed its
prior simultaneously; this change touches only the level, leaves the shape priors untouched, and
was gated on coverage before running.

## Why this is not tuning the prior to the DGP

- The prior is centred at ROAS 1.5 with 95% mass on [0.38, 5.9], justified from published lift
  experiments: measured incremental ROAS above ~6 is rare, below ~0.3 the channel is burning money.
  The same prior would be used on real data.
- The DGP's true ROAS values (1.05–4.32) are not centred on 1.5; the email channel (4.3) sits at
  +1.5σ and is actively shrunk *away* from its truth by this prior. A prior tuned to the DGP would
  have been centred near 2.3.

## Pre-registered gate and results

The acceptance criteria were fixed before the benchmark ran: ROAS 95% coverage ≥ 90%, mean-based
ROAS bias magnitude at most half of +26.4%, divergence exclusions no worse than 4/12.

| metric | before | after | gate |
|---|---|---|---|
| ROAS bias (posterior mean) | +26.4% | **+5.2%** | pass (≤ ~13%) |
| ROAS bias (posterior median) | — | **−1.1%** | — |
| `beta` bias (mean / median) | +40.1% / — | +18.2% / +7.9% | — |
| ROAS 95% coverage | 92% | **90%** | pass, at the boundary |
| divergence exclusions | 4 / 12 | **2 / 12** | pass |
| sampling time (12 replications) | 23 min | **11 min** | — |

Coverage 90% vs 92% is within binomial sampling error (SE ≈ 3 points at n = 50) of both the
previous figure and nominal; it is neither a regression nor an improvement that this replication
count can resolve.

## Costs, stated rather than buried

- **Extreme channels are shrunk.** Email (true ROAS 4.3) is estimated around 2–3 uncalibrated.
  The remaining 25.5% median absolute ROAS error is concentrated in exactly the low-spend channels
  the prior regularises hardest. The remedy is an experiment on that channel — measured in the
  calibration arm at 8.5% median error and 100% coverage for tested channels — not a wider prior,
  which would reintroduce the tails that caused the original offset.
- **Divergences are reduced, not eliminated.** The worst seed still diverges at 9.6%; the residual
  cause is the ADR 0004 flat direction in the saturation parameters of far-below-saturation
  channels, which this change does not touch.
- **The reported `beta` and `half_saturation` biases remain positive** (+8% and +17% at the
  median). They are shape parameters the data genuinely cannot pin; their coverage is 98%.

## Consequences

- The README results table is regenerated and its narrative rewritten: the offset story becomes a
  spread story, and the case for the calibration bridge shifts from "fixes the bias" to "fixes the
  channels the prior cannot" — which the paired comparison now shows directly.
- ADR 0002's account of the measured bias is superseded by this record; ADR 0004's diagnosis
  stands and its rejection stands.
