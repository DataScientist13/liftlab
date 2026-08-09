# ADR 0004 — Reparameterising the saturation curve: tried, measured, rejected

**Status:** rejected
**Date:** 2026-08-09

## Context

The benchmark excludes replications exceeding a 2% divergent-transition rate. Four of twelve were
being excluded, with R-hat and bulk ESS clean in every one — a geometry problem, not an adaptation
problem, and the reason this model is not something to run unattended.

## Diagnosis

On the worst replication, no parameter had a shifted marginal in divergent draws (largest shift
0.23 sd), which rules out a funnel in any single parameter. Posterior correlations named the cause:

```
beta[tv]     <-> half_saturation[tv]   r = +0.910
beta[social] <-> slope[social]         r = -0.823
beta[search] <-> slope[search]         r = -0.771
```

Below saturation the Hill curve is approximately

```
beta * hill(x; k, s)  ~=  beta * (x / k)**s  =  (beta * k**-s) * x**s
```

so daily aggregate data identifies the *product* `beta * k**-s`, not `beta` and `k` separately.
The sampler was traversing a near-perfect ridge. **This diagnosis stands** and is the reason the
divergences happen.

## What was tried

Sample the identified quantity instead: per channel, `kappa` (contribution at median spend) and
`saturation` (response fraction at median spend, in `(0, 1)`), deriving `beta = kappa / saturation`
and `k = ((1 - saturation) / saturation) ** (1 / slope)`. Alongside it, pool `slope` across
channels and drop the hierarchy on channel level, which measurement had already shown did nothing
(the posterior for `tau` sat at 0.38 regardless of its prior).

## Result: rejected

It worked on the axis it targeted and regressed the axis that matters.

| metric | original | reparameterised |
|---|---|---|
| replications excluded (divergences) | 4 / 12 | **2 / 12** |
| sampling time | 23 min | **12 min** |
| **ROAS interval coverage** | **92%** | 86% |
| slope interval coverage | 100% | 86% |

Nominal coverage is 95%. Trading six points of interval coverage — becoming measurably more
overconfident — for fewer divergences and faster sampling is the wrong trade for this project,
whose entire claim is that its uncertainty is honest. A model that samples cleanly and reports
intervals that are too narrow is worse than one that samples badly and reports intervals you can
trust, because the first failure is invisible at the point of use.

A secondary problem: `half_saturation` becomes a derived quantity with a heavy right tail as
`saturation → 0`, and its reported mean relative bias blew up to absurd magnitudes. That is an
artifact of the parameterisation rather than a change in what the model believes about revenue,
but it made the benchmark's own reporting untrustworthy for that column.

## Things that did not help, recorded because their failure is informative

- **Longer warmup.** 500 → 1000 left the worst seed unchanged (8.5% → 9.9%).
- **Higher acceptance target.** 0.95 → 0.995 moved it 6.9% → 6.4%. Divergences insensitive to step
  size are not an adaptation problem.
- **Non-centring the channel hierarchy.** Made it *worse* (8.8% → 11.3%). Non-centring helps when
  data is weak relative to the prior; the whole point of the reparameterisation was to make that
  quantity the best-determined one, which is the regime where centring wins.

## Consequences

- The original parameterisation stands. Divergence exclusions remain a documented limitation.
- The residual cause is understood and is structural: for a channel whose spend never approaches
  saturation, the saturation parameter is genuinely unidentified. In the benchmark DGP, TV's true
  saturation at median spend is 0.055, inside exactly that flat region. **A prior strong enough to
  remove the flat direction would bias the channel it is meant to stabilise.**
- **Identified next step, not yet done:** reparameterise while *matching the implied prior* on
  `half_saturation` to the original `LogNormal(log 1.5, 0.8)`, so the change is purely geometric and
  cannot move coverage. The attempt recorded here changed the prior and the geometry at the same
  time, which is why its coverage regression cannot be attributed to one or the other.
- The practical remedy available today is to fix saturation by assumption for channels whose spend
  never approaches it — an explicit modelling decision, made by an analyst, rather than something
  the sampler is asked to guess.
