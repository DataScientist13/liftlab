# ADR 0002 — MMM parameterisation and the identifiability it does not solve

**Status:** accepted
**Date:** 2026-08-09

## Context

The MMM has to estimate, per channel, a carryover parameter, a saturation point, a curve
steepness, and an effect size — from aggregate daily revenue. These are not cleanly separable.
Slow decay with early saturation mimics fast decay with late saturation, and a channel operated
far below its saturation point looks linear, which leaves its half-saturation informed almost
entirely by the prior.

Design choices therefore have to be made with identifiability in mind rather than expressiveness.

## Decision

**Scale everything.** Spend is divided by its per-channel median, revenue by its mean. Priors then
sit on a unit-free scale, and NUTS sees well-conditioned geometry. Scale factors are stored so
currency-scale results can be recovered.

**Normalise adstock.** Geometric carryover is multiplied by `1 - decay`. Without it, decay
rescales effect size and is confounded with the channel coefficient.

**Do not estimate the adstock family.** Whether a channel is geometric or Weibull is set by the
analyst. The data rarely distinguishes the families, and estimating it invites a bimodal posterior
that is worse than a stated assumption.

**Half-saturation on the spend scale.** After scaling, `1.0` means "half response at typical
spend", a quantity a media buyer can hold an opinion about — which is what makes the prior
elicitable rather than arbitrary.

**Hierarchical, non-centered channel coefficients.** Partial pooling regularises small channels;
the non-centered form avoids the funnel that otherwise produces divergences at small `tau`.

**Yom Kippur suppresses total revenue, not just baseline.** It is a supply-side closure. Media
that ran beforehand sells nothing on a day when the shops are shut.

**`init_to_median` and `target_accept_prob = 0.9`.** Random initialisation can land in the flat
tail of the Hill curve where the gradient vanishes and warmup never recovers.

## What was measured

On one seed (three years, five channels, 1000 warmup / 1000 draws, two chains) the sampler is
healthy — R-hat 1.00, bulk ESS 681–2849, 35 divergences in 2000 draws. Carryover recovers well:
estimated decay 0.13 / 0.36 / 0.66 / 0.35 against true 0.15 / 0.45 / 0.65 / 0.30.

Effect sizes do not recover as well. Four of five ROAS credible intervals covered the true value,
one did not, and the intervals are wide — one channel's 95% interval spanned 1.7 to 27.0. The
failures track half-saturation error: channels whose saturation point is underestimated have their
ROAS inflated, and better-identified channels absorb the difference downward, because total revenue
is conserved.

Two prior changes were made during development, both recorded here because they were made after
seeing diagnostics:

1. `tau_beta` widened from `HalfNormal(0.5)` to `HalfNormal(1.0)`. This changed nothing — the
   posterior sits at 0.38 either way — so over-pooling was ruled out as the explanation.
2. `half_saturation` widened from `LogNormal(0, 0.6)` to `LogNormal(log 1.5, 0.8)`. The original
   placed a half-saturation of four times median spend, unremarkable for upper-funnel media, at
   +2.4 standard deviations. This is prior misspecification visible without reference to the
   answer, and correcting it fixed the worst-affected channel.

No further tuning was done. Iterating priors until a single seed's numbers look good is precisely
the failure mode this project argues against.

## Consequences

- The model is honest about weak identification: it returns wide posteriors rather than confident
  wrong numbers. A wide interval is the correct output when spend never approaches saturation.
- **Point estimates of channel ROAS from MMM alone should not be trusted at face value.** This is
  the empirical case for the calibration bridge, which is the whole thesis of this repository:
  geo-experiment posteriors supply the external information the aggregate series does not contain.
- The single-seed numbers above are diagnostics, not results. They are not published in the README.
  The recovery benchmark — many seeds, reporting bias and interval coverage — is what will be.
