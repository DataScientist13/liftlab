"""Parameter-recovery benchmark: simulate a known DGP, fit, report bias and coverage.

Fitting a model tells you what the data implies under that model. It does not tell you whether
the procedure works. This module answers the second question, which is the one almost no public
MMM repository asks: **when the truth is known, does the method find it, and are the credible
intervals honest?**

Two quantities are reported per parameter:

- **Relative bias** — signed error of the posterior mean against truth, averaged over seeds.
  Answers "does the point estimate systematically miss, and in which direction?"
- **Interval coverage** — the fraction of seeds whose credible interval contained the truth.
  A well-calibrated 95% interval should contain truth about 95% of the time. Materially below
  nominal means the model is overconfident; materially above means it is needlessly vague.

Coverage is the more important of the two. A biased estimator with honest intervals is usable;
a well-centred estimator with overconfident intervals will get someone's budget moved on a
number that was never that certain.

Profiles
--------
``fast`` is a smoke profile: it verifies the harness runs end to end and that sampler
diagnostics are sane. Its seed count is far too small for coverage to mean anything, and the
gate reflects that. ``full`` is the profile whose numbers are worth publishing.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from liftlab.dgp import (
    DEFAULT_CHANNELS,
    DGPConfig,
    GeometricAdstock,
    SyntheticPanel,
    generate_panel,
)
from liftlab.model import ChannelModel, ExperimentResult, MMMData, fit, prepare_data

if TYPE_CHECKING:  # pragma: no cover - typing only
    import arviz as az

__all__ = [
    "FAST",
    "FULL",
    "PROFILES",
    "BenchmarkProfile",
    "check_gate",
    "format_comparison",
    "format_report",
    "main",
    "paired_comparison",
    "run_benchmark",
    "summarise",
]


@dataclass(frozen=True)
class BenchmarkProfile:
    """One benchmark configuration.

    Attributes
    ----------
    name
        Profile identifier, recorded in the report.
    seeds
        DGP seeds. Each seed is an independent draw of both the data and its truth, so the
        number of seeds sets how precisely coverage can be estimated.
    n_periods
        Days of daily data per replication.
    num_warmup, num_samples, num_chains
        Sampler budget per replication. Two chains is the minimum for a meaningful R-hat.
    coverage_tolerance
        Allowed absolute deviation of observed coverage from nominal before the gate fails.
    target_accept_prob
        NUTS acceptance target. Higher than the sampler default because adstock and
        saturation produce curved geometry that the default step size does not handle: at
        0.9 this benchmark saw up to 13% divergent transitions on some seeds.
    max_r_hat, min_ess_bulk, max_divergence_rate
        Sampler health thresholds. A replication breaching any of these is reported and
        excluded from coverage, because a fit that did not explore the posterior properly
        says nothing about the method. Divergences matter as much as R-hat here: a divergent
        chain systematically misses part of the posterior, so its intervals are the wrong
        width and its coverage is not evidence of anything.
    calibrated_channels
        Channels given a simulated incrementality experiment. Chosen by spend rank rather
        than by which channels the model estimates worst — picking them after seeing the
        errors would make the comparison meaningless.
    experiment_days, experiment_relative_se
        Test window length and the experiment's standard error as a fraction of the true
        incremental revenue. 20% is a realistic figure for a well-powered geo test.
    """

    name: str
    seeds: tuple[int, ...]
    n_periods: int
    num_warmup: int
    num_samples: int
    num_chains: int = 2
    target_accept_prob: float = 0.95
    calibrated_channels: tuple[str, ...] = ()
    experiment_days: int = 42
    experiment_relative_se: float = 0.20
    coverage_tolerance: float = 0.10
    max_r_hat: float = 1.05
    min_ess_bulk: float = 100.0
    max_divergence_rate: float = 0.02

    def __post_init__(self) -> None:
        """Reject configurations whose diagnostics cannot be computed.

        R-hat compares variance between chains, so a single-chain run yields NaN — and NaN
        fails every health comparison silently, excluding all replications and producing an
        empty report that still looks like a successful benchmark.
        """
        if self.num_chains < 2:
            msg = f"num_chains must be at least 2 for R-hat to be defined, got {self.num_chains}"
            raise ValueError(msg)
        if not self.seeds:
            msg = "at least one seed is required"
            raise ValueError(msg)


#: Smoke profile. Too few seeds for coverage to be meaningful; checks the harness and sampler.
FAST = BenchmarkProfile(
    name="fast",
    seeds=(0, 1),
    n_periods=240,
    num_warmup=150,
    num_samples=150,
    coverage_tolerance=1.0,  # effectively disabled: 2 seeds cannot estimate coverage
    min_ess_bulk=20.0,
    # A 150-draw smoke run cannot meet the publication divergence bar and is not meant to.
    max_divergence_rate=1.0,
)

#: Publication profile: MMM alone, no experiment calibration.
FULL = BenchmarkProfile(
    name="full",
    seeds=tuple(range(12)),
    n_periods=1_095,
    num_warmup=500,
    num_samples=500,
)

#: The same replications, with the calibration bridge switched on. Identical seeds and
#: sampler budget, so the two tables differ only by the experiment likelihood.
#: Search and social are the two highest-spend channels — the ones an advertiser would
#: actually pay to test — and were chosen on that rule, before any results were seen.
CALIBRATED = BenchmarkProfile(
    name="calibrated",
    seeds=tuple(range(12)),
    n_periods=1_095,
    num_warmup=500,
    num_samples=500,
    calibrated_channels=("search", "social"),
)

PROFILES: dict[str, BenchmarkProfile] = {
    "fast": FAST,
    "full": FULL,
    "calibrated": CALIBRATED,
}

#: Model specification used throughout the benchmark. TV is the one delayed-peak channel.
BENCHMARK_CHANNELS: tuple[ChannelModel, ...] = tuple(
    ChannelModel(slug=c.slug, adstock="weibull" if c.slug == "tv" else "geometric")
    for c in DEFAULT_CHANNELS
)


def _truth_table(panel_truth_roas: dict[str, float], data: MMMData) -> list[dict[str, object]]:
    """Assemble true parameter values on the same scale the model estimates them.

    Half-saturation is divided by the channel's median spend and the coefficient by mean
    revenue, matching the scaling applied in :func:`liftlab.model.prepare_data`. Getting this
    wrong would produce a benchmark that measures a units error rather than estimation error.
    """
    rows: list[dict[str, object]] = []
    geo_position = 0
    for index, spec in enumerate(DEFAULT_CHANNELS):
        rows.append(
            {
                "channel": spec.slug,
                "parameter": "roas",
                "truth": panel_truth_roas[spec.slug],
                "site": "roas",
                "position": index,
            }
        )
        rows.append(
            {
                "channel": spec.slug,
                "parameter": "half_saturation",
                "truth": spec.half_saturation / float(data.spend_scale[index]),
                "site": "half_saturation",
                "position": index,
            }
        )
        rows.append(
            {
                "channel": spec.slug,
                "parameter": "beta",
                "truth": spec.coefficient / data.revenue_scale,
                "site": "beta",
                "position": index,
            }
        )
        rows.append(
            {
                "channel": spec.slug,
                "parameter": "slope",
                "truth": spec.slope,
                "site": "slope",
                "position": index,
            }
        )
        if isinstance(spec.adstock, GeometricAdstock):
            rows.append(
                {
                    "channel": spec.slug,
                    "parameter": "decay",
                    "truth": spec.adstock.decay,
                    "site": "decay",
                    "position": geo_position,
                }
            )
            geo_position += 1
    return rows


def _simulate_experiments(
    panel: SyntheticPanel,
    profile: BenchmarkProfile,
    seed: int,
) -> tuple[ExperimentResult, ...]:
    """Simulate incrementality experiments from the known truth.

    The experiment measures the channel's *actual* incremental revenue over the window,
    observed with Gaussian error. That models a valid, unbiased test — which is the
    assumption the calibration bridge is built on, and the assumption most likely to be
    violated in practice. A geo test with contaminated control markets, or one whose
    creative differs from business-as-usual, is biased, and the bridge would faithfully
    propagate that bias into the MMM. This benchmark measures what calibration buys when
    the experiment is sound; it does not measure what happens when it is not.

    The experiment RNG is separate from the DGP's so that turning calibration on does not
    change the data being fitted.
    """
    if not profile.calibrated_channels:
        return ()

    rng = np.random.default_rng(seed + 10_000)
    index = panel.data.index
    first = int(0.4 * len(index))
    last = min(first + profile.experiment_days - 1, len(index) - 1)
    window = np.zeros(len(index), dtype=bool)
    window[first : last + 1] = True

    experiments = []
    for slug in profile.calibrated_channels:
        true_incremental = float(panel.truth.contributions[slug].to_numpy()[window].sum())
        standard_error = profile.experiment_relative_se * true_incremental
        experiments.append(
            ExperimentResult(
                channel=slug,
                start=index[first].date(),
                end=index[last].date(),
                incremental_revenue=true_incremental + float(rng.normal(0.0, standard_error)),
                standard_error=standard_error,
            )
        )
    return tuple(experiments)


def _diagnostics(idata: az.InferenceData) -> tuple[float, float, int]:
    """Return worst R-hat, smallest bulk ESS, and divergence count."""
    import arviz as az_

    summary = az_.summary(
        idata,
        var_names=["beta", "half_saturation", "slope", "sigma"],
        kind="diagnostics",
    )
    divergences = int(idata.sample_stats["diverging"].sum())
    return float(summary["r_hat"].max()), float(summary["ess_bulk"].min()), divergences


def run_seed(seed: int, profile: BenchmarkProfile) -> pd.DataFrame:
    """Run one replication: generate, fit, and score every tracked parameter.

    Parameters
    ----------
    seed
        DGP seed. Also offsets the sampler seed, so replications are fully independent.
    profile
        Benchmark configuration.

    Returns
    -------
    pandas.DataFrame
        One row per tracked parameter, with truth, posterior summary, and coverage flags.
    """
    panel = generate_panel(DGPConfig(n_periods=profile.n_periods, seed=seed))
    experiments = _simulate_experiments(panel, profile, seed)
    data = prepare_data(panel.data, BENCHMARK_CHANNELS, experiments=experiments)

    started = time.time()
    idata = fit(
        data,
        num_warmup=profile.num_warmup,
        num_samples=profile.num_samples,
        num_chains=profile.num_chains,
        seed=seed,
        target_accept_prob=profile.target_accept_prob,
    )
    elapsed = time.time() - started
    max_r_hat, min_ess, divergences = _diagnostics(idata)
    divergence_rate = divergences / (profile.num_samples * profile.num_chains)

    records = []
    for row in _truth_table(panel.truth.roas, data):
        draws = idata.posterior[row["site"]].values[..., row["position"]].ravel()
        truth = float(row["truth"])  # type: ignore[arg-type]
        lo95, hi95 = np.quantile(draws, [0.025, 0.975])
        lo50, hi50 = np.quantile(draws, [0.25, 0.75])
        mean = float(draws.mean())
        median = float(np.median(draws))
        records.append(
            {
                "seed": seed,
                "channel": row["channel"],
                "calibrated": row["channel"] in profile.calibrated_channels,
                "parameter": row["parameter"],
                "truth": truth,
                "posterior_mean": mean,
                "posterior_median": median,
                "posterior_sd": float(draws.std()),
                "q025": float(lo95),
                "q975": float(hi95),
                "relative_bias": (mean - truth) / truth,
                # The mean of a right-skewed posterior sits above its centre by
                # construction; scoring the median separates that estimator artifact
                # from genuine estimation bias.
                "relative_bias_median": (median - truth) / truth,
                "covered_95": bool(lo95 <= truth <= hi95),
                "covered_50": bool(lo50 <= truth <= hi50),
                "interval_width_95": float(hi95 - lo95),
                "max_r_hat": max_r_hat,
                "min_ess_bulk": min_ess,
                "divergences": divergences,
                "divergence_rate": divergence_rate,
                "seconds": elapsed,
            }
        )
    return pd.DataFrame.from_records(records)


def run_benchmark(profile: BenchmarkProfile, *, verbose: bool = True) -> pd.DataFrame:
    """Run every seed in a profile and concatenate the per-parameter results."""
    frames = []
    for position, seed in enumerate(profile.seeds, start=1):
        if verbose:
            print(
                f"[{position}/{len(profile.seeds)}] seed {seed} ...",
                end=" ",
                flush=True,
            )
        frame = run_seed(seed, profile)
        if verbose:
            print(
                f"{frame['seconds'].iloc[0]:.0f}s  "
                f"r_hat={frame['max_r_hat'].iloc[0]:.3f}  "
                f"div={int(frame['divergences'].iloc[0])}",
                flush=True,
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarise(results: pd.DataFrame, profile: BenchmarkProfile) -> pd.DataFrame:
    """Aggregate per-seed results into a per-parameter report.

    Replications whose sampler diagnostics breach the profile's thresholds are excluded:
    a fit that did not converge measures the sampler budget, not the method.
    """
    grouped = results[_healthy_mask(results, profile)].groupby("parameter", sort=True)
    summary = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean_relative_bias": grouped["relative_bias"].mean(),
            "mean_relative_bias_median": grouped["relative_bias_median"].mean(),
            "median_abs_relative_bias": grouped["relative_bias"].apply(lambda s: s.abs().median()),
            "coverage_95": grouped["covered_95"].mean(),
            "coverage_50": grouped["covered_50"].mean(),
            "median_interval_width": grouped["interval_width_95"].median(),
        }
    )
    return summary.sort_index()


def _healthy_mask(results: pd.DataFrame, profile: BenchmarkProfile) -> pd.Series:
    """Boolean mask of replications whose sampler diagnostics are acceptable."""
    return (
        (results["max_r_hat"] <= profile.max_r_hat)
        & (results["min_ess_bulk"] >= profile.min_ess_bulk)
        & (results["divergence_rate"] <= profile.max_divergence_rate)
    )


def check_gate(summary: pd.DataFrame, profile: BenchmarkProfile) -> list[str]:
    """Return human-readable gate failures; empty means the benchmark passed.

    An empty summary is a failure, not a pass. If every replication was excluded for sampler
    health there is nothing to check, and iterating over no rows would otherwise report
    success for a benchmark that measured nothing at all.
    """
    if summary.empty:
        return ["no replications passed the sampler-health thresholds; nothing was measured"]

    failures = []
    for parameter, row in summary.iterrows():
        deviation = abs(float(row["coverage_95"]) - 0.95)
        if deviation > profile.coverage_tolerance:
            failures.append(
                f"{parameter}: 95% interval coverage {row['coverage_95']:.0%} deviates "
                f"{deviation:.0%} from nominal (tolerance {profile.coverage_tolerance:.0%})"
            )
    return failures


def _markdown_table(frame: pd.DataFrame, index_name: str) -> str:
    """Render a frame as a Markdown table.

    Hand-rolled rather than via ``DataFrame.to_markdown``, which needs ``tabulate`` — not a
    dependency worth carrying for one table in one CLI.
    """
    headers = [index_name, *(str(c) for c in frame.columns)]
    rows = [
        [str(idx), *(str(v) for v in row)]
        for idx, row in zip(frame.index, frame.to_numpy(), strict=True)
    ]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)) + " |"
    divider = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = [
        "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider, *body])


def _calibration_section(results: pd.DataFrame, profile: BenchmarkProfile) -> list[str]:
    """Break ROAS accuracy down by whether the channel had an experiment.

    The interesting question is not whether calibration helps the tested channels — it
    should, by construction. It is whether it helps the *untested* ones, which it can,
    because total revenue is conserved: pinning one channel's contribution constrains what
    is left for the others to explain.
    """
    if not profile.calibrated_channels:
        return []

    healthy = results[_healthy_mask(results, profile) & (results["parameter"] == "roas")]
    grouped = healthy.groupby("calibrated", sort=True)
    table = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean rel. bias": grouped["relative_bias"].mean().map(lambda v: f"{v:+.1%}"),
            "median abs rel. bias": grouped["relative_bias"]
            .apply(lambda s: s.abs().median())
            .map(lambda v: f"{v:.1%}"),
            "coverage 95%": grouped["covered_95"].mean().map(lambda v: f"{v:.0%}"),
        }
    )
    table.index = pd.Index(
        ["untested channels" if not flag else "tested channels" for flag in table.index],
        name="channel group",
    )

    return [
        "## Effect of calibration, by channel group",
        "",
        f"Experiments were simulated for **{', '.join(profile.calibrated_channels)}** — a "
        f"{profile.experiment_days}-day window with a "
        f"{profile.experiment_relative_se:.0%} relative standard error.",
        "",
        _markdown_table(table, "channel group"),
        "",
        "The experiment is simulated as unbiased: it measures the channel's true incremental",
        "revenue with Gaussian noise. A real geo test with contaminated controls would be",
        "biased, and the bridge would propagate that bias faithfully. This measures what",
        "calibration buys when the experiment is sound, not what happens when it is not.",
        "",
    ]


def format_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    profile: BenchmarkProfile,
) -> str:
    """Render the benchmark as a Markdown report."""
    replications = results["seed"].nunique()
    unhealthy = results[~_healthy_mask(results, profile)]["seed"].nunique()
    seconds = results.groupby("seed")["seconds"].first().sum()
    per_seed = results.groupby("seed").first()
    worst_divergence_rate = float(per_seed["divergence_rate"].max())

    table = summary.copy()
    table["mean_relative_bias"] = table["mean_relative_bias"].map(lambda v: f"{v:+.1%}")
    table["mean_relative_bias_median"] = table["mean_relative_bias_median"].map(
        lambda v: f"{v:+.1%}"
    )
    table["median_abs_relative_bias"] = table["median_abs_relative_bias"].map(lambda v: f"{v:.1%}")
    table["coverage_95"] = table["coverage_95"].map(lambda v: f"{v:.0%}")
    table["coverage_50"] = table["coverage_50"].map(lambda v: f"{v:.0%}")
    table["median_interval_width"] = table["median_interval_width"].map(lambda v: f"{v:.2f}")
    table.columns = [
        "n",
        "bias (mean est.)",
        "bias (median est.)",
        "median abs rel. bias",
        "coverage 95%",
        "coverage 50%",
        "median 95% width",
    ]

    lines = [
        "# Parameter-recovery benchmark",
        "",
        f"Profile **{profile.name}** — {replications} replications of "
        f"{profile.n_periods} days across {len(DEFAULT_CHANNELS)} channels, "
        f"{profile.num_warmup} warmup / {profile.num_samples} draws "
        f"x {profile.num_chains} chains.",
        "",
        "Reproduce with:",
        "",
        "```bash",
        f"just recover --profile {profile.name}",
        "```",
        "",
        "## Results",
        "",
        _markdown_table(table, "parameter"),
        "",
        "`coverage 95%` is the fraction of replications whose 95% credible interval contained",
        "the true value. Nominal is 95%: materially below means the model is overconfident,",
        "materially above means it is needlessly vague. Coverage matters more than bias — a",
        "biased estimate with an honest interval is usable, an overconfident one is not.",
        "",
        *_calibration_section(results, profile),
        "## Sampler health",
        "",
        f"Replications excluded: **{unhealthy}** of {replications} — R-hat > "
        f"{profile.max_r_hat}, bulk ESS < {profile.min_ess_bulk:.0f}, or divergence rate > "
        f"{profile.max_divergence_rate:.0%}.",
        "",
        f"Worst divergence rate across replications: **{worst_divergence_rate:.1%}**. "
        "Divergences are treated as a hard exclusion criterion rather than a warning: a "
        "divergent chain systematically fails to explore part of the posterior, so its "
        "intervals are the wrong width and its coverage is not evidence about the method.",
        "",
        f"Total sampling time {seconds / 60:.0f} min.",
        "",
    ]
    return "\n".join(lines)


def paired_comparison(
    baseline: pd.DataFrame,
    treatment: pd.DataFrame,
    profile: BenchmarkProfile,
) -> pd.DataFrame:
    """Compare two benchmark arms on the replications healthy in *both*.

    Restricting to the common healthy set is the point of this function. The two arms
    generally exclude different seeds for divergences, so comparing their published summary
    tables directly mixes the effect being measured with a change in which replications were
    counted. Pairing removes that confound.

    Parameters
    ----------
    baseline, treatment
        Raw per-parameter results from two runs over the same seeds.
    profile
        Supplies the health thresholds and the list of channels that were tested.

    Returns
    -------
    pandas.DataFrame
        ROAS accuracy per channel group, for each arm.
    """
    common = sorted(
        set(baseline.loc[_healthy_mask(baseline, profile), "seed"])
        & set(treatment.loc[_healthy_mask(treatment, profile), "seed"])
    )
    if not common:
        msg = "the two arms share no replication that is healthy in both"
        raise ValueError(msg)

    tested = set(profile.calibrated_channels)
    rows = []
    for arm_name, frame in (("uncalibrated", baseline), ("calibrated", treatment)):
        roas = frame[(frame["seed"].isin(common)) & (frame["parameter"] == "roas")]
        groups = {
            "tested": roas[roas["channel"].isin(tested)],
            "untested": roas[~roas["channel"].isin(tested)],
            "all": roas,
        }
        for group_name, subset in groups.items():
            rows.append(
                {
                    "arm": arm_name,
                    "group": group_name,
                    "n": len(subset),
                    "mean_relative_bias": subset["relative_bias"].mean(),
                    "median_abs_relative_bias": subset["relative_bias"].abs().median(),
                    "coverage_95": subset["covered_95"].mean(),
                    "median_interval_width": subset["interval_width_95"].median(),
                }
            )
    result = pd.DataFrame(rows)
    result.attrs["common_seeds"] = common
    return result


def format_comparison(comparison: pd.DataFrame, profile: BenchmarkProfile) -> str:
    """Render the paired comparison as Markdown."""
    common = comparison.attrs.get("common_seeds", [])
    labels = {
        "tested": f"tested ({', '.join(profile.calibrated_channels)})",
        "untested": "untested",
        "all": "all channels",
    }

    rows = []
    for group in ("tested", "untested", "all"):
        before = comparison[(comparison["group"] == group) & (comparison["arm"] == "uncalibrated")]
        after = comparison[(comparison["group"] == group) & (comparison["arm"] == "calibrated")]
        rows.append(
            {
                "n": int(before["n"].iloc[0]),
                "median abs ROAS error": (
                    f"{before['median_abs_relative_bias'].iloc[0]:.1%}"
                    f" → {after['median_abs_relative_bias'].iloc[0]:.1%}"
                ),
                "mean bias": (
                    f"{before['mean_relative_bias'].iloc[0]:+.1%}"
                    f" → {after['mean_relative_bias'].iloc[0]:+.1%}"
                ),
                "coverage 95%": (
                    f"{before['coverage_95'].iloc[0]:.0%} → {after['coverage_95'].iloc[0]:.0%}"
                ),
                "95% width": (
                    f"{before['median_interval_width'].iloc[0]:.2f}"
                    f" → {after['median_interval_width'].iloc[0]:.2f}"
                ),
            }
        )
    table = pd.DataFrame(rows, index=pd.Index([labels[g] for g in ("tested", "untested", "all")]))

    return "\n".join(
        [
            "# Effect of experiment calibration",
            "",
            f"Paired over the **{len(common)}** replications healthy in both arms "
            f"(seeds {', '.join(str(s) for s in common)}). The arms otherwise exclude "
            "different seeds, and comparing their summary tables directly would confound "
            "the effect with a change in which replications were counted.",
            "",
            "Each cell reads *uncalibrated → calibrated*.",
            "",
            _markdown_table(table, "channel group"),
            "",
            "Calibration sharply improves the channels that were tested and barely moves the "
            "ones that were not. Pinning one channel's contribution does constrain what is left "
            "for the others to explain, but that spillover is weak: in practice you have to test "
            "the channel you want a trustworthy number for.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``just recover``."""
    parser = argparse.ArgumentParser(description="Run the parameter-recovery benchmark.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="fast")
    parser.add_argument("--output", type=Path, default=Path("docs/recovery"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if coverage deviates from nominal beyond the profile tolerance.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BASELINE_CSV", "TREATMENT_CSV"),
        help=(
            "Skip sampling and instead pair two existing raw result files, reporting the "
            "effect of calibration over the replications healthy in both."
        ),
    )
    args = parser.parse_args(argv)

    profile = PROFILES[args.profile]

    if args.compare is not None:
        baseline_path, treatment_path = args.compare
        comparison = paired_comparison(
            pd.read_csv(baseline_path), pd.read_csv(treatment_path), profile
        )
        rendered = format_comparison(comparison, profile)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "comparison.md").write_text(rendered, encoding="utf-8")
        print(rendered)
        return 0

    results = run_benchmark(profile)
    summary = summarise(results, profile)
    report = format_report(results, summary, profile)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{profile.name}.md").write_text(report, encoding="utf-8")
    results.to_csv(args.output / f"{profile.name}-raw.csv", index=False)

    print(report)

    failures = check_gate(summary, profile)
    if failures:
        print("\nGATE FAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        if args.check:
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
