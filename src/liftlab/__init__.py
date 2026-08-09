"""liftlab — Bayesian marketing mix modeling calibrated by geo-incrementality experiments."""

from liftlab.dgp import (
    DEFAULT_CHANNELS,
    ChannelSpec,
    DGPConfig,
    GeometricAdstock,
    GroundTruth,
    SyntheticPanel,
    WeibullAdstock,
    generate_panel,
)
from liftlab.israel_calendar import (
    holiday_frame,
    jewish_holiday_dates,
    weekly_seasonality,
)
from liftlab.optimizer import (
    Allocation,
    ResponseCurves,
    optimize_budget,
)
from liftlab.transforms import (
    geometric_adstock,
    hill_saturation,
    weibull_adstock,
    weibull_weights,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_CHANNELS",
    "Allocation",
    "ChannelSpec",
    "DGPConfig",
    "GeometricAdstock",
    "GroundTruth",
    "ResponseCurves",
    "SyntheticPanel",
    "WeibullAdstock",
    "__version__",
    "generate_panel",
    "geometric_adstock",
    "hill_saturation",
    "holiday_frame",
    "jewish_holiday_dates",
    "optimize_budget",
    "weekly_seasonality",
    "weibull_adstock",
    "weibull_weights",
]
