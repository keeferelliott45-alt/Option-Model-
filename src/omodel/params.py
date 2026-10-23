from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Source = Literal["theory", "convention", "data-fit"]


@dataclass(frozen=True)
class Param:
    name: str
    value: Any
    source: Source
    layer: str
    rationale: str


REGISTRY: dict[str, Param] = {}


def register(name: str, value: Any, source: Source, layer: str, rationale: str) -> Any:
    if name in REGISTRY and REGISTRY[name].value != value:
        raise ValueError(f"parameter {name} re-registered with a different value")
    REGISTRY[name] = Param(name, value, source, layer, rationale)
    return value


def get(name: str) -> Any:
    return REGISTRY[name].value


def render_markdown() -> str:
    rows = ["| Parameter | Value | Source | Layer | Rationale |", "|---|---|---|---|---|"]
    for p in sorted(REGISTRY.values(), key=lambda p: (p.layer, p.name)):
        rows.append(f"| `{p.name}` | `{p.value!r}` | {p.source} | {p.layer} | {p.rationale} |")
    n_fit = sum(p.source == "data-fit" for p in REGISTRY.values())
    header = (
        "# Parameter registry\n\n"
        f"{len(REGISTRY)} parameters, {n_fit} fit to data. Data-fit parameters are only ever chosen inside "
        "walk-forward fit windows and get the most out-of-sample scrutiny.\n\n"
    )
    return header + "\n".join(rows) + "\n"


# --- Layer 2: move decomposition -------------------------------------------------------------
BETA_WINDOW = register("BETA_WINDOW", 90, "theory", "L2",
                       "Thesis: rolling 60-90d betas; upper end for estimate stability.")
BETA_MIN_OBS = register("BETA_MIN_OBS", 60, "convention", "L2",
                        "Minimum valid days inside the beta window before a fit is trusted.")
SECTOR_WINDOW = register("SECTOR_WINDOW", 252, "convention", "L2",
                         "One trading year of history to pick each stock's sector ETF.")
SECTOR_REASSIGN_FREQ = register("SECTOR_REASSIGN_FREQ", "ME", "convention", "L2",
                                "Sector assignment refreshed month-end using only prior data.")
VOLUME_Z_WINDOW = register("VOLUME_Z_WINDOW", 20, "theory", "L2",
                           "Thesis: volume vs 20-day average.")
MIN_PRICE = register("MIN_PRICE", 5.0, "convention", "L2",
                     "Exclude sub-$5 prints where spreads and tick size dominate returns.")

# --- Event study / backtest ------------------------------------------------------------------
EVENT_Z_THRESHOLD = register("EVENT_Z_THRESHOLD", -3.0, "theory", "L2",
                             "Pre-registered: 3-sigma residual drop = 'large unexplained move'.")
EVENT_Z_GRID = register("EVENT_Z_GRID", (-2.5, -3.0, -4.0), "theory", "L2",
                        "Only thresholds the walk-forward selector may choose between.")
HORIZONS = register("HORIZONS", (1, 5, 15, 30), "theory", "L2/L4",
                    "Thesis: forward returns at 1, 5, 15, 30 trading days.")
PRIMARY_HORIZON = register("PRIMARY_HORIZON", 15, "theory", "L2",
                           "Pre-registered primary horizon (thesis worked example uses 15d).")
EVENT_COOLDOWN = register("EVENT_COOLDOWN", 15, "convention", "L2",
                          "Suppress same-ticker events inside the primary horizon to avoid double counting.")
ENTRY = register("ENTRY", "next_open", "convention", "backtest",
                 "Signal is known at close t; first tradable price is the t+1 open.")
COST_BPS_PER_SIDE = register("COST_BPS_PER_SIDE", 5.0, "convention", "backtest",
                             "Conservative half-spread + slippage for S&P 500 names, per side.")
WF_FIT_YEARS = register("WF_FIT_YEARS", 2, "theory", "backtest", "Thesis: fit on rolling ~2 years.")
WF_TEST_MONTHS = register("WF_TEST_MONTHS", 6, "theory", "backtest", "Thesis: test on next 3-6 months.")
MIN_SAMPLE_FINDING = register("MIN_SAMPLE_FINDING", 30, "theory", "backtest",
                              "Thesis: under ~30 instances is low-confidence, not a finding.")
BOOTSTRAP_REPS = register("BOOTSTRAP_REPS", 2000, "convention", "backtest",
                          "Cluster-bootstrap replications for confidence intervals.")
CI_LEVEL = register("CI_LEVEL", 0.95, "convention", "backtest", "Two-sided confidence level.")
SUSPICIOUS_SHARPE = register("SUSPICIOUS_SHARPE", 2.0, "theory", "backtest",
                             "Thesis: Sharpe above ~2-3 is treated as a probable bug until proven otherwise.")
FDR_Q = register("FDR_Q", 0.10, "convention", "backtest",
                 "Benjamini-Hochberg false-discovery rate across one-factor-at-a-time tests.")
EXPANDING_MIN_OBS = register("EXPANDING_MIN_OBS", 252, "convention", "backtest",
                             "Minimum history before an expanding-window quantile is defined.")
RANDOM_SEED = register("RANDOM_SEED", 20261023, "convention", "backtest", "Fixed seed for reproducibility.")
