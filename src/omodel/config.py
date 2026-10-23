from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MARKET_ETF = "SPY"
SECTOR_ETFS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    data_dir: Path = Path(os.environ.get("OMODEL_DATA_DIR", ROOT / "data"))
    reports_dir: Path = ROOT / "reports"
    dev_start: pd.Timestamp = pd.Timestamp("2000-01-03")
    # Fixed date, never relative to "today": a rolling holdout would silently leak into development over time.
    holdout_start: pd.Timestamp = pd.Timestamp("2025-10-23")
    sec_user_agent: str = os.environ.get("SEC_USER_AGENT", "")
    http_timeout: float = 60.0


_settings = Settings()


def settings() -> Settings:
    return _settings


def override(**changes) -> Settings:
    global _settings
    _settings = dataclasses.replace(_settings, **changes)
    return _settings


def reset() -> Settings:
    global _settings
    _settings = Settings()
    return _settings
