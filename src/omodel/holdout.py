from __future__ import annotations

import datetime as _dt
import os
import traceback

import pandas as pd

from omodel.config import settings

UNLOCK_ENV = "OMODEL_UNLOCK_HOLDOUT"
UNLOCK_VALUE = "I_HAVE_FROZEN_THE_DESIGN"


class HoldoutViolation(RuntimeError):
    pass


def holdout_start() -> pd.Timestamp:
    return settings().holdout_start


def is_unlocked() -> bool:
    return os.environ.get(UNLOCK_ENV) == UNLOCK_VALUE


def _log_access(what: str) -> None:
    path = settings().data_dir / "holdout_access.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    caller = traceback.extract_stack(limit=4)[0]
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp}\t{what}\t{caller.filename}:{caller.lineno}\n")


def access_count() -> int:
    path = settings().data_dir / "holdout_access.log"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def guard(obj: pd.DataFrame | pd.Series, what: str = "data") -> pd.DataFrame | pd.Series:
    """Drop rows dated on/after the holdout start unless the holdout was explicitly unlocked."""
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError("holdout.guard requires a DatetimeIndex")
    cutoff = holdout_start()
    if is_unlocked():
        if (obj.index >= cutoff).any():
            _log_access(what)
        return obj
    return obj.loc[obj.index < cutoff]


def assert_clean(index: pd.Index, what: str = "data") -> None:
    if is_unlocked():
        return
    idx = pd.DatetimeIndex(index)
    if len(idx) and idx.max() >= holdout_start():
        raise HoldoutViolation(f"{what} contains dates on/after holdout start {holdout_start().date()}")
