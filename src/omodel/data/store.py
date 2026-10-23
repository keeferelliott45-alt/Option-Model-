from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from omodel.config import settings


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]", "_", key)


def path_for(kind: str, key: str, ext: str = "parquet") -> Path:
    return settings().data_dir / kind / f"{_safe(key)}.{ext}"


def read(kind: str, key: str) -> pd.DataFrame | None:
    path = path_for(kind, key)
    return pd.read_parquet(path) if path.exists() else None


def write(df: pd.DataFrame, kind: str, key: str, meta: dict[str, Any] | None = None) -> Path:
    path = path_for(kind, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    sidecar = {
        "fetched_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        **(meta or {}),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    return path


def read_meta(kind: str, key: str) -> dict | None:
    path = path_for(kind, key).with_suffix(".meta.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_json(kind: str, key: str) -> Any | None:
    path = path_for(kind, key, "json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_json(obj: Any, kind: str, key: str) -> Path:
    path = path_for(kind, key, "json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path
