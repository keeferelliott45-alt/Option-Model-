from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

import pandas as pd

from omodel import holdout
from omodel.config import settings
from omodel.data import cboe_options, macro, sec, universe, yahoo
from omodel.data import panel as panel_mod
from omodel.experiments import phase1


def _dev_tickers(refresh: bool = False) -> list[str]:
    membership = universe.load_membership(refresh=refresh)
    start = settings().dev_start - pd.Timedelta(days=panel_mod.PANEL_WARMUP_DAYS)
    return universe.tickers_between(membership, start, holdout.holdout_start() - pd.Timedelta(days=1))


def _fetch_prices(args: argparse.Namespace) -> int:
    try:
        if args.tickers:
            data = yahoo.load_many(args.tickers, refresh=args.refresh, strict=True)
            print(f"prices: {len(data)} of {len(set(args.tickers))} tickers have data")
            return 0
        membership = universe.load_membership(refresh=args.refresh)
        p = panel_mod.build_panel(refresh=args.refresh, membership=membership)
    except yahoo.LoadError as exc:
        print(f"prices: {exc}", file=sys.stderr)
        return 1
    print(f"prices: panel {p.dates[0].date()}..{p.dates[-1].date()}, {len(p.dates)} days, {len(p.tickers)} tickers")
    return 0


def _fetch_macro(args: argparse.Namespace) -> int:
    df = macro.load_macro_panel(refresh=args.refresh)
    if df.empty:
        print("macro: no data", file=sys.stderr)
        return 1
    filled = ", ".join(f"{c} {df[c].notna().mean():.0%}" for c in df.columns)
    print(f"macro: {df.index[0].date()}..{df.index[-1].date()}, {len(df)} rows; non-null share: {filled}")
    return 0


def _fetch_sec(args: argparse.Namespace) -> int:
    if not settings().sec_user_agent.strip():
        print("fetch-sec: set SEC_USER_AGENT='Your Name your@email' first (SEC requires contact info)",
              file=sys.stderr)
        return 2
    tickers = _dev_tickers()
    days = pd.bdate_range(settings().dev_start, holdout.holdout_start() - pd.Timedelta(days=1))
    _, missing = sec.news_flags(tickers, days, refresh=args.refresh)
    print(f"sec: filings cached for {len(tickers) - len(missing)} of {len(tickers)} tickers; "
          f"{len(missing)} have no CIK in SEC's current map")
    return 0


def _snapshot_options(args: argparse.Namespace) -> int:
    paths = cboe_options.snapshot(args.tickers)
    print(f"options: {len(paths)} new chain snapshot(s) written for {len(set(args.tickers))} ticker(s)")
    return 0


def _phase1(args: argparse.Namespace) -> int:
    res = phase1.run(news=not args.no_news, refresh=args.refresh)
    print(f"phase1 verdict: {res['verdict']}")
    for c in res["criteria"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['id']}: {c['criterion']}")
    for name, path in (res.get("paths") or {}).items():
        print(f"  {name}: {path}")
    return 0


def _params(args: argparse.Namespace) -> int:
    if args.print:
        phase1.import_all()
        from omodel import params

        enc = sys.stdout.encoding or "utf-8"
        sys.stdout.write(params.render_markdown().encode(enc, "replace").decode(enc))
        return 0
    print(f"params: wrote {phase1.write_parameters_doc()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omodel", description="Options overreaction research system (never trades).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-prices", help="Fetch Yahoo daily prices (all dev-window S&P 500 members by default).")
    p.add_argument("--refresh", action="store_true", help="Refetch even if cached.")
    p.add_argument("--tickers", nargs="+", help="Only these tickers (no panel build).")
    p.set_defaults(func=_fetch_prices)

    p = sub.add_parser("fetch-macro", help="Fetch and cache the Layer 1 macro panel.")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(func=_fetch_macro)

    p = sub.add_parser("fetch-sec", help="Cache SEC filing histories for dev-window members (needs SEC_USER_AGENT).")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(func=_fetch_sec)

    p = sub.add_parser("snapshot-options", help="Store today's CBOE delayed option chains (run after ~16:30 ET).")
    p.add_argument("--tickers", nargs="+", required=True)
    p.set_defaults(func=_snapshot_options)

    p = sub.add_parser("phase1", help="Run the pre-registered Phase 1 event study and write the report.")
    p.add_argument("--no-news", action="store_true", help="Skip the SEC 8-K confirming signal.")
    p.add_argument("--refresh", action="store_true", help="Refetch all source data first.")
    p.set_defaults(func=_phase1)

    p = sub.add_parser("params", help="Regenerate docs/PARAMETERS.md from the parameter registry.")
    p.add_argument("--print", action="store_true", help="Print the table instead of writing the file.")
    p.set_defaults(func=_params)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args))
