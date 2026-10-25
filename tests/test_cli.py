import subprocess
import sys

import pandas as pd
import pytest

from omodel import cli, config, params
from omodel.data import http, yahoo


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("CLI unit tests must not touch the network")

    monkeypatch.setattr(http, "get", refuse)


@pytest.mark.parametrize("argv, command", [
    (["fetch-prices"], "fetch-prices"),
    (["fetch-prices", "--refresh", "--tickers", "AAPL", "MSFT"], "fetch-prices"),
    (["fetch-macro", "--refresh"], "fetch-macro"),
    (["fetch-sec"], "fetch-sec"),
    (["snapshot-options", "--tickers", "AAPL"], "snapshot-options"),
    (["phase1", "--no-news", "--refresh"], "phase1"),
    (["params", "--print"], "params"),
])
def test_parser_accepts_every_subcommand(argv, command):
    args = cli.build_parser().parse_args(argv)
    assert args.command == command and callable(args.func)


def test_parser_rejects_missing_or_unknown_command():
    for argv in ([], ["bogus"]):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(argv)


def test_snapshot_defaults_to_current_members(monkeypatch):
    members = pd.DataFrame({"ticker": ["AAA", "BBB", "OLD"],
                            "end_date": [pd.NaT, pd.NaT, pd.Timestamp("2010-01-01")]})
    monkeypatch.setattr(cli.universe, "load_membership", lambda refresh=False: members)
    from omodel.experiments import scanner

    def offline(*a, **k):
        raise ConnectionError("screener offline")

    monkeypatch.setattr(scanner, "scan_universe", offline)  # falls back to S&P members
    seen = {}
    monkeypatch.setattr(cli.cboe_options, "snapshot", lambda t: seen.setdefault("t", t) and [])
    assert cli.main(["snapshot-options"]) == 0
    assert seen["t"] == ["AAA", "BBB"]


def test_flags_default_off():
    args = cli.build_parser().parse_args(["phase1"])
    assert not args.no_news and not args.refresh
    args = cli.build_parser().parse_args(["fetch-prices"])
    assert args.tickers is None and not args.refresh


def test_phase1_passes_flags_and_returns_zero(monkeypatch, capsys):
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return {"verdict": "FAIL", "criteria": [{"id": "1", "criterion": "x", "passed": False}],
                "paths": {"report": "r.md"}}

    monkeypatch.setattr(cli.phase1, "run", fake_run)
    assert cli.main(["phase1", "--no-news"]) == 0
    assert seen == {"news": False, "refresh": False}
    out = capsys.readouterr().out
    assert "phase1 verdict: FAIL" in out and "r.md" in out


def test_params_writes_generated_doc(tmp_path):
    config.override(root=tmp_path)
    assert cli.main(["params"]) == 0
    assert (tmp_path / "docs" / "PARAMETERS.md").read_text(encoding="utf-8") == params.render_markdown()


def test_params_print_includes_registry_from_every_module(capsys):
    assert cli.main(["params", "--print"]) == 0
    out = capsys.readouterr().out
    for name in ("BETA_WINDOW", "CLUSTER_FREQ", "PANEL_WARMUP_DAYS", "SEC_NEWS_WINDOW", "SUBPERIODS"):
        assert f"`{name}`" in out


def test_fetch_prices_with_tickers_and_load_error(monkeypatch, capsys):
    calls = []

    def fake_load_many(tickers, refresh=False, strict=False):
        calls.append((list(tickers), refresh, strict))
        return {t: pd.DataFrame({"close": [1.0]}) for t in tickers}

    monkeypatch.setattr(yahoo, "load_many", fake_load_many)
    assert cli.main(["fetch-prices", "--tickers", "AAA", "BBB", "--refresh"]) == 0
    assert calls == [(["AAA", "BBB"], True, True)]

    def failing(*a, **k):
        raise yahoo.LoadError(["AAA"])

    monkeypatch.setattr(yahoo, "load_many", failing)
    assert cli.main(["fetch-prices", "--tickers", "AAA"]) == 1
    assert "AAA" in capsys.readouterr().err


def test_fetch_sec_requires_user_agent(capsys):
    config.override(sec_user_agent="")
    assert cli.main(["fetch-sec"]) == 2
    assert "SEC_USER_AGENT" in capsys.readouterr().err


def test_fetch_macro_and_snapshot_delegate(monkeypatch, capsys):
    idx = pd.bdate_range("2020-01-01", periods=3, name="date")
    monkeypatch.setattr(cli.macro, "load_macro_panel", lambda refresh=False: pd.DataFrame({"vix": [1.0, 2.0, None]},
                                                                                         index=idx))
    assert cli.main(["fetch-macro"]) == 0
    assert "vix 67%" in capsys.readouterr().out
    got = []

    def fake_snapshot(tickers):
        got.extend(tickers)
        return []

    monkeypatch.setattr(cli.cboe_options, "snapshot", fake_snapshot)
    assert cli.main(["snapshot-options", "--tickers", "AAPL", "MSFT"]) == 0
    assert got == ["AAPL", "MSFT"]


def test_python_dash_m_entry_point():
    proc = subprocess.run([sys.executable, "-m", "omodel", "--help"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "phase1" in proc.stdout
