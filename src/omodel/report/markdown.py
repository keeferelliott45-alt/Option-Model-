from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Sequence
from typing import Any

from omodel import params

NA = "n/a"
LEVEL = f"{params.CI_LEVEL:.0%}"


def is_num(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)


def pct(x: Any, digits: int = 2, signed: bool = True) -> str:
    if not is_num(x):
        return NA
    return f"{float(x) * 100:{'+' if signed else ''}.{digits}f}%"


def bp(x: Any, digits: int = 1) -> str:
    return f"{float(x) * 1e4:+.{digits}f} bp" if is_num(x) else NA


def num(x: Any, digits: int = 2) -> str:
    return f"{float(x):.{digits}f}" if is_num(x) else NA


def count(n: Any) -> str:
    return f"{int(n):,}" if is_num(n) else NA


def day(x: Any) -> str:
    return str(x)[:10] if x not in (None, "") else NA


def ci(lo: Any, hi: Any, fmt: Callable[[Any], str] = pct) -> str:
    return f"[{fmt(lo)}, {fmt(hi)}]"


def mean_ci(s: dict, fmt: Callable[[Any], str] = pct) -> str:
    """Compact cell: mean [CI] (n), flagged when low-confidence."""
    if not s or not s.get("n"):
        return "no events (n=0)"
    flag = " (low-conf.)" if s.get("low_confidence") else ""
    return f"{fmt(s.get('mean'))} {ci(s.get('ci_lo'), s.get('ci_hi'), fmt)} (n={count(s.get('n'))}){flag}"


def stat_line(s: dict, fmt: Callable[[Any], str] = pct) -> str:
    if not s or not s.get("n"):
        return "no events (n=0): nothing to estimate."
    line = (f"mean {fmt(s.get('mean'))}, {LEVEL} CI {ci(s.get('ci_lo'), s.get('ci_hi'), fmt)}; n={count(s.get('n'))} "
            f"events in {count(s.get('n_clusters'))} months (effective {num(s.get('n_eff_clusters'), 0)}); "
            f"median {fmt(s.get('median'))}; share positive {pct(s.get('win_rate'), 1, False)} "
            f"{ci(s.get('win_lo'), s.get('win_hi'), lambda v: pct(v, 1, False))}")
    if s.get("low_confidence"):
        line += " - **LOW CONFIDENCE (too few events or effective months): not a finding**"
    return line


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_(no rows)_"
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(out)


def passfail(ok: Any) -> str:
    return "PASS" if ok else "FAIL"


def _criterion_value(c: dict) -> str:
    v = c.get("value")
    if c["id"] == "4":
        return f"{count(v)} positive" if is_num(v) else NA
    if c["id"] == "5":
        return num(v)
    out = pct(v)
    if c.get("ci"):
        out += f" {ci(*c['ci'])}"
    if c.get("n") is not None:
        out += f", n={count(c['n'])}"
    return out


def _criteria_section(r: dict) -> list[str]:
    rows = [[c["id"], c["criterion"], _criterion_value(c), c["threshold"], passfail(c["passed"])]
            for c in r["criteria"]]
    return ["## Decision: pre-registered criteria", "",
            "All must pass (PREREGISTRATION.md, including Amendment 1: criterion 1 needs both 1 and 1b).", "",
            table(["#", "Criterion", f"Value ({LEVEL} CI, n)", "Threshold", "Result"], rows), ""]


def _tested_section(m: dict, p: dict) -> list[str]:
    h = m.get("primary_horizon")
    cost = m.get("cost_bps_per_side")
    return [
        "## What was tested", "",
        f"- **Hypothesis H1** (docs/PREREGISTRATION.md): after a day whose standardized residual is z <= "
        f"{m.get('primary_z')}, the abnormal log return from the **t+1 open to the t+{h} close**, net of costs, "
        "has a positive mean.",
        f"- **Residual model:** stock return = alpha + beta_mkt * SPY + beta_sec * sector ETF, betas fit on the prior "
        f"{_pv(p, 'BETA_WINDOW')} trading days only (min {_pv(p, 'BETA_MIN_OBS')} obs); the sector ETF is re-picked "
        "each month-end from the prior year of data. z = residual / residual std of the fit window.",
        f"- **Events:** point-in-time S&P 500 members on the event date, as-traded price >= "
        f"${_pv(p, 'MIN_PRICE')}, same-ticker events within {_pv(p, 'EVENT_COOLDOWN')} days suppressed.",
        f"- **Return:** stock minus (beta_mkt * SPY + beta_sec * sector ETF) over the same open-to-close window with "
        f"the betas as of the event date (alpha excluded), minus {num(cost, 1)} bp per side "
        f"({num(2 * cost if is_num(cost) else None, 1)} bp round trip).",
        f"- **Statistics:** means with {LEVEL} CIs from a calendar-month clustered bootstrap "
        f"({count(_pv(p, 'BOOTSTRAP_REPS'))} reps); anything under {_pv(p, 'MIN_SAMPLE_FINDING')} events or "
        f"{_pv(p, 'MIN_EFFECTIVE_CLUSTERS')} effective months is low-confidence, not a finding.",
        f"- **Development window** {day(m.get('dev_start'))} to {day(m.get('dev_end'))}; the holdout from "
        f"{day(m.get('holdout_start'))} was {'UNLOCKED' if m.get('holdout_unlocked') else 'not touched'}.", ""]


def _readings_section(m: dict) -> list[str]:
    readings = m.get("readings") or []
    if not readings:
        return []
    intro = ("Choices made in code that PREREGISTRATION.md does not spell out. They were fixed before any real-data "
             "result and belong verbatim in a dated amendment before the first real run "
             f"(readings sha256 `{m.get('readings_sha256')}`).")
    return ["## Implementation readings", "", intro, ""] + [f"{i}. {x}" for i, x in enumerate(readings, 1)] + [""]


def _provenance_lines(m: dict) -> list[str]:
    prov = m.get("provenance") or {}
    status = prov.get("git_status", "unknown")
    head = prov.get("git_head")
    line = f"Code: git `{head[:12] if head else 'none'}` ({status}); code sha256 `{prov.get('code_sha256') or NA}`."
    if status == "clean":
        return [line]
    files = prov.get("uncommitted") or []
    listed = f" Uncommitted: {', '.join(files[:10])}{', ...' if len(files) > 10 else ''}." if files else ""
    return [line, "", f"**PROVENANCE WARNING: the pre-registration and/or the code were NOT committed when this ran "
                      f"(git: {status}).{listed} The commit history cannot show that the design preceded these "
                      "results; treat them as provisional.**"]


def _pv(p: dict, name: str) -> Any:
    return (p.get(name) or {}).get("value", NA)


def _wf_section(r: dict) -> list[str]:
    wf = r["walkforward"]
    rows = [[day(c.get("test_start")), day(c.get("test_end")), c.get("chosen") or "none (no config with enough "
             "fit events)", count(c.get("fit_n")), num(c.get("fit_score")), count(c.get("test_n"))]
            for c in wf.get("choices", [])]
    counts = ", ".join(f"{k}: {v}" for k, v in (wf.get("chosen_counts") or {}).items()) or NA
    p = r.get("params", {})
    return [
        "## OUT-OF-SAMPLE result (walk-forward)", "",
        f"The only result in which anything was chosen. On each {_pv(p, 'WF_FIT_YEARS')}-year fit window the "
        f"threshold with the best t-score among {_pv(p, 'EVENT_Z_GRID')} was picked, using only events whose "
        f"{r['meta'].get('primary_horizon')}-day exit fell before the test window (purged), then applied unchanged "
        f"to the next {_pv(p, 'WF_TEST_MONTHS')} months. The test windows are concatenated.", "",
        f"- **OUT-OF-SAMPLE net abnormal return:** {stat_line(wf['oos'])}",
        f"- Test period {day(wf.get('oos_start'))} to {day(wf.get('oos_end'))}, {count(wf.get('n_splits'))} splits; "
        f"thresholds chosen: {counts}.", "",
        "<details><summary>Walk-forward choices per test window</summary>", "",
        table(["Test start", "Test end", "Chosen", "Fit n", "Fit t-score", "Test n"], rows), "", "</details>", ""]


def _full_section(r: dict) -> list[str]:
    pr = r["primary"]
    h = r["meta"].get("primary_horizon")
    return [
        "## Full-window result: pre-registered single test (no parameters fit)", "",
        "Every event in the development window at the pre-registered threshold and horizon; nothing was chosen. "
        f"Only the {h}-day net abnormal return decides; other horizons and raw returns are secondary.", "",
        f"- **Net abnormal {h}d return:** {stat_line(pr)}",
        f"- Before costs (abnormal {h}d): {stat_line(pr.get('gross', {}))}", ""]


def _horizon_rows(r: dict) -> list[list[str]]:
    by = {(row["horizon"], row["kind"]): row for row in r["horizons"]}
    return [[f"{x}d", mean_ci(by.get((x, "raw"), {})), mean_ci(by.get((x, "abn"), {})),
             mean_ci(by.get((x, "net_abn"), {}))] for x in sorted({row["horizon"] for row in r["horizons"]})]


def _placebo_section(r: dict) -> list[str]:
    pl = r["placebo"]
    ex = pl["excess"]
    h = r["meta"].get("primary_horizon")
    return [
        "## Placebo", "",
        "Same tickers, one random draw per event from member days with finite model inputs and the same price "
        "filter, excluding each real event day and the days inside its own forward window (past events only, so the "
        "placebo never conditions on its own future), run through the identical return code.", "",
        f"- **Placebo abnormal {h}d return (before the fixed cost):** {stat_line(pl['gross'])}",
        f"- Placebo net of costs: {stat_line(pl['net'])}",
        f"- **Events minus placebo (criterion 1b):** {pct(ex.get('diff'))}, {LEVEL} CI "
        f"{ci(ex.get('lo'), ex.get('hi'))}; "
        f"events n={count(ex.get('n_high'))} (effective months {num(ex.get('n_eff_high'), 0)}), placebo "
        f"n={count(ex.get('n_low'))} (effective months {num(ex.get('n_eff_low'), 0)}).",
        f"- Diagnostic only (not a criterion): placebo minus {h} x each ticker's mean daily abnormal return over the "
        f"window: {stat_line(pl.get('drift_adjusted') or {})}. Near zero here while the raw placebo is not points to "
        "drift shared by every day of these tickers (e.g. survivors), not to lookahead.",
        "",
        "A placebo mean away from zero means a pipeline bug or a drift shared by every day (e.g. survivors); the "
        "pre-registration treats the run as invalid until the cause is found. The diagnostic above does not restore "
        "validity; only an amendment written before any result could.", ""]


def _calendar_section(r: dict) -> list[str]:
    c = r["calendar"]
    nw = c.get("newey_west") or {}
    warn = (" **Above the plausibility limit: treat as a bug hunt, not a result.**" if c.get("suspicious") else "")
    return [
        "## Calendar-time portfolio", "",
        "Equal weight across open positions each day, flat (0) on days without one, costs charged on entry and exit "
        "days.", "",
        f"- Mean daily return {bp(nw.get('mean'), 2)} (Newey-West se {bp(nw.get('se'), 2)}, t {num(nw.get('t'))}, "
        f"{count(nw.get('lags'))} lags), n={count(nw.get('n'))} trading days, {count(c.get('active_days'))} with "
        f"positions (mean {num(c.get('mean_positions'), 1)} open).",
        f"- Annualized Sharpe **{num(c.get('sharpe'))}**.{warn}", ""]


def _subperiod_section(r: dict) -> list[str]:
    rows = [[s.get("period"), mean_ci(s), num(s.get("n_eff_clusters"), 0)] for s in r["subperiods"]]
    headers = ["Period", f"Net abnormal mean [{LEVEL} CI] (n)", "Effective months"]
    return ["## Stability by sub-period", "", table(headers, rows), ""]


def _secondary_section(r: dict) -> list[str]:
    sec = r["secondary"]
    h = r["meta"].get("primary_horizon")
    out = ["## Secondary analyses (reported, not decision-driving)", "",
           "### Other horizons and raw returns", "",
           f"Mean [{LEVEL} CI] (n); raw = unhedged stock return, abnormal = factor-hedged. Only the {h}d net abnormal "
           "cell repeats the pre-registered test.", "",
           table(["Horizon", "Raw", "Abnormal", "Net abnormal"], _horizon_rows(r)), ""]
    close_rows = [[f"{s['horizon']}d", mean_ci(s)] for s in sec["close_entry"]]
    out += ["### Entry at the event-day close (how much is untradeable bounce)", "",
            table(["Horizon", "Net abnormal, close entry"], close_rows), "",
            f"Close entry minus next-open entry, per event ({h}d): {stat_line(sec['bounce'])}", ""]
    by = {(s["horizon"], s["kind"]): s for s in sec["up"]}
    up_rows = [[f"{x}d", mean_ci(by.get((x, "abn"), {})), mean_ci(by.get((x, "net_abn"), {}))]
               for x in sorted({s["horizon"] for s in sec["up"]})]
    out += ["### Up-moves (z >= +3), long-side returns", "",
            "An overreaction story predicts a negative mean here (gains that fade).", "",
            table(["Horizon", "Abnormal", "Net abnormal (long)"], up_rows), ""]
    grid_rows = [[f"z <= {s.get('z')}", mean_ci(s)] for s in sec["threshold_grid"]]
    out += ["### Threshold grid: in-sample, descriptive only", "",
            "Shown for context. Picking the best row here would be in-sample selection; the walk-forward section is "
            "the only place a threshold is chosen.", "", table(["Threshold", f"Net abnormal {h}d"], grid_rows), ""]
    f_rows = [[f["name"], f"{f.get('thesis_sign')} ({f.get('thesis')})", count(f.get("n")),
               f"{pct(f.get('diff'))} {ci(f.get('diff_lo'), f.get('diff_hi'))}", num(f.get("p_value"), 3),
               num(f.get("p_adj"), 3), "yes" if f.get("reject") else "no",
               "yes" if f.get("low_confidence") else "no"] for f in sec["factors"]]
    out += ["### Confirming signals, one at a time", "",
            f"Events split into expanding-window terciles (cut points from earlier dates only; binary flags split "
            f"False/True). Difference = top minus bottom group mean net abnormal {h}d return, month-clustered "
            "bootstrap CI, Benjamini-Hochberg across the battery. A low-confidence row is never a discovery.", "",
            table(["Signal", "Thesis predicts", "n", f"Top - bottom [{LEVEL} CI]", "p", "BH p", "Discovery",
                   "Low-conf."], f_rows), ""]
    news = sec.get("news") or {}
    line = f"SEC 8-K news proxy: {news.get('status', NA)}"
    if news.get("status") == "ok":
        missing = news.get("tickers_without_cik") or []
        line += (f"; {pct(news.get('events_covered_share'), 1, False)} of events covered, "
                 f"{pct(news.get('events_with_8k_share'), 1, False)} of covered events had an 8-K effective on the "
                 f"event day or the {news.get('window_sessions')} session(s) before; {len(missing)} of "
                 f"{news.get('n_tickers')} tickers have no CIK in SEC's current map"
                 + (f" ({', '.join(missing[:30])}{', ...' if len(missing) > 30 else ''})" if missing else ""))
    out += [line, ""]
    return out


def _coverage_section(r: dict) -> list[str]:
    cov = r["coverage"]
    s = cov.get("summary") or {}
    out = ["## Data coverage and survivorship", ""]
    if s:
        out += [f"Free price data covers **{num(s['first_pct'], 1)}%** of S&P 500 member-days in {s['first_year']}, "
                f"**{num(s['last_pct'], 1)}%** in {s['last_year']} and {num(s['dev_pct'], 1)}% over the development "
                f"window (worst year {num(s['min_pct'], 1)}%). Up to {count(s['members_with_no_data'])} members a year "
                "have no price at all. Missing names are mostly delisted, acquired or failed companies, so the "
                "sample over-represents survivors; this favours H1, which makes a PASS an upper bound and a FAIL "
                "robust.", ""]
        rows = [[row.get("year"), count(row.get("member_days")), count(row.get("covered_days")),
                 num(row.get("coverage_pct"), 1), count(row.get("members")), count(row.get("members_with_no_data"))]
                for row in cov.get("table", [])]
        out += ["<details><summary>Coverage by year</summary>", "",
                table(["Year", "Member-days", "Covered", "Coverage %", "Members", "Members with no data"], rows),
                "", "</details>", ""]
    else:
        out += [f"Coverage not measured: {cov.get('status')}. The survivorship bias of free price data still "
                "applies to any real-data run.", ""]
    rows = [[f"{m['horizon']}d", count(m["n_events"]), f"{count(m['missing'])} ({pct(m['share'], 1, False)})",
             count(m.get("etf_or_beta_missing")), count(m["embargoed_or_beyond_panel"]),
             pct(m.get("placebo_share"), 1, False)] for m in r["missing_exits"]]
    out += ["Events whose stock price disappears before the exit are excluded, not counted as losses. A higher "
            "share than on placebo days is a survivorship warning sign. Events lost to a sector-ETF or SPY price gap "
            "(or undefined betas) are excluded and counted separately.", "",
            table(["Horizon", "Events", "Missing stock price", "Lost to ETF/beta gap", "Embargoed / beyond data",
                   "Placebo missing stock price"], rows), ""]
    return out + _stale_section(r)


def _stale_section(r: dict) -> list[str]:
    st = r.get("stale_entries") or {}
    if not st:
        return []
    years = ", ".join(f"{y['year']}: {y['stale']}/{y['events']}" for y in st.get("by_year", [])) or "none"
    return ["### Entry opens that were not real trades", "",
            f"Rule: {st.get('rule')}. Such an open is a free-data artifact (often the prior close copied forward), so "
            "using it would quietly turn the next-open entry into a close entry. These entries are treated as "
            "missing for events and placebo alike.", "",
            f"- Primary events affected: {count(st.get('n_stale'))} of {count(st.get('n_events'))} "
            f"({pct(st.get('share'), 1, False)}); placebo days: {count(st.get('placebo_n_stale'))} "
            f"({pct(st.get('placebo_share'), 1, False)}). By year (stale/events): {years}.",
            f"- Sensitivity, stale opens used as fills (literal reading, not decision-driving): "
            f"{stat_line(st.get('including_stale') or {})}", ""]


def _fluke_section(r: dict) -> list[str]:
    m, pr, wf = r["meta"], r["primary"], r["walkforward"]["oos"]
    h = m.get("primary_horizon")
    cost = m.get("cost_bps_per_side")
    gross = (pr.get("gross") or {}).get("mean")
    conc = pr.get("concentration") or {}
    sec = r["secondary"]
    cov = (r.get("coverage") or {}).get("summary") or {}
    miss = next((x for x in r["missing_exits"] if x["horizon"] == h), {})
    n_stats = (len(r["horizons"]) + len(r["subperiods"]) + len(sec["threshold_grid"]) + len(sec["close_entry"])
               + len(sec["up"]) + len(sec["factors"]) + 6)
    items = []
    if cov:
        items.append(f"**Survivorship.** Coverage is {num(cov['first_pct'], 0)}% of member-days in "
                     f"{cov['first_year']} and {num(cov['dev_pct'], 0)}% over the window. A positive result is a "
                     "fluke if the uncovered member-days hide drops that never recovered; the placebo comparison "
                     "removes drift shared by all survivor days, not selection that is specific to post-drop names.")
    else:
        items.append(f"**Survivorship.** Not measured in this run ({r['coverage'].get('status')}).")
    items.append(f"**Disappearing exits.** {pct(miss.get('share'), 1, False)} of events lose their price before the "
                 f"{h}-day exit versus {pct(miss.get('placebo_share'), 1, False)} of placebo days; if those were "
                 "mostly collapses, the reported mean is too high.")
    busiest = ", ".join(f"{k}: {v}" for k, v in (conc.get("top_months") or {}).items())
    items.append(f"**Few episodes.** The events fall in {count(conc.get('n_months'))} calendar months (effective "
                 f"{num(pr.get('n_eff_clusters'), 0)}); the 5 busiest months hold "
                 f"{pct(conc.get('top_share'), 0, False)} of them ({busiest}). If the mean comes from a handful of "
                 "crash months, it describes those episodes, not a repeatable pattern.")
    if is_num(gross) and gross > 0:
        items.append(f"**Costs.** The gross mean is {pct(gross)}, so the net mean reaches zero at a round-trip cost "
                     f"of {num(gross * 1e4, 0)} bp against the assumed {num(2 * cost if is_num(cost) else None, 0)} bp."
                     " Spreads right after a large drop are wider than usual; a realistic cost above the break-even "
                     "erases the result.")
    elif is_num(gross):
        items.append(f"**Costs.** The mean is already {pct(gross)} before costs, so cost assumptions cannot explain "
                     "a negative result.")
    st = r.get("stale_entries") or {}
    if st:
        with_stale = (st.get("including_stale") or {}).get("mean")
        items.append(f"**Fake opens.** {pct(st.get('share'), 1, False)} of events had a t+1 open that was not a real "
                     f"trade and were dropped; using them as fills gives {pct(with_stale)} instead of "
                     f"{pct(pr.get('mean'))}. A large gap means early-year data quality, not the effect, drives part "
                     "of the result.")
    items.append(f"**Untradeable bounce.** Entering at the event-day close instead of the next open changes the mean "
                 f"by {pct(sec['bounce'].get('mean'))} {ci(sec['bounce'].get('ci_lo'), sec['bounce'].get('ci_hi'))}; "
                 "that part could not have been captured.")
    oos_ci, full_ci = ci(wf.get("ci_lo"), wf.get("ci_hi")), ci(pr.get("ci_lo"), pr.get("ci_hi"))
    items.append(f"**Selection.** Out-of-sample (walk-forward) mean {pct(wf.get('mean'))} {oos_ci} versus the "
                 f"full-window {pct(pr.get('mean'))} {full_ci}. An out-of-sample mean well below the full-window one "
                 "would mean the threshold choice, not the effect, drives the in-sample number.")
    items.append(f"**Chance.** This report computes about {n_stats} summary statistics; with no effect anywhere "
                 f"roughly {num((1 - params.CI_LEVEL) * n_stats, 1)} would exclude zero at {LEVEL} by luck. Only the "
                 "pre-registered criteria decide, and the factor battery is BH-corrected.")
    items.append(f"**Pipeline.** Random member days give {pct(r['placebo']['gross'].get('mean'))} "
                 f"{ci(r['placebo']['gross'].get('ci_lo'), r['placebo']['gross'].get('ci_hi'))} before costs. A "
                 "non-zero placebo would mean lookahead or a shared drift that inflates the event mean too.")
    first = next((c for c in r["criteria"] if c["id"] == "1"), {})
    if not first.get("passed") and is_num(pr.get("se")) and is_num(pr.get("ci_hi")):
        if pr["ci_hi"] > 0:
            items.append(f"**For a FAIL to be a fluke** (a real effect missed): net means up to {pct(pr['ci_hi'])} "
                         f"are still consistent with the data, and an effect below about {pct(2 * pr['se'])} is too "
                         "small for this sample to detect.")
        else:
            tail = pct((1 - params.CI_LEVEL) / 2, 1, False)
            items.append(f"**For a FAIL to be a fluke** (a real effect missed): a positive true mean would have to lie "
                         f"above the whole {LEVEL} CI {ci(pr.get('ci_lo'), pr.get('ci_hi'))}. Chance alone puts it "
                         f"there about {tail} of the time, unless the CI is too narrow (few independent months, or "
                         "dependence the month clusters miss).")
    return ["## What would have to be true for this to be a fluke", ""] + [f"- {x}" for x in items] + [""]


def render_phase1(results: dict) -> str:
    r, m = results, results["meta"]
    n_pass = sum(bool(c["passed"]) for c in r["criteria"])
    verdict = r["verdict"]
    head = ["# Phase 1 report: is 'overreaction' a real, exploitable pattern?", "",
            f"**Verdict: {verdict}** ({n_pass} of {len(r['criteria'])} pre-registered criteria passed)."]
    if verdict != "PASS":
        failed = ", ".join(c["id"] for c in r["criteria"] if not c["passed"])
        head.append(f"Failed: criterion {failed}. Per the pre-registration: stop and report; later phases are not "
                    "built on this premise.")
    if r.get("run_invalid"):
        head.append("**The placebo criterion failed, so this run is INVALID until the cause (a pipeline bug, or a "
                    "drift shared by every day such as survivor drift) is found.**")
    sha = m.get("prereg_sha256")
    head += ["", f"Run {m.get('run_utc')}; data {day(m.get('data_first_date'))} to "
                 f"{day(m.get('data_last_date'))}; {count(m.get('n_tickers'))} tickers, "
                 f"{count(m.get('n_dev_days'))} development trading days; PREREGISTRATION.md sha256 "
                 f"`{sha if sha else 'NOT FOUND'}`.", *_provenance_lines(m), ""]
    p = r.get("params", {})
    n_fit = sum(1 for v in p.values() if v.get("source") == "data-fit")
    tail = ["## Parameters", "",
            f"{len(p)} registered parameters, {n_fit} fit to data. Full list with provenance: docs/PARAMETERS.md.", ""]
    parts = (head + _criteria_section(r) + _tested_section(m, p) + _readings_section(m) + _wf_section(r) + _full_section(r)
             + _placebo_section(r) + _calendar_section(r) + _subperiod_section(r) + _secondary_section(r)
             + _coverage_section(r) + _fluke_section(r) + tail)
    return "\n".join(parts)
