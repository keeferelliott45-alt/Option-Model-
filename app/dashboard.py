import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS, DATA = ROOT / "reports", ROOT / "data"

st.set_page_config(page_title="Options Scanner", page_icon="📈", layout="wide")


def _scan_days() -> list[str]:
    return sorted((p.stem for p in (REPORTS / "scanner").glob("*.csv")), reverse=True)


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _tail(path: Path, n: int = 8) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log yet)"


days = _scan_days()
st.title("Options Scanner")
st.warning("**Paper trading only.** The pricing engine passed its historical tests, but the picks themselves are "
           "unproven until the forward test on saved option chains. A human decides every trade.")

tab_today, tab_history, tab_models, tab_status = st.tabs(["Today's picks", "History", "Model checks", "Daily job"])

with tab_today:
    if not days:
        st.info("No scan yet. The daily job writes one each weekday before the US open.")
    else:
        day = st.selectbox("Scan date", days, index=0)
        meta = _json(REPORTS / "scanner" / f"{day}.json")
        picks = pd.read_csv(REPORTS / "scanner" / f"{day}.csv")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Picks", len(picks))
        c2.metric("Contracts scored", f"{meta.get('contracts_scored', 0):,}")
        c3.metric("Stocks scored", meta.get("tickers_scored", 0))
        c4.metric("Market regime", (meta.get("regime") or "?").split(",")[0])
        st.caption(f"Regime: {meta.get('regime', '?')}")
        if picks.empty:
            st.info("Nothing cleared the liquidity, earnings, quality and value hurdles on this day.")
        else:
            side = st.multiselect("Direction", ["bullish", "bearish"], default=["bullish", "bearish"])
            bull = {"long_call", "bull_call_debit", "bull_put_credit"}
            picks["direction"] = picks["name"].map(lambda n: "bullish" if n in bull else "bearish")
            view = picks[picks["direction"].isin(side)].copy()
            view["EV % of risk"] = (100 * view["ev_pct_risk"]).round(1)
            view["P(profit) %"] = (100 * view["prob_profit"]).round(1)
            view["model vol %"] = (100 * view["vol_fc"]).round(1)
            cols = {"ticker": "Stock", "name": "Structure", "legs": "Legs", "expiry": "Expiry", "dte": "Days",
                    "ev": "EV $", "max_loss": "Max loss $", "EV % of risk": "EV % of risk",
                    "P(profit) %": "P(profit) %", "bucket": "Delta bucket", "model vol %": "Model vol %",
                    "veto": "Fundamentals", "fundamentals": "Notes"}
            st.dataframe(view[list(cols)].rename(columns=cols).round(2), hide_index=True, use_container_width=True)
            st.caption("EV = expected profit after buying at the ask, selling at the bid and commissions. "
                       "0.05-0.20 delta contracts need a higher EV bar: they depend most on the forecast's tails.")

with tab_history:
    if days:
        frames = [pd.read_csv(REPORTS / "scanner" / f"{d}.csv").assign(scan_date=d) for d in days]
        hist = pd.concat(frames, ignore_index=True)
        st.metric("Total picks logged", len(hist))
        if not hist.empty:
            st.bar_chart(hist.groupby("scan_date").size().rename("picks"))
            st.dataframe(hist[["scan_date", "ticker", "name", "legs", "expiry", "ev", "max_loss", "prob_profit"]],
                         hide_index=True, use_container_width=True)
        st.caption("Outcomes are scored in the forward test once enough chains have been saved.")

with tab_models:
    st.subheader("What passed testing (and drives the engine)")
    s1 = pd.read_csv(REPORTS / "stage1" / "factors.csv") if (REPORTS / "stage1" / "factors.csv").exists() else None
    if s1 is not None:
        passed = s1[s1["passed"]]["factor"].tolist()
        st.write("**Stage 1 factors that passed:** " + (", ".join(passed) or "none"))
        st.write("**Return-predicting factors:** none passed, so the forecast has no directional drift.")
    for name, label in (("b2_results.json", "Calibration on no-earnings windows (B2)"),
                        ("e_results.json", "Tail calibration (E)")):
        r = _json(REPORTS / "stage2" / name)
        if r:
            st.write(f"**{label}:** {'PASS' if r.get('passed') else 'FAIL'}")
    st.info("Known limits: options spanning earnings are never shown (engine not calibrated there); stocks are "
            "skipped for 22 trading days after earnings; the downside tail is slightly underestimated "
            "(realized crashes ~1.2x model), so short puts are a bit riskier than shown.")
    for f in ("stage1", "stage2"):
        p = REPORTS / f / "report.md"
        if p.exists():
            with st.expander(f"Full {f} report"):
                st.markdown(p.read_text(encoding="utf-8"))

with tab_status:
    snaps = sorted((DATA / "options_snapshots").glob("*")) if (DATA / "options_snapshots").exists() else []
    st.metric("Days of option chains saved", len(snaps))
    if snaps:
        st.write(f"Latest saved session: **{snaps[-1].name}** ({len(list(snaps[-1].glob('*.parquet')))} stocks)")
    st.write("Forward test of the picks needs about 6 months of saved chains.")
    st.text_area("Scan log (last lines)", _tail(DATA / "scan_daily.log"), height=160)
    st.text_area("Snapshot log (last lines)", _tail(DATA / "snapshot_daily.log"), height=160)
