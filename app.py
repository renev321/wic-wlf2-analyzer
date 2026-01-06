import json
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="WIC_WLF2 Log Analyzer", layout="wide")

# ----------------------------
# Optional password gate
# ----------------------------
def auth_gate():
    pwd = st.secrets.get("APP_PASSWORD", "")
    if not pwd:
        # Open app (still free). Add APP_PASSWORD in Streamlit Secrets if you want a simple gate.
        return True

    if "authed" not in st.session_state:
        st.session_state.authed = False

    if st.session_state.authed:
        return True

    with st.sidebar:
        st.subheader("Login")
        entered = st.text_input("Password", type="password")
        if st.button("Unlock"):
            if entered == pwd:
                st.session_state.authed = True
                st.success("Unlocked")
                st.rerun()
            else:
                st.error("Wrong password")
    st.stop()

auth_gate()

st.title("WIC_WLF2 JSONL Analyzer")
st.caption("Upload one or more WIC_WLF2_YYYY-MM.jsonl files → readable stats, charts, and advice.")

# ----------------------------
# Parsing
# ----------------------------
def load_jsonl_bytes(data: bytes):
    rows = []
    bad = 0
    for i, line in enumerate(data.decode("utf-8", errors="ignore").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            obj["_line"] = i
            rows.append(obj)
        except Exception:
            bad += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df, bad

    df["ts_parsed"] = pd.to_datetime(df.get("ts"), errors="coerce")
    df["type"] = df.get("type", "").astype(str).str.upper()
    return df, bad

def pair_trades(df: pd.DataFrame) -> pd.DataFrame:
    entries = df[df["type"] == "ENTRY"].copy()
    exits   = df[df["type"] == "EXIT"].copy()

    # first ENTRY per atmId
    entry_cols = [
        "atmId","ts_parsed","dir","template","orderType","trigger",
        "orHigh","orLow","orSize","ewo","atr","useAtrEngine","atrSlMult",
        "tp1R","tp2R","tsBehindTP1Atr","trailStepTicks","deltaRatio","dailyPnL"
    ]
    entry_cols = [c for c in entry_cols if c in entries.columns]
    e1 = (entries.sort_values("ts_parsed")
                 .groupby("atmId", as_index=False)[entry_cols]
                 .first()
                 .rename(columns={"ts_parsed":"entry_time"}))

    # last EXIT per atmId
    exit_cols = [
        "atmId","ts_parsed","outcome","exitReason","tradeRealized","dayRealized",
        "maxUnreal","minUnreal","forcedCloseReason","dailyHalt"
    ]
    exit_cols = [c for c in exit_cols if c in exits.columns]
    xlast = (exits.sort_values("ts_parsed")
                  .groupby("atmId", as_index=False)[exit_cols]
                  .last()
                  .rename(columns={"ts_parsed":"exit_time"}))

    t = xlast.merge(e1, on="atmId", how="left")

    # numeric conversions
    for c in ["tradeRealized","dayRealized","maxUnreal","minUnreal","orSize","atr","ewo","deltaRatio","dir"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    t["has_entry"] = t["entry_time"].notna()
    t["duration_sec"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds()

    # fill outcome if missing (pandas-safe: no ndarray passed to fillna)
    default_outcome = pd.Series(
        np.where(t["tradeRealized"].fillna(0) >= 0, "WIN", "LOSS"),
        index=t.index
    )
    
    if "outcome" not in t.columns:
        t["outcome"] = default_outcome
    else:
        # ensure missing values are real NaNs/<NA>, then fill only where missing
        t["outcome"] = t["outcome"].astype("string")
        missing = t["outcome"].isna()
        t.loc[missing, "outcome"] = default_outcome.loc[missing]


    t["exitReason"] = t.get("exitReason", "").fillna("")
    t["forcedCloseReason"] = t.get("forcedCloseReason", "").fillna("")

    t = t.sort_values("exit_time").reset_index(drop=True)
    t["equity"] = t["tradeRealized"].fillna(0).cumsum()
    t["equity_peak"] = t["equity"].cummax()
    t["drawdown"] = t["equity"] - t["equity_peak"]
    t["exit_date"] = pd.to_datetime(t["exit_time"]).dt.date
    t["exit_hour"] = pd.to_datetime(t["exit_time"]).dt.hour
    return t

def profit_factor(t: pd.DataFrame) -> float:
    wins = t[t["tradeRealized"] > 0]["tradeRealized"].sum()
    losses = t[t["tradeRealized"] < 0]["tradeRealized"].sum()
    if losses == 0:
        return np.nan
    return wins / abs(losses)

def make_summary(t: pd.DataFrame) -> dict:
    n = len(t)
    win_rate = float((t["tradeRealized"] > 0).mean()) if n else np.nan
    pf = profit_factor(t)
    total = float(t["tradeRealized"].sum()) if n else 0.0
    max_dd = float(t["drawdown"].min()) if n else 0.0
    exp = float(t["tradeRealized"].mean()) if n else np.nan
    cov = float(t["has_entry"].mean()*100) if n else 0.0
    return dict(trades=n, win_rate=win_rate, profit_factor=pf, total_pnl=total, max_drawdown=max_dd, expectancy=exp, entry_coverage_pct=cov)

def advice_cards(summary: dict, trades: pd.DataFrame) -> list[str]:
    tips = []

    if summary["entry_coverage_pct"] < 95:
        tips.append(
            f"ENTRY coverage is only {summary['entry_coverage_pct']:.1f}%. "
            "Many EXITs have no matching ENTRY, so OR/ATR/EWO breakdowns are weaker. "
            "Fix: log ENTRY on first fill (OnExecutionUpdate) for every trade."
        )

    if "exitReason" in trades.columns and len(trades) > 20:
        top = trades["exitReason"].value_counts(dropna=False).head(1)
        if len(top) and top.index[0] in ("", "MANUAL") and (top.iloc[0] / len(trades) > 0.6):
            tips.append(
                "Most exits are labeled MANUAL/blank. This usually means the strategy can’t classify SL/TP/TRAIL reliably "
                "(common after removing ATM). Fix: infer TP/SL using managed order events (stop/target fills) and log it."
            )

    if abs(summary["max_drawdown"]) > 0 and abs(summary["total_pnl"]) > 0:
        dd_ratio = abs(summary["max_drawdown"]) / max(1.0, abs(summary["total_pnl"]))
        if dd_ratio > 0.5:
            tips.append(
                f"Max drawdown is large relative to total PnL (DD/PNL ≈ {dd_ratio:.2f}). "
                "Consider tighter daily loss guard, fewer setups/session, or filter out weak OR sizes / low EWO magnitude."
            )

    if not np.isnan(summary["profit_factor"]) and summary["profit_factor"] < 1.2:
        tips.append(
            f"Profit factor is {summary['profit_factor']:.2f}. Consider increasing selectivity "
            "(MinORSize/MaxORSize, MinEWOMagnitude, break candle body filter) or reducing chase entries."
        )

    if not tips:
        tips.append("No major red flags detected. Next: segment by OR size / ATR / hour to find your best regime.")
    return tips

# ----------------------------
# UI: upload
# ----------------------------
uploads = st.file_uploader("Upload .jsonl logs (you can upload multiple months)", type=["jsonl"], accept_multiple_files=True)

if not uploads:
    st.info("Upload your JSONL file(s) to begin.")
    st.stop()

dfs = []
bad_total = 0
for up in uploads:
    df, bad = load_jsonl_bytes(up.getvalue())
    bad_total += bad
    if df.empty:
        st.warning(f"{up.name}: no rows parsed.")
        continue
    df["source_file"] = up.name
    dfs.append(df)

if not dfs:
    st.error("No valid rows found in the uploaded file(s).")
    st.stop()

df_all = pd.concat(dfs, ignore_index=True)
trades = pair_trades(df_all)
summary_f = make_summary(trades)

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.subheader("Filters")

date_min = trades["exit_time"].min()
date_max = trades["exit_time"].max()
if pd.isna(date_min) or pd.isna(date_max):
    st.error("Could not parse timestamps (ts). Ensure 'ts' is like 'YYYY-MM-DD HH:mm:ss.fff'.")
    st.stop()

d1, d2 = st.sidebar.date_input("Exit date range", value=(date_min.date(), date_max.date()))
mask = (pd.to_datetime(trades["exit_time"]).dt.date >= d1) & (pd.to_datetime(trades["exit_time"]).dt.date <= d2)

dir_opts = sorted([x for x in trades["dir"].dropna().unique().tolist()]) if "dir" in trades.columns else []
if dir_opts:
    dir_sel = st.sidebar.multiselect("Direction (dir)", dir_opts, default=dir_opts)
    mask &= trades["dir"].isin(dir_sel)

reason_opts = sorted([str(x) for x in trades["exitReason"].fillna("").unique().tolist()])
reason_sel = st.sidebar.multiselect("Exit reason", reason_opts, default=reason_opts)
mask &= trades["exitReason"].fillna("").isin(reason_sel)

t = trades[mask].copy()
summary_f = make_summary(t)

# ----------------------------
# Overview
# ----------------------------
st.subheader("Overview")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Trades", f"{summary_f['trades']}")
c2.metric("Win rate", f"{summary_f['win_rate']*100:.1f}%" if not np.isnan(summary_f['win_rate']) else "n/a")
c3.metric("Profit factor", f"{summary_f['profit_factor']:.2f}" if not np.isnan(summary_f['profit_factor']) else "n/a")
c4.metric("Total PnL", f"{summary_f['total_pnl']:.0f}")
c5.metric("Max drawdown", f"{summary_f['max_drawdown']:.0f}")
c6.metric("Entry coverage", f"{summary_f['entry_coverage_pct']:.1f}%")

st.caption(f"Bad lines skipped during parsing: {bad_total}")

left, right = st.columns(2)
with left:
    st.markdown("**Equity curve**")
    st.line_chart(t.set_index("exit_time")["equity"])
with right:
    st.markdown("**Drawdown**")
    st.line_chart(t.set_index("exit_time")["drawdown"])

st.markdown("**PnL distribution (rough)**")
st.bar_chart(t["tradeRealized"].round(0).value_counts().sort_index().head(200))

# ----------------------------
# Breakdowns
# ----------------------------
st.subheader("Breakdowns")

colA, colB = st.columns(2)
with colA:
    st.markdown("**By exit reason**")
    by_exit = (t.groupby("exitReason", dropna=False)
                 .agg(trades=("atmId","count"),
                      win_rate=("tradeRealized", lambda x: (x>0).mean()),
                      avg_pnl=("tradeRealized","mean"),
                      total_pnl=("tradeRealized","sum"))
                 .sort_values("trades", ascending=False)
                 .reset_index())
    st.dataframe(by_exit, use_container_width=True, height=320)

with colB:
    st.markdown("**By hour (exit)**")
    by_hour = (t.groupby("exit_hour")
                 .agg(trades=("atmId","count"),
                      win_rate=("tradeRealized", lambda x: (x>0).mean()),
                      avg_pnl=("tradeRealized","mean"),
                      total_pnl=("tradeRealized","sum"))
                 .reset_index()
                 .sort_values("exit_hour"))
    st.dataframe(by_hour, use_container_width=True, height=320)

# OR/ATR quartiles (only where ENTRY exists)
t_entry = t[t["has_entry"] & t["orSize"].notna()].copy()
colC, colD = st.columns(2)

with colC:
    st.markdown("**OR size quartiles (ENTRY only)**")
    if len(t_entry) >= 8:
        t_entry["or_bin"] = pd.qcut(t_entry["orSize"], q=4, duplicates="drop")
        by_or = (t_entry.groupby("or_bin")
                    .agg(trades=("atmId","count"),
                         win_rate=("tradeRealized", lambda x: (x>0).mean()),
                         avg_pnl=("tradeRealized","mean"),
                         total_pnl=("tradeRealized","sum"),
                         or_min=("orSize","min"),
                         or_med=("orSize","median"),
                         or_max=("orSize","max"))
                    .reset_index())
        by_or["or_bin"] = by_or["or_bin"].astype(str)
        st.dataframe(by_or.sort_values("or_min"), use_container_width=True, height=320)
    else:
        st.info("Not enough ENTRY-linked trades to compute OR quartiles.")

with colD:
    st.markdown("**ATR quartiles (ENTRY only)**")
    if "atr" in t_entry.columns and t_entry["atr"].notna().sum() >= 8:
        te = t_entry[t_entry["atr"].notna()].copy()
        te["atr_bin"] = pd.qcut(te["atr"], q=4, duplicates="drop")
        by_atr = (te.groupby("atr_bin")
                    .agg(trades=("atmId","count"),
                         win_rate=("tradeRealized", lambda x: (x>0).mean()),
                         avg_pnl=("tradeRealized","mean"),
                         total_pnl=("tradeRealized","sum"),
                         atr_min=("atr","min"),
                         atr_med=("atr","median"),
                         atr_max=("atr","max"))
                    .reset_index())
        by_atr["atr_bin"] = by_atr["atr_bin"].astype(str)
        st.dataframe(by_atr.sort_values("atr_min"), use_container_width=True, height=320)
    else:
        st.info("No ATR data linked to ENTRY rows.")

# ----------------------------
# Trades table
# ----------------------------
st.subheader("Trades (filtered)")
cols = [c for c in [
    "atmId","entry_time","exit_time","duration_sec","dir","orSize","atr","ewo","deltaRatio",
    "tradeRealized","maxUnreal","minUnreal","outcome","exitReason","forcedCloseReason","source_file"
] if c in t.columns]
st.dataframe(t[cols], use_container_width=True, height=420)

# ----------------------------
# Advice
# ----------------------------
st.subheader("Advice")
for tip in advice_cards(summary_f, t):
    st.warning(tip)

# ----------------------------
# Download
# ----------------------------
st.subheader("Download")
csv_bytes = t.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered trades CSV",
    data=csv_bytes,
    file_name="WIC_WLF2_trades_filtered.csv",
    mime="text/csv",
)
