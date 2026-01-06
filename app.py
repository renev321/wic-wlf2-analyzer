import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="WIC_WLF2 Analizador", layout="wide")

# ============================
# Seguridad opcional (password)
# ============================
def auth_gate():
    pwd = st.secrets.get("APP_PASSWORD", "")
    if not pwd:
        return True

    if "authed" not in st.session_state:
        st.session_state.authed = False
    if st.session_state.authed:
        return True

    with st.sidebar:
        st.subheader("🔐 Acceso")
        entered = st.text_input("Contraseña", type="password")
        if st.button("Entrar"):
            if entered == pwd:
                st.session_state.authed = True
                st.success("Acceso concedido")
                st.rerun()
            else:
                st.error("Contraseña incorrecta")
    st.stop()

auth_gate()

# ============================
# Helpers / traducciones
# ============================
DIR_MAP = {1: "Compra (Long)", -1: "Venta (Short)"}

REASON_MAP_ES = {
    "SL": "Stop Loss (SL)",
    "TP": "Take Profit (TP)",
    "BE": "Break-even (BE)",
    "TRAIL": "Trailing Stop",
    "SESSION_END": "Fin de sesión",
    "TIME_STOP": "Time stop / Guardia diaria",
    "MANUAL": "Manual / Forzado",
    "": "Sin etiqueta",
    None: "Sin etiqueta",
}

def reason_to_es(x: str) -> str:
    x = "" if x is None else str(x).strip().upper()
    return REASON_MAP_ES.get(x, x)

def side_from_dir(d):
    try:
        d = int(d)
    except:
        return "Desconocido"
    return DIR_MAP.get(d, "Desconocido")

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

    df["type"] = df.get("type", "").astype(str).str.upper()
    df["ts_parsed"] = pd.to_datetime(df.get("ts"), errors="coerce")
    return df, bad

# ============================
# Emparejamiento robusto
# ============================
def _infer_dir_from_entry_fields(row) -> float:
    """
    Inferencia SOLO si tenemos datos del ENTRY:
    - trigger vs orHigh/orLow
    """
    trig = row.get("trigger", np.nan)
    oh   = row.get("orHigh", np.nan)
    ol   = row.get("orLow", np.nan)

    if pd.isna(trig) or pd.isna(oh) or pd.isna(ol):
        return np.nan

    if trig >= oh:
        return 1.0
    if trig <= ol:
        return -1.0
    return np.nan

def pair_trades(df: pd.DataFrame, time_fallback_minutes: int = 45) -> pd.DataFrame:
    entries_raw = df[df["type"] == "ENTRY"].copy()
    exits_raw   = df[df["type"] == "EXIT"].copy()

    if "dir" in entries_raw.columns:
        entries_raw.rename(columns={"dir": "dir_entry"}, inplace=True)
    if "dir" in exits_raw.columns:
        exits_raw.rename(columns={"dir": "dir_exit"}, inplace=True)

    # numéricos si existen
    for c in ["dir_entry", "dir_exit", "trigger", "orHigh", "orLow", "orSize", "atr", "ewo", "deltaRatio",
              "tradeRealized", "maxUnreal", "minUnreal"]:
        if c in entries_raw.columns:
            entries_raw[c] = pd.to_numeric(entries_raw[c], errors="coerce")
        if c in exits_raw.columns:
            exits_raw[c] = pd.to_numeric(exits_raw[c], errors="coerce")

    # ===== 1) Pair por atmId =====
    entry_cols = [
        "atmId","ts_parsed","dir_entry","template","orderType","trigger",
        "orHigh","orLow","orSize","ewo","atr","useAtrEngine","atrSlMult",
        "tp1R","tp2R","tsBehindTP1Atr","trailStepTicks","deltaRatio","dailyPnL"
    ]
    entry_cols = [c for c in entry_cols if c in entries_raw.columns]
    e_by_id = (entries_raw.sort_values("ts_parsed")
                      .groupby("atmId", as_index=False)[entry_cols]
                      .first()
                      .rename(columns={"ts_parsed":"entry_time"}))

    exit_cols = [
        "atmId","ts_parsed","outcome","exitReason","tradeRealized","dayRealized",
        "maxUnreal","minUnreal","forcedCloseReason","dailyHalt","dir_exit"
    ]
    exit_cols = [c for c in exit_cols if c in exits_raw.columns]
    x_by_id = (exits_raw.sort_values("ts_parsed")
                      .groupby("atmId", as_index=False)[exit_cols]
                      .last()
                      .rename(columns={"ts_parsed":"exit_time"}))

    t = x_by_id.merge(e_by_id, on="atmId", how="left")
    t["paired_by_time"] = False
    t["paired_strong"] = t["entry_time"].notna()  # por atmId

    # ===== 2) Fallback por tiempo =====
    missing_mask = t["entry_time"].isna()
    if missing_mask.any() and not entries_raw.empty:
        window = pd.Timedelta(minutes=time_fallback_minutes)

        e2 = entries_raw.copy()
        e2 = e2[e2["ts_parsed"].notna()].sort_values("ts_parsed").reset_index(drop=True)
        e2.rename(columns={"ts_parsed":"entry_time_raw"}, inplace=True)

        used_entry_idx = set()

        unresolved = t[missing_mask & t["exit_time"].notna()].copy()
        unresolved = unresolved.sort_values("exit_time").reset_index()

        ptr = 0
        candidates = []

        for _, row in unresolved.iterrows():
            exit_time = row["exit_time"]

            while ptr < len(e2) and e2.loc[ptr, "entry_time_raw"] <= exit_time:
                candidates.append(ptr)
                ptr += 1

            best = None
            for idx in reversed(candidates):
                if idx in used_entry_idx:
                    continue
                et = e2.loc[idx, "entry_time_raw"]
                if exit_time - et <= window:
                    best = idx
                    break
                else:
                    break

            if best is None:
                continue

            used_entry_idx.add(best)
            ridx = row["index"]

            for col in [
                "dir_entry","template","orderType","trigger",
                "orHigh","orLow","orSize","ewo","atr","useAtrEngine","atrSlMult",
                "tp1R","tp2R","tsBehindTP1Atr","trailStepTicks","deltaRatio","dailyPnL"
            ]:
                if col in e2.columns:
                    t.loc[ridx, col] = e2.loc[best, col]

            t.loc[ridx, "entry_time"] = e2.loc[best, "entry_time_raw"]
            t.loc[ridx, "paired_by_time"] = True
            t.loc[ridx, "paired_strong"] = False

    # ===== 3) dir final =====
    if "dir_entry" in t.columns and "dir_exit" in t.columns:
        t["dir"] = pd.to_numeric(t["dir_entry"], errors="coerce").combine_first(
            pd.to_numeric(t["dir_exit"], errors="coerce")
        )
    elif "dir_entry" in t.columns:
        t["dir"] = pd.to_numeric(t["dir_entry"], errors="coerce")
    elif "dir_exit" in t.columns:
        t["dir"] = pd.to_numeric(t["dir_exit"], errors="coerce")
    else:
        t["dir"] = np.nan

    # ===== 4) inferencia por trigger/orHigh/orLow si aún falta dir =====
    still = t["dir"].isna()
    if still.any():
        inferred = t[still].apply(_infer_dir_from_entry_fields, axis=1)
        t.loc[still, "dir"] = inferred

    # numéricos safe
    for c in ["tradeRealized","dayRealized","maxUnreal","minUnreal","orSize","atr","ewo","deltaRatio"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    # Outcome robusto
    default_outcome = pd.Series(
        np.where(t.get("tradeRealized", 0).fillna(0) >= 0, "WIN", "LOSS"),
        index=t.index
    )
    if "outcome" not in t.columns:
        t["outcome"] = default_outcome
    else:
        t["outcome"] = t["outcome"].astype("string")
        missing = t["outcome"].isna()
        t.loc[missing, "outcome"] = default_outcome.loc[missing]

    t["exitReason"] = t.get("exitReason", "").fillna("").astype(str)
    t["exitReason_ES"] = t["exitReason"].map(reason_to_es)
    t["forcedCloseReason"] = t.get("forcedCloseReason", "").fillna("").astype(str)

    t["has_entry"] = t["entry_time"].notna()
    t["duration_sec"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds()
    t["dur_min"] = t["duration_sec"] / 60.0

    t = t.sort_values("exit_time").reset_index(drop=True)
    t["equity"] = t.get("tradeRealized", 0).fillna(0).cumsum()
    t["equity_peak"] = t["equity"].cummax()
    t["drawdown"] = t["equity"] - t["equity_peak"]

    t["exit_date"] = pd.to_datetime(t["exit_time"]).dt.date
    t["exit_hour"] = pd.to_datetime(t["exit_time"]).dt.hour

    t["lado"] = t["dir"].apply(side_from_dir)

    return t

# ============================
# Métricas
# ============================
def profit_factor(t: pd.DataFrame) -> float:
    wins = t.loc[t["tradeRealized"] > 0, "tradeRealized"].sum()
    losses = t.loc[t["tradeRealized"] < 0, "tradeRealized"].sum()
    if losses == 0:
        return np.nan
    return float(wins / abs(losses))

def streaks_from_pnl(pnls: pd.Series, reset_each_day: bool, dates: pd.Series | None):
    max_w = 0
    max_l = 0
    cur_w = 0
    cur_l = 0

    prev_date = None
    for i, v in enumerate(pnls.fillna(0).tolist()):
        if reset_each_day and dates is not None:
            d = dates.iloc[i]
            if prev_date is None:
                prev_date = d
            elif d != prev_date:
                # reset al cambiar el día
                cur_w = 0
                cur_l = 0
                prev_date = d

        if v > 0:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        elif v < 0:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)
        else:
            cur_w = 0
            cur_l = 0

    return max_w, max_l

def summary_block(t: pd.DataFrame, streak_reset_each_day: bool) -> dict:
    n = len(t)
    win_rate = float((t["tradeRealized"] > 0).mean()) if n else np.nan
    pf = profit_factor(t)
    total = float(t["tradeRealized"].sum()) if n else 0.0
    max_dd = float(t["drawdown"].min()) if n else 0.0
    exp = float(t["tradeRealized"].mean()) if n else np.nan

    max_win = float(t["tradeRealized"].max()) if n else 0.0
    max_loss = float(t["tradeRealized"].min()) if n else 0.0

    avg_win = float(t.loc[t["tradeRealized"] > 0, "tradeRealized"].mean()) if (t["tradeRealized"] > 0).any() else np.nan
    avg_loss = float(t.loc[t["tradeRealized"] < 0, "tradeRealized"].mean()) if (t["tradeRealized"] < 0).any() else np.nan

    best_streak, worst_streak = streaks_from_pnl(
        t["tradeRealized"], reset_each_day=streak_reset_each_day, dates=t["exit_date"]
    )

    buys  = int((t["dir"] == 1).sum())
    sells = int((t["dir"] == -1).sum())
    unknown = int(t["dir"].isna().sum())

    wins = int((t["tradeRealized"] > 0).sum())
    losses = int((t["tradeRealized"] < 0).sum())

    paired_by_time = int(t.get("paired_by_time", False).sum()) if "paired_by_time" in t.columns else 0
    strong = int(t.get("paired_strong", False).sum()) if "paired_strong" in t.columns else 0

    return {
        "trades": n,
        "buys": buys,
        "sells": sells,
        "unknown_dir": unknown,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": pf,
        "total_pnl": total,
        "max_drawdown": max_dd,
        "expectancy": exp,
        "max_win": max_win,
        "max_loss": max_loss,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_streak": best_streak,
        "worst_streak": worst_streak,
        "paired_by_time": paired_by_time,
        "paired_strong": strong,
    }

def interpretacion_rapida(s: dict):
    msgs = []
    warns = []

    if s["trades"] < 20:
        warns.append("Pocas operaciones en el filtro actual (<20). Las conclusiones pueden ser poco estables.")

    if not np.isnan(s["profit_factor"]):
        if s["profit_factor"] < 1.0:
            warns.append(f"Profit Factor {s['profit_factor']:.2f} → este conjunto de reglas pierde dinero en promedio.")
        elif s["profit_factor"] < 1.2:
            warns.append(f"Profit Factor {s['profit_factor']:.2f} → ventaja débil (falta filtro / gestión / horario).")
        else:
            msgs.append(f"Profit Factor {s['profit_factor']:.2f} → parece haber ventaja estadística en este filtro.")

    if s["max_drawdown"] < 0:
        dd = abs(s["max_drawdown"])
        pnl = abs(s["total_pnl"])
        if pnl > 0 and (dd / pnl) > 0.7:
            warns.append("Drawdown alto vs PnL total → estabilidad mejorable (filtros, daily stop, menos trades, evitar horas malas).")

    if s["worst_streak"] >= 5:
        warns.append(f"Racha perdedora máxima = {s['worst_streak']} → necesitas plan anti-rachas (daily stop, bajar tamaño, filtros).")

    return msgs, warns

def side_breakdown(t: pd.DataFrame, streak_reset_each_day: bool) -> pd.DataFrame:
    out = (t.groupby("lado", dropna=False)
             .agg(
                 trades=("atmId", "count"),
                 wins=("tradeRealized", lambda x: int((x > 0).sum())),
                 losses=("tradeRealized", lambda x: int((x < 0).sum())),
                 win_rate=("tradeRealized", lambda x: float((x > 0).mean()) if len(x) else np.nan),
                 total_pnl=("tradeRealized", "sum"),
                 avg_pnl=("tradeRealized", "mean"),
                 max_win=("tradeRealized", "max"),
                 max_loss=("tradeRealized", "min"),
             )
             .reset_index())

    out["profit_factor"] = [profit_factor(t[t["lado"] == lado]) for lado in out["lado"].tolist()]

    streak_w, streak_l = [], []
    for lado in out["lado"].tolist():
        g = t[t["lado"] == lado].sort_values("exit_time").reset_index(drop=True)
        bw, wl = streaks_from_pnl(g["tradeRealized"], reset_each_day=streak_reset_each_day, dates=g["exit_date"])
        streak_w.append(bw)
        streak_l.append(wl)
    out["racha_ganadora_max"] = streak_w
    out["racha_perdedora_max"] = streak_l

    return out.sort_values("trades", ascending=False)

# ============================
# UI
# ============================
st.title("📊 WIC_WLF2 Analizador")
st.caption("Sube uno o varios archivos WIC_WLF2_YYYY-MM.jsonl → métricas claras, tablas legibles, gráficos y guía práctica.")

with st.expander("💡 Cómo leer esto", expanded=False):
    st.markdown(
        """
**Curva de capital (Equity)**  
- PnL acumulado trade a trade. Subida estable → mejor.

**Drawdown (Caída desde máximos)**  
- La peor caída desde el último máximo de equity (tu peor “dolor”).

**Profit Factor (PF)**  
- Ganancias / pérdidas.  
- < 1.0 → pierde dinero  
- 1.0–1.2 → débil  
- > 1.2 → empieza a ser interesante (depende del riesgo)

**Rachas**  
- Conteo de wins/losses consecutivos según el orden temporal de los trades.  
- Si activas “reset por día”, la racha se reinicia cada nuevo día de trading.
        """
    )

with st.expander("🔒 Privacidad / sesiones", expanded=False):
    st.markdown(
        """
- Cada persona que abre el link tiene **su propia sesión**.  
- Tus amigos **no ven** tus archivos subidos (a menos que tú se los pases).  
- Este app **no guarda** logs: solo analiza lo que se sube en esa sesión.
        """
    )

uploads = st.file_uploader(
    "📥 Sube archivos .jsonl (puedes subir varios meses)",
    type=["jsonl"],
    accept_multiple_files=True
)

if not uploads:
    st.info("Sube tu(s) JSONL para comenzar.")
    st.stop()

dfs = []
bad_total = 0
for up in uploads:
    df, bad = load_jsonl_bytes(up.getvalue())
    bad_total += bad
    if df.empty:
        st.warning(f"{up.name}: no se pudieron leer filas.")
        continue
    df["source_file"] = up.name
    dfs.append(df)

if not dfs:
    st.error("No se encontraron filas válidas en los archivos.")
    st.stop()

df_all = pd.concat(dfs, ignore_index=True)

with st.sidebar:
    st.subheader("⚙️ Ajustes de emparejamiento")
    fallback_minutes = st.slider("Ventana fallback por tiempo (min)", 5, 180, 45, 5)

    st.subheader("🔥 Rachas")
    streak_reset_each_day = st.checkbox("Resetear racha por día", value=False)
    streak_conf_mode = st.selectbox(
        "Rachas usando…",
        ["Todos los trades (incluye fallback)", "Solo emparejadas por atmId (alta confianza)"],
        index=1
    )

trades = pair_trades(df_all, time_fallback_minutes=fallback_minutes)

# ===== filtros
st.sidebar.subheader("🎛️ Filtros")

date_min = trades["exit_time"].min()
date_max = trades["exit_time"].max()
d1, d2 = st.sidebar.date_input("Rango de fechas (por salida)", value=(date_min.date(), date_max.date()))

mask = (pd.to_datetime(trades["exit_time"]).dt.date >= d1) & (pd.to_datetime(trades["exit_time"]).dt.date <= d2)

lado_opts = sorted(trades["lado"].dropna().unique().tolist())
lado_sel = st.sidebar.multiselect("Lado", lado_opts, default=lado_opts)
mask &= trades["lado"].isin(lado_sel)

reason_opts_es = sorted(trades["exitReason_ES"].fillna("").unique().tolist())
reason_sel = st.sidebar.multiselect("Motivo de salida", reason_opts_es, default=reason_opts_es)
mask &= trades["exitReason_ES"].fillna("").isin(reason_sel)

t = trades[mask].copy().sort_values("exit_time").reset_index(drop=True)

# recompute equity/drawdown on filtered view
t["equity"] = t["tradeRealized"].fillna(0).cumsum()
t["equity_peak"] = t["equity"].cummax()
t["drawdown"] = t["equity"] - t["equity_peak"]

st.caption(f"🧾 Líneas inválidas ignoradas al parsear: {bad_total}")

strong_count = int(t.get("paired_strong", False).sum()) if "paired_strong" in t.columns else 0
fallback_count = int(t.get("paired_by_time", False).sum()) if "paired_by_time" in t.columns else 0
if fallback_count > 0:
    st.info(f"🧩 Emparejadas por fallback de tiempo: {fallback_count}. (Útil, pero puede afectar rachas/orden).")
st.caption(f"✅ Emparejadas por atmId (alta confianza): {strong_count}")

if streak_conf_mode.startswith("Solo emparejadas"):
    t_streak = t[t["paired_strong"] == True].copy().sort_values("exit_time").reset_index(drop=True)
else:
    t_streak = t

s = summary_block(t_streak, streak_reset_each_day=streak_reset_each_day)

unknown_dir = int(t["dir"].isna().sum())
if unknown_dir > 0:
    st.warning(
        f"⚠️ {unknown_dir} operaciones siguen sin dirección (Compra/Venta). "
        f"WIN/LOSS se calcula por PnL (tradeRealized), no por lado."
    )

# ============================
# Resumen rápido
# ============================
st.subheader("✅ Lo más importante (resumen rápido)")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Operaciones", f"{len(t)}")
c2.metric("Compras (Long)", f"{int((t['dir']==1).sum())}")
c3.metric("Ventas (Short)", f"{int((t['dir']==-1).sum())}")
c4.metric("Ganadas", f"{int((t['tradeRealized']>0).sum())}")
c5.metric("Perdidas", f"{int((t['tradeRealized']<0).sum())}")
wr = float((t["tradeRealized"] > 0).mean()) if len(t) else np.nan
c6.metric("% Acierto", f"{(wr*100):.1f}%" if not np.isnan(wr) else "n/a")

pf_all = profit_factor(t)
exp_all = float(t["tradeRealized"].mean()) if len(t) else np.nan
max_dd_all = float(t["drawdown"].min()) if len(t) else 0.0

c7, c8, c9, c10, c11, c12 = st.columns(6)
c7.metric("PnL Total", f"{float(t['tradeRealized'].sum()):.0f}")
c8.metric("Profit Factor", f"{pf_all:.2f}" if not np.isnan(pf_all) else "n/a")
c9.metric("Expectancia", f"{exp_all:.1f}" if not np.isnan(exp_all) else "n/a")
c10.metric("Max Drawdown", f"{max_dd_all:.0f}")

# rachas calculadas según el modo elegido
label_racha = "Racha (según modo elegido)"
c11.metric("Mejor racha (wins)", f"{s['best_streak']}")
c12.metric("Peor racha (losses)", f"{s['worst_streak']}")
st.caption(f"🔥 Rachas calculadas con: **{streak_conf_mode}** | reset por día: **{streak_reset_each_day}**")

c13, c14, c15, c16 = st.columns(4)
c13.metric("Mayor win", f"{float(t['tradeRealized'].max()):.0f}" if len(t) else "0")
c14.metric("Mayor loss", f"{float(t['tradeRealized'].min()):.0f}" if len(t) else "0")
avg_win = float(t.loc[t["tradeRealized"] > 0, "tradeRealized"].mean()) if (t["tradeRealized"] > 0).any() else np.nan
avg_loss = float(t.loc[t["tradeRealized"] < 0, "tradeRealized"].mean()) if (t["tradeRealized"] < 0).any() else np.nan
c15.metric("Win promedio", f"{avg_win:.1f}" if not np.isnan(avg_win) else "n/a")
c16.metric("Loss promedio", f"{avg_loss:.1f}" if not np.isnan(avg_loss) else "n/a")

msgs, warns = interpretacion_rapida({
    "trades": len(t),
    "profit_factor": pf_all,
    "max_drawdown": max_dd_all,
    "total_pnl": float(t["tradeRealized"].sum()),
    "worst_streak": s["worst_streak"],
})
for m in msgs:
    st.success(m)
for w in warns:
    st.warning(w)

# ============================
# MFE / MAE global
# ============================
st.subheader("📌 Extremos intratrade (MFE / MAE)")

if "maxUnreal" in t.columns and t["maxUnreal"].notna().any():
    idx = t["maxUnreal"].idxmax()
    row = t.loc[idx]
    st.metric("Mejor MFE (maxUnreal)", f"{float(row['maxUnreal']):.0f}")
    st.caption(f"TradeId={row['atmId']} | Salida={row['exit_time']} | Lado={row['lado']}")
else:
    st.info("No hay maxUnreal en los logs.")

if "minUnreal" in t.columns and t["minUnreal"].notna().any():
    idx = t["minUnreal"].idxmin()
    row = t.loc[idx]
    st.metric("Peor MAE (minUnreal)", f"{float(row['minUnreal']):.0f}")
    st.caption(f"TradeId={row['atmId']} | Salida={row['exit_time']} | Lado={row['lado']}")
else:
    st.info("No hay minUnreal en los logs.")

# ============================
# Por lado
# ============================
st.subheader("🧭 Rendimiento por lado (Compra vs Venta)")
sb = side_breakdown(t, streak_reset_each_day=streak_reset_each_day)
sb2 = sb.copy()
sb2["win_rate"] = (sb2["win_rate"] * 100).round(1)
sb2.rename(columns={
    "lado":"Lado",
    "trades":"Trades",
    "wins":"Wins",
    "losses":"Losses",
    "win_rate":"WinRate(%)",
    "total_pnl":"PnL Total",
    "avg_pnl":"PnL Prom",
    "profit_factor":"PF",
    "max_win":"Max Win",
    "max_loss":"Max Loss",
    "racha_ganadora_max":"Racha Win Max",
    "racha_perdedora_max":"Racha Loss Max",
}, inplace=True)
st.dataframe(sb2, use_container_width=True, height=260)

# ============================
# Motivos de salida
# ============================
st.subheader("🚪 Motivos de salida (impacto)")
by_exit = (t.groupby("exitReason_ES", dropna=False)
             .agg(
                 trades=("atmId","count"),
                 win_rate=("tradeRealized", lambda x: float((x>0).mean()) if len(x) else np.nan),
                 avg_pnl=("tradeRealized","mean"),
                 total_pnl=("tradeRealized","sum"),
             )
             .reset_index()
             .sort_values("trades", ascending=False))
by_exit["win_rate"] = (by_exit["win_rate"] * 100).round(1)
by_exit.rename(columns={
    "exitReason_ES":"Motivo",
    "trades":"Trades",
    "win_rate":"WinRate(%)",
    "avg_pnl":"PnL Prom",
    "total_pnl":"PnL Total",
}, inplace=True)
st.dataframe(by_exit, use_container_width=True, height=300)

# ============================
# Gráficos
# ============================
st.subheader("📈 Gráficos")

g1, g2 = st.columns(2)
with g1:
    st.markdown("### Curva de capital (Equity)")
    st.caption("PnL acumulado operación por operación.")
    st.line_chart(t.set_index("exit_time")["equity"])

with g2:
    st.markdown("### Drawdown (caída desde máximos)")
    st.caption("Caída desde el último máximo de equity. Más bajo = peor.")
    st.line_chart(t.set_index("exit_time")["drawdown"])

# ===== Distribución + percentiles
st.markdown("### Distribución de PnL por trade + percentiles")
if len(t) > 0:
    pnls = t["tradeRealized"].dropna()
    if len(pnls) > 0:
        p10, p25, p50, p75, p90 = np.percentile(pnls, [10,25,50,75,90])

        hist = alt.Chart(t.dropna(subset=["tradeRealized"])).mark_bar().encode(
            x=alt.X("tradeRealized:Q", bin=alt.Bin(maxbins=60), title="PnL por trade"),
            y=alt.Y("count():Q", title="Cantidad")
        ).properties(height=280)

        lines = pd.DataFrame({
            "x": [p10,p25,p50,p75,p90],
            "label": ["P10","P25","P50","P75","P90"]
        })
        vlines = alt.Chart(lines).mark_rule().encode(
            x="x:Q",
            tooltip=["label","x"]
        )
        st.altair_chart(hist + vlines, use_container_width=True)
        st.caption(f"Percentiles: P10={p10:.0f} | P25={p25:.0f} | P50={p50:.0f} | P75={p75:.0f} | P90={p90:.0f}")

# ===== CDF / curva percentil
st.markdown("### Curva de percentiles (CDF)")
if len(t) > 0 and t["tradeRealized"].notna().any():
    pnls = np.sort(t["tradeRealized"].dropna().values)
    y = np.linspace(0, 100, len(pnls))
    df_cdf = pd.DataFrame({"PnL": pnls, "Percentil": y})
    cdf = alt.Chart(df_cdf).mark_line().encode(
        x=alt.X("PnL:Q", title="PnL"),
        y=alt.Y("Percentil:Q", title="Percentil (%)")
    ).properties(height=280)
    st.altair_chart(cdf, use_container_width=True)

# ===== Hora del día
st.markdown("### Rendimiento por hora")
by_hour = (t.groupby("exit_hour")
            .agg(
                trades=("atmId","count"),
                win_rate=("tradeRealized", lambda x: float((x>0).mean()) if len(x) else np.nan),
                avg_pnl=("tradeRealized","mean"),
                total_pnl=("tradeRealized","sum"),
            )
            .reset_index()
            .sort_values("exit_hour"))
by_hour["win_rate_pct"] = (by_hour["win_rate"] * 100).round(1)

cA, cB = st.columns(2)
with cA:
    ch = alt.Chart(by_hour).mark_bar().encode(
        x=alt.X("exit_hour:O", title="Hora"),
        y=alt.Y("win_rate_pct:Q", title="WinRate (%)"),
        tooltip=["exit_hour","trades","win_rate_pct","avg_pnl","total_pnl"]
    ).properties(height=260)
    st.altair_chart(ch, use_container_width=True)
with cB:
    ch2 = alt.Chart(by_hour).mark_bar().encode(
        x=alt.X("exit_hour:O", title="Hora"),
        y=alt.Y("avg_pnl:Q", title="PnL promedio"),
        tooltip=["exit_hour","trades","win_rate_pct","avg_pnl","total_pnl"]
    ).properties(height=260)
    st.altair_chart(ch2, use_container_width=True)

# ===== Scatters útiles para tuning
st.markdown("### Scatters útiles para tuning (PnL vs factores)")
def scatter_if(col, title):
    if col in t.columns and t[col].notna().any():
        dfp = t.dropna(subset=[col, "tradeRealized"])
        chart = alt.Chart(dfp).mark_circle(size=60, opacity=0.45).encode(
            x=alt.X(f"{col}:Q", title=title),
            y=alt.Y("tradeRealized:Q", title="PnL"),
            tooltip=["exit_time","lado","tradeRealized",col]
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)

s1, s2 = st.columns(2)
with s1:
    scatter_if("orSize", "OR Size")
    scatter_if("atr", "ATR")
with s2:
    scatter_if("ewo", "EWO")
    scatter_if("deltaRatio", "DeltaRatio")

# ============================
# Tabla trades
# ============================
st.subheader("📋 Lista de operaciones (vista humana)")

cols = []
for c in [
    "atmId","lado","paired_strong","paired_by_time","entry_time","exit_time","dur_min",
    "tradeRealized","outcome","exitReason_ES","forcedCloseReason",
    "trigger","orHigh","orLow","orSize","atr","ewo","deltaRatio",
    "maxUnreal","minUnreal","source_file"
]:
    if c in t.columns:
        cols.append(c)

t_view = t[cols].copy()
t_view.rename(columns={
    "atmId":"TradeId",
    "lado":"Lado",
    "paired_strong":"Emparejada por atmId",
    "paired_by_time":"Emparejada por tiempo",
    "entry_time":"Entrada",
    "exit_time":"Salida",
    "dur_min":"Duración (min)",
    "tradeRealized":"PnL",
    "outcome":"Resultado",
    "exitReason_ES":"Motivo salida",
    "forcedCloseReason":"Motivo forzado",
    "trigger":"Trigger",
    "orHigh":"OR High",
    "orLow":"OR Low",
    "orSize":"OR Size",
    "atr":"ATR",
    "ewo":"EWO",
    "deltaRatio":"DeltaRatio",
    "maxUnreal":"MFE (maxUnreal)",
    "minUnreal":"MAE (minUnreal)",
    "source_file":"Archivo",
}, inplace=True)

st.dataframe(t_view, use_container_width=True, height=520)

# ============================
# Export
# ============================
st.subheader("⬇️ Exportar")
csv_bytes = t_view.to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar CSV (operaciones filtradas)",
    data=csv_bytes,
    file_name="WIC_WLF2_operaciones_filtradas.csv",
    mime="text/csv",
)
