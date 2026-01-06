import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="WIC_WLF2 Analizador JSONL", layout="wide")

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
# Helpers (ES)
# ============================
DIR_MAP = {1: "Compra (Long)", -1: "Venta (Short)"}

REASON_MAP_ES = {
    "SL": "Stop Loss (SL)",
    "TP": "Take Profit (TP)",
    "BE": "Break-even (BE)",
    "TRAIL": "Trailing Stop",
    "SESSION_END": "Fin de sesión",
    "TIME_STOP": "Time stop / guardia diaria",
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
    return DIR_MAP.get(d, f"Dir={d}")

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

    for c in ["tradeRealized","dayRealized","maxUnreal","minUnreal","orSize","atr","ewo","deltaRatio","dir"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    t["has_entry"] = t["entry_time"].notna()
    t["duration_sec"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds()

    # Outcome robusto (sin fillna(array) para pandas nuevos)
    default_outcome = pd.Series(
        np.where(t["tradeRealized"].fillna(0) >= 0, "WIN", "LOSS"),
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

    t = t.sort_values("exit_time").reset_index(drop=True)
    t["equity"] = t["tradeRealized"].fillna(0).cumsum()
    t["equity_peak"] = t["equity"].cummax()
    t["drawdown"] = t["equity"] - t["equity_peak"]
    t["exit_date"] = pd.to_datetime(t["exit_time"]).dt.date
    t["exit_hour"] = pd.to_datetime(t["exit_time"]).dt.hour

    t["lado"] = t["dir"].apply(side_from_dir)

    # duración humana
    t["dur_min"] = t["duration_sec"] / 60.0

    return t

def profit_factor(t: pd.DataFrame) -> float:
    wins = t.loc[t["tradeRealized"] > 0, "tradeRealized"].sum()
    losses = t.loc[t["tradeRealized"] < 0, "tradeRealized"].sum()
    if losses == 0:
        return np.nan
    return float(wins / abs(losses))

def streaks_from_pnl(pnls: pd.Series):
    # rachas en secuencia temporal
    max_w = 0
    max_l = 0
    cur_w = 0
    cur_l = 0
    for v in pnls.fillna(0).tolist():
        if v > 0:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        elif v < 0:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)
        else:
            # neutro rompe racha
            cur_w = 0
            cur_l = 0
    return max_w, max_l

def summary_block(t: pd.DataFrame) -> dict:
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

    best_streak, worst_streak = streaks_from_pnl(t["tradeRealized"])

    buys = int((t["dir"] == 1).sum()) if "dir" in t.columns else 0
    sells = int((t["dir"] == -1).sum()) if "dir" in t.columns else 0

    wins = int((t["tradeRealized"] > 0).sum())
    losses = int((t["tradeRealized"] < 0).sum())

    return {
        "trades": n,
        "buys": buys,
        "sells": sells,
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
    }

def side_breakdown(t: pd.DataFrame) -> pd.DataFrame:
    if "lado" not in t.columns:
        return pd.DataFrame()

    def pf_group(g):
        return profit_factor(g)

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
    # PF por grupo
    out["profit_factor"] = [pf_group(t[t["lado"] == r["lado"]]) for _, r in out.iterrows()]

    # rachas por lado
    streaks = []
    for lado in out["lado"].tolist():
        g = t[t["lado"] == lado].sort_values("exit_time")
        bw, wl = streaks_from_pnl(g["tradeRealized"])
        streaks.append((bw, wl))
    out["racha_ganadora_max"] = [x[0] for x in streaks]
    out["racha_perdedora_max"] = [x[1] for x in streaks]

    return out.sort_values("trades", ascending=False)

def interpretacion_rapida(s: dict):
    # reglas simples (ajustables)
    msgs = []
    warns = []

    if s["trades"] < 20:
        warns.append("Pocas operaciones en el filtro actual (<20). Las conclusiones pueden ser poco estables.")

    if not np.isnan(s["profit_factor"]):
        if s["profit_factor"] < 1.0:
            warns.append(f"Profit Factor {s['profit_factor']:.2f} (<1.0) → en promedio estás perdiendo dinero con este filtro.")
        elif s["profit_factor"] < 1.2:
            warns.append(f"Profit Factor {s['profit_factor']:.2f} (bajo) → falta selectividad o mejora de R:R / filtros.")
        else:
            msgs.append(f"Profit Factor {s['profit_factor']:.2f} → bien (hay edge en este filtro).")

    if s["max_drawdown"] < 0:
        dd = abs(s["max_drawdown"])
        pnl = abs(s["total_pnl"])
        if pnl > 0 and (dd / pnl) > 0.7:
            warns.append("Drawdown alto respecto al PnL total. Riesgo/estabilidad mejorables (filtros, guardia diaria, menos trades).")
        else:
            msgs.append("Drawdown razonable vs PnL (según este filtro).")

    if s["worst_streak"] >= 5:
        warns.append(f"Racha perdedora máxima = {s['worst_streak']} → necesitas plan para streaks (daily stop, reducción tamaño, filtro).")

    return msgs, warns

# ============================
# UI
# ============================
st.title("📊 WIC_WLF2 Analizador JSONL (Español)")
st.caption("Sube uno o varios archivos WIC_WLF2_YYYY-MM.jsonl → métricas claras, tablas legibles, gráficos y guía de interpretación.")

with st.expander("🧠 Cómo leer esto (muy importante)", expanded=True):
    st.markdown(
        """
- **Compra (Long)** = dir **1**. **Venta (Short)** = dir **-1**. (Ya lo mostramos traducido)
- **Equity (Curva de capital)**: suma acumulada de PnL trade a trade. Si sube estable → bien.
- **Drawdown**: cuánto caes desde el máximo anterior de equity. Te dice el “dolor” máximo.
- **Profit Factor (PF)**: Ganancias totales / Pérdidas totales.  
  - PF < 1.0 = mal (pierde dinero)  
  - 1.0–1.2 = débil  
  - > 1.2 = empieza a ser sólido (depende del sistema)
- **Expectancia**: PnL promedio por trade.  
- **Rachas**: te ayudan a decidir guardias diarias, tamaño de posición, filtros, etc.
        """
    )

with st.expander("🔒 Privacidad / sesiones", expanded=False):
    st.markdown(
        """
- Cada persona que abre el link tiene **su propia sesión**.
- Tus amigos **no ven** tus archivos subidos a menos que tú se los envíes o compartas pantalla.
- Este app **no guarda** logs en el servidor (solo analiza lo que se sube en esa sesión).
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
trades = pair_trades(df_all)

# ===== Filtros (sidebar)
st.sidebar.subheader("🎛️ Filtros")

date_min = trades["exit_time"].min()
date_max = trades["exit_time"].max()
if pd.isna(date_min) or pd.isna(date_max):
    st.error("No se pudo parsear 'ts'. Revisa formato 'YYYY-MM-DD HH:mm:ss.fff'.")
    st.stop()

d1, d2 = st.sidebar.date_input("Rango de fechas (por salida)", value=(date_min.date(), date_max.date()))
mask = (pd.to_datetime(trades["exit_time"]).dt.date >= d1) & (pd.to_datetime(trades["exit_time"]).dt.date <= d2)

lado_opts = sorted(trades["lado"].dropna().unique().tolist())
lado_sel = st.sidebar.multiselect("Lado", lado_opts, default=lado_opts)
mask &= trades["lado"].isin(lado_sel)

reason_opts_es = sorted(trades["exitReason_ES"].fillna("").unique().tolist())
reason_sel = st.sidebar.multiselect("Motivo de salida", reason_opts_es, default=reason_opts_es)
mask &= trades["exitReason_ES"].fillna("").isin(reason_sel)

t = trades[mask].copy()
t = t.sort_values("exit_time").reset_index(drop=True)

# recompute equity/drawdown on filtered view
t["equity"] = t["tradeRealized"].fillna(0).cumsum()
t["equity_peak"] = t["equity"].cummax()
t["drawdown"] = t["equity"] - t["equity_peak"]

s = summary_block(t)

st.caption(f"🧾 Líneas inválidas ignoradas al parsear: {bad_total}")

# ============================
# Bloque 1: Lo más importante
# ============================
st.subheader("✅ Lo más importante (resumen rápido)")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Operaciones", f"{s['trades']}")
c2.metric("Compras (Long)", f"{s['buys']}")
c3.metric("Ventas (Short)", f"{s['sells']}")
c4.metric("Ganadas", f"{s['wins']}")
c5.metric("Perdidas", f"{s['losses']}")
c6.metric("% Acierto", f"{(s['win_rate']*100):.1f}%" if not np.isnan(s["win_rate"]) else "n/a")

c7, c8, c9, c10, c11, c12 = st.columns(6)
c7.metric("PnL Total", f"{s['total_pnl']:.0f}")
c8.metric("Profit Factor", f"{s['profit_factor']:.2f}" if not np.isnan(s["profit_factor"]) else "n/a")
c9.metric("Expectancia", f"{s['expectancy']:.1f}" if not np.isnan(s["expectancy"]) else "n/a")
c10.metric("Max Drawdown", f"{s['max_drawdown']:.0f}")
c11.metric("Mejor racha (wins)", f"{s['best_streak']}")
c12.metric("Peor racha (losses)", f"{s['worst_streak']}")

c13, c14, c15, c16 = st.columns(4)
c13.metric("Mayor win", f"{s['max_win']:.0f}")
c14.metric("Mayor loss", f"{s['max_loss']:.0f}")
c15.metric("Win promedio", f"{s['avg_win']:.1f}" if not np.isnan(s["avg_win"]) else "n/a")
c16.metric("Loss promedio", f"{s['avg_loss']:.1f}" if not np.isnan(s["avg_loss"]) else "n/a")

msgs, warns = interpretacion_rapida(s)
for m in msgs:
    st.success(m)
for w in warns:
    st.warning(w)

# ============================
# Bloque 2: Por lado (Compra/Venta)
# ============================
st.subheader("🧭 Rendimiento por lado (Compra vs Venta)")
sb = side_breakdown(t)
if sb.empty:
    st.info("No hay suficiente información para agrupar por lado.")
else:
    # formateo
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
# Bloque 3: Motivos de salida (traducción + impacto)
# ============================
st.subheader("🚪 Motivos de salida (qué significa y cómo afecta)")
colA, colB = st.columns(2)

with colA:
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
    st.dataframe(by_exit, use_container_width=True, height=320)

with colB:
    st.markdown("**Leyenda rápida (desde tu código):**")
    st.markdown(
        """
- **Stop Loss (SL)**: salida por stop.
- **Take Profit (TP)**: salida por objetivo.
- **Break-even (BE)**: salida cerca del precio de entrada (sin pérdida).
- **Trailing Stop**: stop dinámico después de TP1 (runner).
- **Manual / Forzado**: cierre manual, guardia diaria, o cierre por hora.
        """
    )

# ============================
# Bloque 4: Gráficos (con explicación)
# ============================
st.subheader("📈 Gráficos (con sentido)")
g1, g2 = st.columns(2)

with g1:
    st.markdown("### Curva de capital (Equity)")
    st.caption("Suma acumulada de PnL. Si sube estable y con drawdowns controlados → mejor sistema.")
    st.line_chart(t.set_index("exit_time")["equity"])

with g2:
    st.markdown("### Drawdown")
    st.caption("Caída desde el último máximo de equity. Te muestra el peor 'dolor' del sistema.")
    st.line_chart(t.set_index("exit_time")["drawdown"])

st.markdown("### Distribución de PnL por trade")
st.caption("Te ayuda a ver si tienes muchas pérdidas pequeñas vs pocas grandes, o viceversa.")
st.bar_chart(t["tradeRealized"].round(0).value_counts().sort_index().head(200))

# ============================
# Bloque 5: Breakdowns útiles para tunear (OR/ATR/Hora)
# ============================
st.subheader("🔧 Breakdowns para tunear entradas (lo que sirve de verdad)")

colC, colD = st.columns(2)

with colC:
    st.markdown("**Por hora (hora de salida)**")
    by_hour = (t.groupby("exit_hour")
                 .agg(
                     trades=("atmId","count"),
                     win_rate=("tradeRealized", lambda x: float((x>0).mean()) if len(x) else np.nan),
                     avg_pnl=("tradeRealized","mean"),
                     total_pnl=("tradeRealized","sum"),
                 )
                 .reset_index()
                 .sort_values("exit_hour"))
    by_hour["win_rate"] = (by_hour["win_rate"]*100).round(1)
    by_hour.rename(columns={
        "exit_hour":"Hora",
        "trades":"Trades",
        "win_rate":"WinRate(%)",
        "avg_pnl":"PnL Prom",
        "total_pnl":"PnL Total",
    }, inplace=True)
    st.dataframe(by_hour, use_container_width=True, height=320)

with colD:
    st.markdown("**OR Size / ATR (solo si existe ENTRY enlazado)**")
    t_entry = t[t["has_entry"]].copy()

    # OR quartiles
    if "orSize" in t_entry.columns and t_entry["orSize"].notna().sum() >= 8:
        te = t_entry[t_entry["orSize"].notna()].copy()
        te["OR_bin"] = pd.qcut(te["orSize"], q=4, duplicates="drop")
        by_or = (te.groupby("OR_bin")
                   .agg(
                       trades=("atmId","count"),
                       win_rate=("tradeRealized", lambda x: float((x>0).mean()) if len(x) else np.nan),
                       avg_pnl=("tradeRealized","mean"),
                       total_pnl=("tradeRealized","sum"),
                       or_min=("orSize","min"),
                       or_med=("orSize","median"),
                       or_max=("orSize","max"),
                   )
                   .reset_index())
        by_or["win_rate"] = (by_or["win_rate"]*100).round(1)
        by_or["OR_bin"] = by_or["OR_bin"].astype(str)
        by_or.rename(columns={
            "OR_bin":"Rango OR (cuartil)",
            "trades":"Trades",
            "win_rate":"WinRate(%)",
            "avg_pnl":"PnL Prom",
            "total_pnl":"PnL Total",
            "or_min":"OR min",
            "or_med":"OR med",
            "or_max":"OR max",
        }, inplace=True)
        st.dataframe(by_or.sort_values("OR min"), use_container_width=True, height=220)
    else:
        st.info("No hay suficientes ENTRY con orSize para cuartiles de OR.")

    # ATR quartiles
    if "atr" in t_entry.columns and t_entry["atr"].notna().sum() >= 8:
        te = t_entry[t_entry["atr"].notna()].copy()
        te["ATR_bin"] = pd.qcut(te["atr"], q=4, duplicates="drop")
        by_atr = (te.groupby("ATR_bin")
                   .agg(
                       trades=("atmId","count"),
                       win_rate=("tradeRealized", lambda x: float((x>0).mean()) if len(x) else np.nan),
                       avg_pnl=("tradeRealized","mean"),
                       total_pnl=("tradeRealized","sum"),
                       atr_min=("atr","min"),
                       atr_med=("atr","median"),
                       atr_max=("atr","max"),
                   )
                   .reset_index())
        by_atr["win_rate"] = (by_atr["win_rate"]*100).round(1)
        by_atr["ATR_bin"] = by_atr["ATR_bin"].astype(str)
        by_atr.rename(columns={
            "ATR_bin":"Rango ATR (cuartil)",
            "trades":"Trades",
            "win_rate":"WinRate(%)",
            "avg_pnl":"PnL Prom",
            "total_pnl":"PnL Total",
            "atr_min":"ATR min",
            "atr_med":"ATR med",
            "atr_max":"ATR max",
        }, inplace=True)
        st.dataframe(by_atr.sort_values("ATR min"), use_container_width=True, height=220)
    else:
        st.info("No hay suficientes ENTRY con ATR para cuartiles de ATR.")

# ============================
# Bloque 6: Trades table (humana)
# ============================
st.subheader("📋 Lista de operaciones (vista humana)")

cols = []
for c in [
    "atmId","lado","entry_time","exit_time","dur_min",
    "tradeRealized","outcome","exitReason_ES","forcedCloseReason",
    "orSize","atr","ewo","deltaRatio","maxUnreal","minUnreal","source_file"
]:
    if c in t.columns:
        cols.append(c)

t_view = t[cols].copy()
t_view.rename(columns={
    "atmId":"TradeId",
    "lado":"Lado",
    "entry_time":"Entrada",
    "exit_time":"Salida",
    "dur_min":"Duración (min)",
    "tradeRealized":"PnL",
    "outcome":"Resultado",
    "exitReason_ES":"Motivo salida",
    "forcedCloseReason":"Motivo forzado",
    "orSize":"OR Size",
    "atr":"ATR",
    "ewo":"EWO",
    "deltaRatio":"DeltaRatio",
    "maxUnreal":"Max Unreal",
    "minUnreal":"Min Unreal",
    "source_file":"Archivo",
}, inplace=True)

st.dataframe(t_view, use_container_width=True, height=420)

# ============================
# Descarga
# ============================
st.subheader("⬇️ Exportar")
csv_bytes = t_view.to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar CSV (operaciones filtradas)",
    data=csv_bytes,
    file_name="WIC_WLF2_operaciones_filtradas.csv",
    mime="text/csv",
)
