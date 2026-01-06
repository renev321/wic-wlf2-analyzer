import json
import numpy as np
import pandas as pd
import streamlit as st

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

    # tolerancia mínima (tick) para evitar igualdad rara
    if trig >= oh:
        return 1.0
    if trig <= ol:
        return -1.0
    return np.nan

def pair_trades(df: pd.DataFrame, time_fallback_minutes: int = 45) -> pd.DataFrame:
    entries_raw = df[df["type"] == "ENTRY"].copy()
    exits_raw   = df[df["type"] == "EXIT"].copy()

    # Normaliza dirs
    if "dir" in entries_raw.columns:
        entries_raw.rename(columns={"dir": "dir_entry"}, inplace=True)
    if "dir" in exits_raw.columns:
        exits_raw.rename(columns={"dir": "dir_exit"}, inplace=True)

    # Asegura numéricos (si existen)
    for c in ["dir_entry", "dir_exit", "trigger", "orHigh", "orLow", "orSize", "atr", "ewo", "deltaRatio"]:
        if c in entries_raw.columns:
            entries_raw[c] = pd.to_numeric(entries_raw[c], errors="coerce")
        if c in exits_raw.columns:
            exits_raw[c] = pd.to_numeric(exits_raw[c], errors="coerce")

    # ===== 1) Pair por atmId (normal) =====
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

    # ===== 2) Fallback por tiempo (si falta ENTRY) =====
    # Idea: si un EXIT no encontró su ENTRY por atmId, buscamos el ENTRY más cercano ANTES del exit_time
    # dentro de una ventana (ej: 45 min) y que aún no esté "usado".

    missing_mask = t["entry_time"].isna()
    if missing_mask.any() and not entries_raw.empty:
        window = pd.Timedelta(minutes=time_fallback_minutes)

        # Prepara lista de ENTRY (raw) ordenada por tiempo
        e2 = entries_raw.copy()
        e2 = e2[e2["ts_parsed"].notna()].sort_values("ts_parsed").reset_index(drop=True)
        e2.rename(columns={"ts_parsed":"entry_time_raw"}, inplace=True)

        # Mantén un set de entradas ya asignadas (por índice en e2)
        used_entry_idx = set()

        # Exits a resolver, ordenados por tiempo
        unresolved = t[missing_mask & t["exit_time"].notna()].copy()
        unresolved = unresolved.sort_values("exit_time").reset_index()

        # puntero para ir avanzando entries <= exit_time
        ptr = 0
        candidates = []  # índices de entries disponibles (no usados), con entry_time_raw <= current exit_time

        for _, row in unresolved.iterrows():
            exit_time = row["exit_time"]
            # avanza ptr agregando entries que ya ocurrieron antes de exit_time
            while ptr < len(e2) and e2.loc[ptr, "entry_time_raw"] <= exit_time:
                candidates.append(ptr)
                ptr += 1

            # filtra candidatos no usados y dentro de ventana
            best = None
            best_time = None
            for idx in reversed(candidates):  # reversed para encontrar el más reciente rápido
                if idx in used_entry_idx:
                    continue
                et = e2.loc[idx, "entry_time_raw"]
                if exit_time - et <= window:
                    best = idx
                    best_time = et
                    break
                else:
                    # como estamos en reversed (más recientes primero), si este ya se pasó de ventana
                    # los más viejos estarán peor -> podemos cortar
                    # pero ojo: en reversed, si el más reciente ya está fuera, entonces TODOS están fuera.
                    break

            if best is None:
                continue

            used_entry_idx.add(best)

            # Copia campos del ENTRY RAW al trade en t
            ridx = row["index"]  # índice real en t
            for col in [
                "dir_entry","template","orderType","trigger",
                "orHigh","orLow","orSize","ewo","atr","useAtrEngine","atrSlMult",
                "tp1R","tp2R","tsBehindTP1Atr","trailStepTicks","deltaRatio","dailyPnL"
            ]:
                if col in e2.columns:
                    t.loc[ridx, col] = e2.loc[best, col]
            t.loc[ridx, "entry_time"] = best_time
            t.loc[ridx, "paired_by_time"] = True

    # ===== 3) Construye dir final (preferencia ENTRY) =====
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

    # ===== 4) Si aún falta dir, intenta inferencia por trigger/orHigh/orLow =====
    if "dir" in t.columns:
        still = t["dir"].isna()
        if still.any():
            inferred = t[still].apply(_infer_dir_from_entry_fields, axis=1)
            t.loc[still, "dir"] = inferred

    # ===== Outcome robusto (evita fillna(ndarray)) =====
    for c in ["tradeRealized","dayRealized","maxUnreal","minUnreal","orSize","atr","ewo","deltaRatio"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

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

    # flags / duración
    t["has_entry"] = t["entry_time"].notna()
    t["duration_sec"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds()
    t["dur_min"] = t["duration_sec"] / 60.0

    # Orden temporal + equity/drawdown
    t = t.sort_values("exit_time").reset_index(drop=True)
    t["equity"] = t.get("tradeRealized", 0).fillna(0).cumsum()
    t["equity_peak"] = t["equity"].cummax()
    t["drawdown"] = t["equity"] - t["equity_peak"]

    t["exit_date"] = pd.to_datetime(t["exit_time"]).dt.date
    t["exit_hour"] = pd.to_datetime(t["exit_time"]).dt.hour

    # Lado humano
    t["lado"] = t["dir"].apply(side_from_dir)

    return t

def profit_factor(t: pd.DataFrame) -> float:
    wins = t.loc[t["tradeRealized"] > 0, "tradeRealized"].sum()
    losses = t.loc[t["tradeRealized"] < 0, "tradeRealized"].sum()
    if losses == 0:
        return np.nan
    return float(wins / abs(losses))

def streaks_from_pnl(pnls: pd.Series):
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

    buys  = int((t["dir"] == 1).sum())
    sells = int((t["dir"] == -1).sum())
    unknown = int(t["dir"].isna().sum())

    wins = int((t["tradeRealized"] > 0).sum())
    losses = int((t["tradeRealized"] < 0).sum())

    paired_by_time = int(t.get("paired_by_time", False).sum()) if "paired_by_time" in t.columns else 0

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
    }

def side_breakdown(t: pd.DataFrame) -> pd.DataFrame:
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
        g = t[t["lado"] == lado].sort_values("exit_time")
        bw, wl = streaks_from_pnl(g["tradeRealized"])
        streak_w.append(bw)
        streak_l.append(wl)
    out["racha_ganadora_max"] = streak_w
    out["racha_perdedora_max"] = streak_l

    return out.sort_values("trades", ascending=False)

def interpretacion_rapida(s: dict):
    msgs = []
    warns = []

    if s["trades"] < 20:
        warns.append("Pocas operaciones en el filtro actual (<20). Las conclusiones pueden ser poco estables.")

    if not np.isnan(s["profit_factor"]):
        if s["profit_factor"] < 1.0:
            warns.append(f"Profit Factor {s['profit_factor']:.2f} → este conjunto de reglas pierde dinero en promedio.")
        elif s["profit_factor"] < 1.2:
            warns.append(f"Profit Factor {s['profit_factor']:.2f} → débil (falta filtro / gestión / horario).")
        else:
            msgs.append(f"Profit Factor {s['profit_factor']:.2f} → hay edge en este filtro.")

    if s["max_drawdown"] < 0:
        dd = abs(s["max_drawdown"])
        pnl = abs(s["total_pnl"])
        if pnl > 0 and (dd / pnl) > 0.7:
            warns.append("Drawdown alto vs PnL total → estabilidad mejorable (filtros, daily stop, menos trades, evitar horas malas).")

    if s["worst_streak"] >= 5:
        warns.append(f"Racha perdedora máxima = {s['worst_streak']} → necesitas plan anti-rachas (daily stop, bajar tamaño, filtros).")

    return msgs, warns

# ============================
# UI
# ============================
st.title("📊 WIC_WLF2 Analizador")
st.caption("Sube uno o varios archivos WIC_WLF2_YYYY-MM.jsonl → métricas claras, tablas legibles, gráficos y guía práctica.")

with st.expander("💡 Cómo leer esto", expanded=False):
    st.markdown(
        """
**Compra (Long) / Venta (Short)**  
- Ya lo mostramos como texto (nadie necesita ver números).

**Curva de capital (Equity)**  
- Tu PnL acumulado operación por operación.  
- Subida “suave” + drawdown pequeño → mejor estabilidad.

**Drawdown (Caída máxima)**  
- La peor caída desde el último máximo de equity (tu peor “dolor” histórico).

**Profit Factor (PF)**  
- Ganancias totales / pérdidas totales.  
- < 1.0 → pierde dinero  
- 1.0–1.2 → débil  
- > 1.2 → empieza a ser interesante (depende del riesgo)

**Expectancia**  
- PnL promedio por operación. Si es negativa, necesitas ajustar filtros o gestión.
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

# Pairing con fallback por tiempo (ajustable)
with st.sidebar:
    st.subheader("⚙️ Ajustes de emparejamiento")
    fallback_minutes = st.slider("Ventana fallback por tiempo (min)", 5, 180, 45, 5)

trades = pair_trades(df_all, time_fallback_minutes=fallback_minutes)

# ===== filtros (sidebar)
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

s = summary_block(t)

st.caption(f"🧾 Líneas inválidas ignoradas al parsear: {bad_total}")
if s["paired_by_time"] > 0:
    st.info(f"🧩 Trades emparejados por fallback de tiempo: {s['paired_by_time']} (cuando falló atmId).")

if s["unknown_dir"] > 0:
    st.warning(
        f"⚠️ {s['unknown_dir']} operaciones siguen sin dirección. "
        f"Eso significa que NO hay datos de ENTRY suficientes (ni por atmId ni por tiempo) "
        f"para inferir el lado con seguridad."
    )

# ============================
# Resumen rápido
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
# Por lado
# ============================
st.subheader("🧭 Rendimiento por lado (Compra vs Venta)")
sb = side_breakdown(t)
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
    st.caption("Tu PnL acumulado operación por operación.")
    st.line_chart(t.set_index("exit_time")["equity"])

with g2:
    st.markdown("### Drawdown (caída desde máximos)")
    st.caption("Cuánto cae tu equity desde el último máximo. Drawdowns grandes = más dolor/riesgo.")
    st.line_chart(t.set_index("exit_time")["drawdown"])

# ============================
# Tabla trades
# ============================
st.subheader("📋 Lista de operaciones")

cols = []
for c in [
    "atmId","lado","paired_by_time","entry_time","exit_time","dur_min",
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
    "paired_by_time":"Fallback tiempo",
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
    "maxUnreal":"Max Unreal",
    "minUnreal":"Min Unreal",
    "source_file":"Archivo",
}, inplace=True)

st.dataframe(t_view, use_container_width=True, height=420)

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
