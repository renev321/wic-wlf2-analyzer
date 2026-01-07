import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px

st.set_page_config(page_title="WIC_WLF2 Analizador", layout="wide")

# ============================================================
# (Opcional) Password: define APP_PASSWORD en Streamlit Secrets
# ============================================================
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")
if APP_PASSWORD:
    st.sidebar.subheader("🔐 Acceso")
    pwd = st.sidebar.text_input("Contraseña", type="password")
    if pwd != APP_PASSWORD:
        st.sidebar.warning("Contraseña incorrecta.")
        st.stop()

# ============================================================
# Helpers parse/normalize
# ============================================================
def parse_jsonl_bytes(b: bytes):
    txt = b.decode("utf-8", errors="replace")
    recs, bad = [], 0
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            bad += 1
    return recs, bad


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ts" in df.columns:
        df["ts_parsed"] = pd.to_datetime(df["ts"], errors="coerce")
    elif "timestamp" in df.columns:
        df["ts_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["ts_parsed"] = pd.NaT

    if "type" not in df.columns:
        df["type"] = ""

    return df

# ============================================================
# Arrays (volumen/delta/OHLC) y métricas pre-entrada
# ============================================================
def _as_list(x):
    """Convierte a lista si viene como list/tuple/np.array o string tipo '[...]'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, np.ndarray)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except Exception:
                return []
    return []


def add_pressure_activity_features(trades: pd.DataFrame) -> pd.DataFrame:
    """Añade métricas de 'Presión y Actividad' usando arrays LastN (contexto pre-entrada).
    Si no existen arrays, devuelve el DF sin cambios.
    """
    if trades is None or trades.empty:
        return trades

    t = trades.copy()

    # Asegura numéricos base si existen
    for c in ["tickSize", "pointValue", "slTicks", "qtyTP1", "qtyRunner", "volEntrySoFar", "deltaEntrySoFar", "dir"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    def _row_feat(r):
        dl = _as_list(r.get("deltaLastN"))
        vl = _as_list(r.get("volLastN"))
        ol = _as_list(r.get("openLastN"))
        hl = _as_list(r.get("highLastN"))
        ll = _as_list(r.get("lowLastN"))
        cl = _as_list(r.get("closeLastN"))

        if not dl or not vl or not hl or not ll or not cl or not ol:
            return pd.Series({
                "pre_presion_sum": np.nan,
                "pre_presion_abs": np.nan,
                "pre_vol_sum": np.nan,
                "pre_precio_neto": np.nan,
                "pre_rango_precio": np.nan,
                "pre_mov_ticks": np.nan,
                "pre_absorcion": np.nan,
                "pre_actividad_rel": np.nan,
                "pre_presion_por_vol": np.nan,
                "pre_sin_apoyo": False,
            })

        presion_sum = float(np.nansum(dl))
        presion_abs = float(abs(presion_sum))
        vol_sum = float(np.nansum(vl))

        # Movimiento de precio pre-entrada
        try:
            precio_neto = float(cl[-1] - ol[0])
        except Exception:
            precio_neto = np.nan

        try:
            rango = float(np.nanmax(hl) - np.nanmin(ll))
        except Exception:
            rango = np.nan

        tick = r.get("tickSize")
        if tick is None or (isinstance(tick, float) and np.isnan(tick)) or tick == 0 or rango != rango:
            mov_ticks = np.nan
        else:
            mov_ticks = rango / float(tick)

        # Absorción: presión alta + avance bajo (normalizado por ticks movidos)
        absorcion = (presion_abs / max(mov_ticks, 1.0)) if mov_ticks == mov_ticks else np.nan

        # Actividad relativa: vol hasta entrada vs mediana del vol reciente
        med_vol = float(np.nanmedian(vl)) if len(vl) else np.nan
        vol_entry = r.get("volEntrySoFar")
        if med_vol == med_vol and med_vol > 0 and vol_entry == vol_entry:
            actividad_rel = float(vol_entry / med_vol)
        else:
            actividad_rel = np.nan

        # Presión por volumen (magnitud relativa)
        presion_por_vol = float(presion_abs / vol_sum) if vol_sum == vol_sum and vol_sum > 0 else np.nan

        # “Sin apoyo”: precio avanza en una dirección pero la presión va al lado opuesto
        d = r.get("dir")
        sin_apoyo = False
        if d == d and precio_neto == precio_neto:
            if d > 0 and precio_neto > 0 and presion_sum < 0:
                sin_apoyo = True
            elif d < 0 and precio_neto < 0 and presion_sum > 0:
                sin_apoyo = True

        return pd.Series({
            "pre_presion_sum": presion_sum,
            "pre_presion_abs": presion_abs,
            "pre_vol_sum": vol_sum,
            "pre_precio_neto": precio_neto,
            "pre_rango_precio": rango,
            "pre_mov_ticks": mov_ticks,
            "pre_absorcion": absorcion,
            "pre_actividad_rel": actividad_rel,
            "pre_presion_por_vol": presion_por_vol,
            "pre_sin_apoyo": sin_apoyo,
        })

    feats = t.apply(_row_feat, axis=1)
    for c in feats.columns:
        t[c] = feats[c]

    return t


# ============================================================
# Formatting & traffic-light helpers
# ============================================================
def pct(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:.{d}f}%"

def fmt(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:.{d}f}"

def traffic_pf(pf: float):
    if pf is None or np.isnan(pf):
        return "⚪ PF N/A"
    if pf < 1.0:  return "🔴 PF < 1 (pierde)"
    if pf < 1.2:  return "🟡 PF 1.0–1.2 (débil)"
    if pf < 1.5:  return "🟢 PF 1.2–1.5 (bueno)"
    return "🟣 PF > 1.5 (muy bueno)"

def traffic_exp(exp: float):
    if exp is None or np.isnan(exp):
        return "⚪ Promedio N/A"
    if exp < 0:   return "🔴 Promedio < 0"
    if exp < 10:  return "🟡 Promedio bajo pero positivo"
    return "🟢 Promedio sólido"


# ============================================================
# Core metrics
# ============================================================
def profit_factor(trades: pd.DataFrame) -> float:
    wins = trades.loc[trades["tradeRealized"] > 0, "tradeRealized"].sum()
    losses = trades.loc[trades["tradeRealized"] < 0, "tradeRealized"].sum()
    if losses == 0:
        return np.nan
    return float(wins / abs(losses))


def max_streak(outcomes: pd.Series, target: str):
    best_len, cur = 0, 0
    best_end = None
    for i, o in enumerate(outcomes.tolist()):
        if o == target:
            cur += 1
            if cur > best_len:
                best_len = cur
                best_end = i
        else:
            cur = 0
    if best_len == 0:
        return 0, None, None
    start = best_end - best_len + 1
    return best_len, start, best_end


def drawdown_details(t: pd.DataFrame):
    if t.empty:
        return np.nan, None, None
    dd = t["drawdown"].fillna(0)
    trough_idx = int(dd.idxmin())
    trough_time = t.loc[trough_idx, "exit_time"]
    peak_idx = int(t.loc[:trough_idx, "equity"].idxmax())
    peak_time = t.loc[peak_idx, "exit_time"]
    return float(dd.min()), peak_time, trough_time


def hour_bucket_label(h):
    if pd.isna(h):
        return "Sin hora"
    h = int(h)
    return f"{h:02d}:00–{h:02d}:59"



def pair_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 fila por trade (atmId):
    - EXIT: último por atmId
    - ENTRY: primero por atmId
    Si falta ENTRY -> lado = "Sin datos (faltó ENTRY)" (NO inventamos)
    """
    entries = df[df["type"] == "ENTRY"].copy()
    exits = df[df["type"] == "EXIT"].copy()

    entry_cols = [
        "atmId", "ts_parsed", "dir",
        "instrument", "tickSize", "pointValue",
        "template", "orderType", "trigger",
        "orHigh", "orLow", "orSize",
        "ewo", "atr", "useAtrEngine", "atrSlMult",
        "tp1R", "tp2R", "tp1Ticks", "tp2Ticks",
        "tsBehindTP1Atr", "trailStepTicks",
        "slTicks", "qtyTP1", "qtyRunner", "avgEntry",
        "deltaRatio", "dailyPnL",
        "cvd", "deltaBar", "deltaPrevBar",
        "volEntrySoFar", "deltaEntrySoFar",
        "snapN", "snapIncludesEntryBar",
        "volLastN", "deltaLastN",
        "openLastN", "highLastN", "lowLastN", "closeLastN"
    ]
    entry_cols = [c for c in entry_cols if c in entries.columns]

    if len(entry_cols) == 0:
        e1 = pd.DataFrame(columns=["atmId", "entry_time"])
    else:
        e1 = (entries.sort_values("ts_parsed")
                    .groupby("atmId", as_index=False)[entry_cols]
                    .first()
                    .rename(columns={"ts_parsed": "entry_time"}))

    exit_cols = [
        "atmId", "ts_parsed",
        "outcome", "exitReason", "tradeRealized", "dayRealized",
        "maxUnreal", "minUnreal", "forcedCloseReason", "dailyHalt",
        "instrument", "tickSize", "pointValue",
        "slTicks", "qtyTP1", "qtyRunner", "avgEntry",
        "cvd", "deltaBar", "deltaPrevBar"
    ]
    exit_cols = [c for c in exit_cols if c in exits.columns]

    if len(exit_cols) == 0:
        x1 = pd.DataFrame(columns=["atmId", "exit_time", "tradeRealized"])
    else:
        x1 = (exits.sort_values("ts_parsed")
                    .groupby("atmId", as_index=False)[exit_cols]
                    .last()
                    .rename(columns={"ts_parsed": "exit_time"}))

    t = pd.merge(x1, e1, on="atmId", how="left")

    # Unifica columnas duplicadas de EXIT/ENTRY (pandas crea sufijos _x/_y)
    def _coalesce(name: str):
        x = f"{name}_x"
        y = f"{name}_y"
        if name in t.columns:
            return
        if (x in t.columns) or (y in t.columns):
            base = t[y] if y in t.columns else pd.Series([np.nan]*len(t))
            if x in t.columns:
                base = base.where(base.notna(), t[x])
            t[name] = base
            drop_cols = [c for c in [x, y] if c in t.columns]
            if drop_cols:
                t.drop(columns=drop_cols, inplace=True)

    for nm in [
        "dir", "instrument", "tickSize", "pointValue",
        "slTicks", "qtyTP1", "qtyRunner", "avgEntry",
        "template", "orderType", "trigger",
        "orHigh", "orLow", "orSize",
        "ewo", "atr", "useAtrEngine", "atrSlMult",
        "tp1R", "tp2R", "tp1Ticks", "tp2Ticks",
        "tsBehindTP1Atr", "trailStepTicks",
        "deltaRatio", "dailyPnL",
        "cvd", "deltaBar", "deltaPrevBar"
    ]:
        _coalesce(nm)

    t["has_entry"] = t.get("entry_time").notna() if "entry_time" in t.columns else False

    # Numéricos
    for c in [
        "tradeRealized", "dayRealized", "maxUnreal", "minUnreal",
        "orSize", "atr", "ewo", "deltaRatio", "dir",
        "atrSlMult", "tp1R", "tp2R", "trailStepTicks",
        "tickSize", "pointValue", "slTicks", "qtyTP1", "qtyRunner", "avgEntry",
        "volEntrySoFar", "deltaEntrySoFar"
    ]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    # outcome robusto si falta
    if "tradeRealized" in t.columns:
        calc_outcome = np.where(t["tradeRealized"].fillna(0) >= 0, "WIN", "LOSS")
        if "outcome" not in t.columns:
            t["outcome"] = calc_outcome
        else:
            t["outcome"] = t["outcome"].where(t["outcome"].notna(), calc_outcome)

    # Duración
    if "entry_time" in t.columns and "exit_time" in t.columns:
        t["duration_sec"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds()
    else:
        t["duration_sec"] = np.nan

    # Equity / drawdown
    t = t.sort_values("exit_time").reset_index(drop=True)
    t["equity"] = t["tradeRealized"].fillna(0).cumsum()
    t["equity_peak"] = t["equity"].cummax()
    t["drawdown"] = t["equity"] - t["equity_peak"]

    # Hora
    t["exit_hour"] = t["exit_time"].dt.hour
    t["exit_hour_label"] = t["exit_hour"].apply(hour_bucket_label)

    # Lado (Compra/Venta) solo si hay dir
    def side_label(x):
        if pd.isna(x):
            return "Sin datos (faltó ENTRY)"
        if x > 0:
            return "Compra (Long)"
        if x < 0:
            return "Venta (Short)"
        return "Sin datos (dir=0)"

    if "dir" in t.columns:
        t["lado"] = t["dir"].apply(side_label)
    else:
        t["lado"] = "Sin datos (faltó ENTRY)"

    # ============================================================
    # Métricas tuning: Riesgo estimado, Ganancia/Riesgo, Captura/Devolución
    # ============================================================
    if "tickSize" in t.columns and "pointValue" in t.columns:
        t["tick_value"] = t["tickSize"] * t["pointValue"]
    else:
        t["tick_value"] = np.nan

    if "qtyTP1" in t.columns or "qtyRunner" in t.columns:
        q1 = t["qtyTP1"] if "qtyTP1" in t.columns else 0
        qr = t["qtyRunner"] if "qtyRunner" in t.columns else 0
        t["qty_total"] = pd.to_numeric(q1, errors="coerce").fillna(0) + pd.to_numeric(qr, errors="coerce").fillna(0)
    else:
        t["qty_total"] = np.nan

    if "slTicks" in t.columns and "tick_value" in t.columns and "qty_total" in t.columns:
        t["risk_$"] = t["slTicks"] * t["tick_value"] * t["qty_total"]
        t.loc[t["risk_$"] <= 0, "risk_$"] = np.nan
    else:
        t["risk_$"] = np.nan

    if "tradeRealized" in t.columns:
        t["rr"] = t["tradeRealized"] / t["risk_$"]
    else:
        t["rr"] = np.nan

    if "tradeRealized" in t.columns and "maxUnreal" in t.columns:
        winners = t["tradeRealized"] > 0
        t["captura_pct"] = np.where(winners & (t["maxUnreal"] > 0), t["tradeRealized"] / t["maxUnreal"], np.nan)
        t["devolucion_pct"] = np.where(winners & (t["maxUnreal"] > 0),
                                       (t["maxUnreal"] - t["tradeRealized"]) / t["maxUnreal"], np.nan)
    else:
        t["captura_pct"] = np.nan
        t["devolucion_pct"] = np.nan

    # Presión/Actividad pre-entrada (si hay arrays)
    t = add_pressure_activity_features(t)

    return t


def summarize(t: pd.DataFrame) -> dict:
    if t.empty:
        return {}
    n = len(t)
    wins = int((t["tradeRealized"] > 0).sum())
    losses = int((t["tradeRealized"] < 0).sum())
    win_rate = (wins / n * 100) if n else np.nan

    pf = profit_factor(t)
    expectancy = float(t["tradeRealized"].mean()) if n else np.nan

    max_dd, dd_peak_time, dd_trough_time = drawdown_details(t)

    max_win = float(t["tradeRealized"].max())
    max_loss = float(t["tradeRealized"].min())

    outcomes = pd.Series(np.where(t["tradeRealized"].fillna(0) >= 0, "WIN", "LOSS"))
    wlen, _, _ = max_streak(outcomes, "WIN")
    llen, _, _ = max_streak(outcomes, "LOSS")

    return {
        "n": n, "wins": wins, "losses": losses, "win_rate": win_rate,
        "pnl_total": float(t["tradeRealized"].sum()),
        "pf": pf, "expectancy": expectancy,
        "max_dd": max_dd, "dd_peak_time": dd_peak_time, "dd_trough_time": dd_trough_time,
        "max_win": max_win, "max_loss": max_loss,
        "best_win_streak": wlen, "best_loss_streak": llen,
    }


# ============================================================
# Grouping / bins
# ============================================================
def make_bins_quantiles(df: pd.DataFrame, col: str, q: int):
    s = df[col].dropna()
    if len(s) < q * 10:
        return None
    try:
        return pd.qcut(df[col], q=q, duplicates="drop")
    except Exception:
        return None



def group_metrics(df: pd.DataFrame, group_col: str, min_trades: int, recommended_trades: int = None):
    """Métricas por grupo con control de muestra.
    - NO oculta grupos pequeños: los marca como 🟢/🟡/🔴.
    - 'WinRate (ajustado)' evita que 2/2 (=100%) gane contra 30 trades.
    """
    if recommended_trades is None:
        recommended_trades = max(60, min_trades)

    rows = []
    for g, sub in df.groupby(group_col):
        n = int(len(sub))
        if n == 0:
            continue

        tr = pd.to_numeric(sub.get("tradeRealized"), errors="coerce")
        wins = int((tr > 0).sum()) if tr is not None else 0
        wr = wins / n * 100 if n else np.nan
        wr_adj = (wins + 1) / (n + 2) * 100  # suavizado

        pf = profit_factor(sub) if "tradeRealized" in sub.columns else np.nan
        exp = float(tr.mean()) if tr is not None else np.nan
        pnl = float(tr.sum()) if tr is not None else np.nan

        score = exp * np.log1p(n) if exp == exp else np.nan

        if n >= recommended_trades:
            estado = "🟢 Suficiente"
        elif n >= min_trades:
            estado = "🟡 Muestra pequeña"
        else:
            estado = "🔴 No concluyente"

        rows.append({
            "Grupo": str(g),
            "Trades": n,
            "Estado": estado,
            "WinRate %": wr,
            "WinRate (ajustado) %": wr_adj,
            "Profit Factor": pf,
            "Promedio por trade": exp,
            "PnL Total": pnl,
            "Score (ponderado)": score,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Score (ponderado)", "Trades"], ascending=[False, False], na_position="last").reset_index(drop=True)
    return out


# ============================================================
# Advice engines (NEXT LEVEL)
# ============================================================
def advice_from_table(tbl: pd.DataFrame, title: str, min_trades: int):
    if tbl is None or tbl.empty:
        st.info(f"En **{title}** no hay suficiente muestra (mínimo {min_trades} trades por grupo).")
        return

    best = tbl.iloc[0]
    worst = tbl.iloc[-1]

    st.markdown("**✅ Consejos rápidos (tabla):**")
    st.write(
        f"🏆 Mejor: **{best['Grupo']}** | Trades={int(best['Trades'])} | PF={fmt(best['Profit Factor'],2)} | "
        f"Promedio/trade={fmt(best['Promedio por trade'],1)} | PnL={fmt(best['PnL Total'],0)}"
    )
    st.write(
        f"🧨 Peor: **{worst['Grupo']}** | Trades={int(worst['Trades'])} | PF={fmt(worst['Profit Factor'],2)} | "
        f"Promedio/trade={fmt(worst['Promedio por trade'],1)} | PnL={fmt(worst['PnL Total'],0)}"
    )

    # warnings
    if best["Trades"] < min_trades * 2:
        st.warning("⚠️ El mejor grupo tiene muestra pequeña. Ideal confirmar con más meses de logs.")
    if not np.isnan(best["Profit Factor"]) and best["Profit Factor"] < 1.0:
        st.error("🚨 Incluso el mejor grupo tiene PF < 1 → este filtro no está salvando el sistema en estos datos.")
    if not np.isnan(worst["Profit Factor"]) and worst["Profit Factor"] < 1.0:
        st.warning("👉 Hay grupos con PF < 1 → candidatos a filtrar/evitar.")

    st.caption("Nota: WinRate Ajustado + Score ponderado evitan que 2 trades al 100% “ganen” contra 30 trades.")


def pnl_shape_insights(t: pd.DataFrame):
    if t.empty:
        return []

    pnl = t["tradeRealized"].dropna()
    if pnl.empty:
        return []

    med = float(pnl.median())
    p25 = float(pnl.quantile(0.25))
    p75 = float(pnl.quantile(0.75))
    p10 = float(pnl.quantile(0.10))
    p90 = float(pnl.quantile(0.90))

    max_win = float(pnl.max())
    max_loss = float(pnl.min())
    avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else np.nan
    avg_loss = float(pnl[pnl < 0].mean()) if (pnl < 0).any() else np.nan

    insights = []
    insights.append(f"📌 Mediana PnL: **{med:.1f}** | IQR (25–75%): **[{p25:.1f}, {p75:.1f}]**")

    if not np.isnan(avg_win) and not np.isnan(avg_loss) and avg_loss != 0:
        ratio = abs(avg_win / avg_loss)
        if ratio < 0.8:
            insights.append("⚠️ Pérdidas promedio > ganancias promedio → revisa SL, entradas tardías, o horarios peligrosos.")
        elif ratio < 1.2:
            insights.append("🟡 Ganancia y pérdida promedio similares → el edge depende más del winrate + filtros.")
        else:
            insights.append("🟢 Ganancias promedio > pérdidas promedio → buena relación base si el winrate acompaña.")

    if max_win != 0:
        bomb = abs(max_loss) / abs(max_win)
        if bomb >= 1.5:
            insights.append("🚨 Pérdidas gigantes vs la mayor ganancia. Revisa stops, slippage, noticias, o “chasing”.")
        elif bomb >= 1.0:
            insights.append("⚠️ La peor pérdida compite con la mejor ganancia. Controla los outliers.")

    insights.append(f"🧭 Zona típica (10–90%): **[{p10:.1f}, {p90:.1f}]**. Fuera de esto son outliers.")
    return insights


def equity_recovery_insights(t: pd.DataFrame):
    if t.empty:
        return []

    pnl = t["tradeRealized"].fillna(0)
    exp = float(pnl.mean()) if len(pnl) else np.nan
    std = float(pnl.std()) if len(pnl) > 2 else np.nan

    max_dd, peak_t, trough_t = drawdown_details(t)

    insights = []
    if peak_t is not None and trough_t is not None:
        insights.append(f"📉 Max Drawdown: **{max_dd:.0f}** (desde {peak_t} hasta {trough_t})")
    else:
        insights.append(f"📉 Max Drawdown: **{max_dd:.0f}**")

    if exp < 0:
        insights.append("🚨 Promedio por trade negativo: necesitas filtros fuertes o cambiar lógica antes de “tunear fino”.")
    else:
        insights.append(f"✅ Promedio por trade: **{exp:.1f}** ({traffic_exp(exp)})")

    if not np.isnan(std) and std > 0 and not np.isnan(exp):
        cv = abs(std / exp) if exp != 0 else np.inf
        if exp > 0 and cv > 6:
            insights.append("⚠️ PnL muy volátil vs el promedio → sube muestra o filtra momentos de alta variabilidad.")
        elif exp > 0 and cv > 3:
            insights.append("🟡 Variabilidad moderada-alta → usa guardias diarias conservadoras.")
        elif exp > 0:
            insights.append("🟢 PnL relativamente estable vs el promedio.")

    if exp > 0 and not np.isnan(max_dd):
        dd_trades_equiv = abs(max_dd / exp)
        if dd_trades_equiv > 80:
            insights.append("🚨 Drawdown enorme relativo al promedio → recuperación puede tardar mucho.")
        elif dd_trades_equiv > 40:
            insights.append("⚠️ Drawdown alto relativo al promedio → requiere disciplina (guardia diaria / filtros).")
        else:
            insights.append("🟢 Drawdown razonable relativo al promedio.")

    return insights


def factor_danger_zone_insights(df_known: pd.DataFrame, xcol: str, q: int, min_trades: int, title: str):
    bins = make_bins_quantiles(df_known, xcol, q)
    if bins is None:
        return ["ℹ️ No hay suficiente data para detectar zonas por rangos."]

    tmp = df_known.copy()
    tmp["_bin"] = bins.astype(str)
    tbl = group_metrics(tmp, "_bin", min_trades=min_trades)

    if tbl.empty:
        return [f"ℹ️ No hay rangos con ≥ {min_trades} trades para {title}."]

    # zonas malas
    bad = tbl[(tbl["Promedio por trade"] < 0) & (tbl["Profit Factor"] < 1.0)].copy()
    good = tbl[(tbl["Promedio por trade"] > 0) & (tbl["Profit Factor"] >= 1.2)].copy()

    insights = []
    if not bad.empty:
        b = bad.sort_values("Score (ponderado)").iloc[0]
        insights.append(f"🚫 Zona peligrosa: **{b['Grupo']}** → promedio<0 y PF<1 (candidato a EVITAR).")
    else:
        insights.append("✅ No se detectan rangos claramente peligrosos (con muestra suficiente).")

    if not good.empty:
        g = good.sort_values("Score (ponderado)", ascending=False).iloc[0]
        insights.append(f"✅ Zona fuerte: **{g['Grupo']}** → PF≥1.2 y promedio>0 (candidato a priorizar).")

    spread = float(tbl["Promedio por trade"].max() - tbl["Promedio por trade"].min())
    if spread < 5:
        insights.append("ℹ️ Este factor casi no separa rendimiento (spread pequeño) → filtro débil por sí solo.")
    else:
        insights.append("📌 Este factor SÍ separa rendimiento → útil para tunear rangos.")
    return insights


# ============================================================
# Charts
# ============================================================
def plot_equity_drawdown(t: pd.DataFrame):
    fig1 = px.line(t, x="exit_time", y="equity", title="Equity (curva de capital acumulada)")
    fig1.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))

    fig2 = px.line(t, x="exit_time", y="drawdown", title="Drawdown (caída desde el máximo de equity)")
    fig2.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
    fig2.add_hline(y=0, line_width=1, line_dash="dash")
    return fig1, fig2


def plot_pnl_hist(t: pd.DataFrame):
    fig = px.histogram(t, x="tradeRealized", nbins=40, title="Distribución de PnL por operación")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    fig.add_vline(x=0, line_width=1, line_dash="dash")
    return fig


def plot_scatter_advanced(df_known: pd.DataFrame, xcol: str, title: str):
    cols = [c for c in [xcol, "tradeRealized", "exit_time", "lado", "exitReason"] if c in df_known.columns]
    tmp = df_known[cols].dropna(subset=[xcol, "tradeRealized"]).copy()
    if tmp.empty:
        st.info("No hay suficiente data para el scatter.")
        return

    tmp["Resultado"] = np.where(tmp["tradeRealized"] >= 0, "Ganancia", "Pérdida")

    fig = px.scatter(
        tmp,
        x=xcol,
        y="tradeRealized",
        color="Resultado",
        hover_data=[c for c in ["exit_time", "lado", "exitReason", "tradeRealized"] if c in tmp.columns],
        title=title
    )
    fig.update_traces(marker=dict(size=7, opacity=0.60))
    fig.add_hline(y=0, line_width=1, line_dash="dash")
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Lectura: verde = PnL ≥ 0, rojo = PnL < 0. Con muestra pequeña, úsalo como idea, no como regla.")



def plot_factor_bins(df_known: pd.DataFrame, col: str, q: int, min_trades: int, recommended_trades: int, title: str,
                     show_scatter: bool, scatter_df: pd.DataFrame):
    bins = make_bins_quantiles(df_known, col, q)
    if bins is None:
        st.info(f"No hay suficiente data para crear rangos en **{title}**.")
        return

    tmp = df_known.copy()
    tmp["_bin"] = bins.astype(str)

    tbl = group_metrics(tmp, "_bin", min_trades=min_trades, recommended_trades=recommended_trades)
    if tbl.empty:
        st.info(f"En **{title}** no hay data para agrupar.")
        return

    # Leyenda de muestra
    n_ok = int((tbl["Estado"] == "🟢 Suficiente").sum())
    n_small = int((tbl["Estado"] == "🟡 Muestra pequeña").sum())
    n_bad = int((tbl["Estado"] == "🔴 No concluyente").sum())
    st.caption(f"Muestra por rango: 🟢{n_ok} 🟡{n_small} 🔴{n_bad}  | "
               f"Mínimo para mirar: {min_trades}  | Recomendado para decidir: {recommended_trades}")

    # Para gráficos: solo rangos con muestra mínima
    tbl_chart = tbl[tbl["Trades"] >= min_trades].copy()

    if not tbl_chart.empty:
        # Promedio por trade: escala divergente centrada en 0 (azul=negativo, rojo=positivo)
        fig_exp = px.bar(
            tbl_chart,
            x="Grupo",
            y="Promedio por trade",
            color="Promedio por trade",
            color_continuous_scale="RdBu_r",
            title=f"{title} → Promedio por trade (rangos)",
            text_auto=True,
        )
        fig_exp.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), coloraxis_colorbar_title="Promedio")
        fig_exp.add_hline(y=0, line_width=1, line_dash="dash")

        # Profit Factor: rojo=bajo, verde=alto
        fig_pf = px.bar(
            tbl_chart,
            x="Grupo",
            y="Profit Factor",
            color="Profit Factor",
            color_continuous_scale="RdYlGn",
            title=f"{title} → Profit Factor (rangos)",
            text_auto=True,
        )
        fig_pf.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), coloraxis_colorbar_title="PF")
        fig_pf.add_hline(y=1.0, line_width=1, line_dash="dash")

        st.plotly_chart(fig_exp, use_container_width=True)
        st.plotly_chart(fig_pf, use_container_width=True)
        st.caption("Profit Factor (PF) = Ganancias totales / Pérdidas totales. PF>1 indica que el conjunto gana; PF>1.5 suele ser muy sólido (con muestra suficiente).")

        advice_from_table(tbl_chart, title=title, min_trades=min_trades)
    else:
        st.warning("⚠️ Todo queda con muestra muy pequeña para gráficos. Se muestra tabla completa abajo.")

    st.dataframe(tbl, use_container_width=True)

    st.markdown("**🧠 Zonas recomendadas / peligrosas (automático):**")
    for s in factor_danger_zone_insights(df_known, col, q, min_trades, title):
        if "🚫" in s:
            st.warning(s)
        elif "✅" in s:
            st.success(s)
        elif "🚨" in s:
            st.error(s)
        else:
            st.info(s)

    if show_scatter:
        plot_scatter_advanced(scatter_df, col, f"{title}: Scatter PnL vs {col}")


def plot_hour_analysis(t: pd.DataFrame, min_trades: int):
    tbl = group_metrics(t, "exit_hour_label", min_trades=min_trades)
    if tbl.empty:
        st.info(f"No hay suficientes trades por hora para min_trades={min_trades}.")
        return None

    fig = px.bar(tbl, x="Grupo", y="Score (ponderado)",
                 title="Horas más prometedoras (Score ponderado por tamaño de muestra)")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(tbl, use_container_width=True)
    advice_from_table(tbl, title="Hora", min_trades=min_trades)
    return tbl


def plot_heatmap_weekday_hour(t: pd.DataFrame, min_trades: int):
    tmp = t.copy()

    if "exit_time" not in tmp.columns or tmp["exit_time"].isna().all():
        st.info("No hay exit_time válido para generar el heatmap.")
        return

    tmp = tmp.dropna(subset=["exit_time"]).copy()
    tmp["weekday"] = tmp["exit_time"].dt.day_name()
    if "exit_hour" not in tmp.columns or tmp["exit_hour"].isna().all():
        tmp["exit_hour"] = tmp["exit_time"].dt.hour

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    tmp["weekday_order"] = pd.Categorical(tmp["weekday"], categories=weekday_order, ordered=True)

    agg = tmp.groupby(["weekday_order", "exit_hour"]).agg(
        Trades=("tradeRealized", "size"),
        Promedio=("tradeRealized", "mean"),
    ).reset_index()

    agg.loc[agg["Trades"] < min_trades, "Promedio"] = np.nan
    pivot = agg.pivot(index="weekday_order", columns="exit_hour", values="Promedio")

    # Si con el mínimo actual no queda nada visible, avisamos y no dibujamos un heatmap vacío.
    if pivot.count().sum() == 0:
        st.info(
            f"No hay celdas con ≥ {min_trades} trades para el heatmap. "
            "Baja el mínimo o usa el panel por hora (arriba), que requiere menos muestra."
        )
        return

    # Escala azul→rojo (negativo→positivo). Centramos en 0 para lectura rápida.
    vals = pivot.values.astype(float)
    max_abs = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else None
    zmin = -max_abs if max_abs else None
    zmax = max_abs if max_abs else None

    fig = px.imshow(
        pivot,
        aspect="auto",
        title=f"Heatmap: Promedio por trade (Día x Hora) | solo celdas con ≥ {min_trades} trades",
        origin="lower",
        color_continuous_scale="RdBu_r",
        zmin=zmin,
        zmax=zmax,
        zmid=0,
        text_auto=True,
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Azul = promedio negativo, rojo = promedio positivo. Si casi todo queda vacío, baja el mínimo (muestra pequeña).")
    st.caption("Lectura: celdas positivas = mejor promedio/trade. Vacías = poca muestra (no concluyente).")


# ============================================================
# UI
# ============================================================
st.title("📊 WIC_WLF2 Analizador")

uploaded = st.file_uploader(
    "📤 Sube uno o varios archivos .jsonl (meses)",
    type=["jsonl"],
    accept_multiple_files=True
)
if not uploaded:
    st.stop()

all_records, bad_total = [], 0
for uf in uploaded:
    recs, bad = parse_jsonl_bytes(uf.getvalue())
    bad_total += bad
    all_records.extend(recs)

if not all_records:
    st.error("No se pudo leer ningún registro JSON válido.")
    st.stop()

df = pd.DataFrame(all_records)
df = normalize_columns(df)
t = pair_trades(df)

if bad_total > 0:
    st.caption(f"ℹ️ Líneas inválidas ignoradas al parsear: **{bad_total}**")

missing_entry = int((~t["has_entry"]).sum())
if missing_entry > 0:
    st.warning(
        f"⚠️ **{missing_entry} operaciones no tienen ENTRY** en los archivos cargados. "
        "En esas, no se puede saber Compra/Venta ni ORSize/ATR/EWO/DeltaRatio. "
        "Se muestran como: “Sin datos (faltó ENTRY)”."
    )

# Sidebar controls
st.sidebar.subheader("⚙️ Ajustes")
min_trades = st.sidebar.slider("Mínimo trades por grupo (para mirar)", 5, 120, 30, 5)
recommended_trades = st.sidebar.slider("Trades recomendados por grupo (para decidir)", 20, 300, 80, 10)
q_bins = st.sidebar.slider("Número de rangos (cuantiles)", 3, 12, 5, 1)

show_adv_scatter = st.sidebar.checkbox("Mostrar scatter (útil para ver outliers)", value=False)
last_n_scatter = st.sidebar.slider("Scatter: últimos N trades (0=todo)", 0, 3000, 800, 100)

summary = summarize(t)

# ============================================================
# Summary
# ============================================================
st.subheader("✅ Resumen rápido (lo más importante)")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Operaciones", f"{summary['n']}")
c2.metric("Ganadas", f"{summary['wins']}")
c3.metric("Perdidas", f"{summary['losses']}")
c4.metric("% Acierto", f"{summary['win_rate']:.1f}%")
c5.metric("PnL Total", f"{summary['pnl_total']:.0f}")
c6.metric("Profit Factor", f"{summary['pf']:.2f}" if not np.isnan(summary["pf"]) else "N/A")

c7, c8, c9, c10, c11, c12 = st.columns(6)
c7.metric("Promedio por trade (Expectancia)", f"{summary['expectancy']:.1f}")
c8.metric("Max Drawdown", f"{summary['max_dd']:.0f}")
c9.metric("Racha wins seguidos", f"{summary['best_win_streak']}")
c10.metric("Racha losses seguidos", f"{summary['best_loss_streak']}")
c11.metric("Mayor win", f"{summary['max_win']:.1f}")
c12.metric("Mayor loss", f"{summary['max_loss']:.1f}")

with st.expander("📌 Cómo leer estas métricas (simple)"):
    st.write("**Promedio por trade (Expectancia)**: lo que ganas/pierdes en promedio por operación. Si es positivo, bien.")
    st.write("**Profit Factor**: ganancias totales / pérdidas totales. PF > 1.0 indica ventaja. PF > 1.2 suele ser más sólido.")
    st.write("**Drawdown**: la peor caída desde el máximo de tu equity; representa el “dolor máximo” del sistema.")
    st.write("**Rachas**: cuántas operaciones ganadas/perdidas seguidas (útil para guardias diarias y sizing).")

st.markdown("### ⚙️ Avisos según tus ajustes")
total_n = int(summary.get("n", 0))
if total_n < 60:
    st.info("Muestra pequeña (menos de 60 trades). Úsalo para orientar el tuning, pero evita cambiar reglas por 3 trades.")

# Si el usuario pide demasiados rangos o mínimos para la muestra, casi todo quedará vacío.
if q_bins * max(1, min_trades) > max(1, total_n):
    st.warning(
        "⚠️ Con estos ajustes vas a ver muchos paneles vacíos: estás pidiendo demasiados rangos o un mínimo demasiado alto para tu muestra. "
        "Sugerencia con esta cantidad de trades: 3–5 rangos y mínimo 10–20 trades por grupo."
    )
if min_trades > max(10, total_n // 2):
    st.warning("⚠️ Mínimo por grupo muy alto vs tu total de trades. Baja el mínimo si quieres ver más comparaciones (con cuidado).")
if recommended_trades > total_n:
    st.info("Recomendación: tu 'trades recomendados para decidir' es mayor que tu muestra actual, así que todas las conclusiones son provisionales.")

# ============================================================
# Main charts
# ============================================================
st.subheader("📈 Gráficos principales (claros + consejos)")

fig_eq, fig_dd = plot_equity_drawdown(t)
colA, colB = st.columns(2)
with colA:
    st.plotly_chart(fig_eq, use_container_width=True)
with colB:
    st.plotly_chart(fig_dd, use_container_width=True)

st.markdown("### 🧠 Consejos automáticos (Equity / Drawdown)")
for s in equity_recovery_insights(t):
    if "🚨" in s:
        st.error(s)
    elif "⚠️" in s:
        st.warning(s)
    elif "🟡" in s:
        st.info(s)
    else:
        st.success(s)

st.plotly_chart(plot_pnl_hist(t), use_container_width=True)

st.markdown("### 🧠 Consejos automáticos (Distribución de PnL)")
for s in pnl_shape_insights(t):
    if "🚨" in s:
        st.error(s)
    elif "⚠️" in s:
        st.warning(s)
    elif "🟡" in s:
        st.info(s)
    else:
        st.info(s)


# ============================================================
# Ganancia vs Riesgo (RR) + Captura / Devolución
# ============================================================
st.subheader("🎯 Ganancia vs Riesgo (RR) y manejo de la operación")

if "rr" not in t.columns or t["rr"].dropna().empty:
    st.info(
        "No hay datos suficientes para calcular RR. Para esto necesitas en el log (ENTRY o EXIT): "
        "tickSize, pointValue, slTicks y cantidades (qtyTP1/qtyRunner)."
    )
else:
    rr_df = t[t["rr"].notna()].copy()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RR mediana", f"{rr_df['rr'].median():.2f}")
    c2.metric("RR promedio", f"{rr_df['rr'].mean():.2f}")
    c3.metric("% RR ≥ 1", f"{(rr_df['rr'] >= 1).mean()*100:.1f}%")
    c4.metric("% RR ≥ 2", f"{(rr_df['rr'] >= 2).mean()*100:.1f}%")
    c5.metric("Trades con RR", f"{len(rr_df)}")

    # Métricas extra (más fáciles de leer con muestras pequeñas)
    rr_wins = rr_df.loc[rr_df["rr"] > 0, "rr"]
    rr_losses = rr_df.loc[rr_df["rr"] < 0, "rr"]
    small_losses = rr_df[(rr_df["rr"] < 0) & (rr_df["rr"] > -0.5)]

    c6, c7, c8 = st.columns(3)
    c6.metric("RR promedio (ganadores)", f"{rr_wins.mean():.2f}" if not rr_wins.empty else "N/A")
    c7.metric("RR promedio (perdedores)", f"{rr_losses.mean():.2f}" if not rr_losses.empty else "N/A")
    c8.metric("% pérdidas pequeñas (-0.5R a 0)", f"{len(small_losses)/len(rr_df)*100:.1f}%" if len(rr_df) else "N/A")

    fig_rr = px.histogram(rr_df, x="rr", nbins=30, title="Distribución de RR (Ganancia/Riesgo)")
    fig_rr.add_vline(x=0, line_width=1, line_dash="dash")
    fig_rr.add_vline(x=1, line_width=1, line_dash="dash")
    fig_rr.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_rr, use_container_width=True)

    st.markdown("**Cómo leer RR (rápido)**")
    rr_med = rr_df["rr"].median()
    rr_mean = rr_df["rr"].mean()
    if rr_mean > 0 and rr_med < 0:
        st.info("Promedio > 0 y mediana < 0: el resultado depende de pocos trades grandes; la mayoría pierde.")
    elif rr_mean > 0 and rr_med > 0:
        st.info("Promedio > 0 y mediana > 0: la mayoría de trades aporta; comportamiento más estable.")
    elif rr_mean < 0:
        st.info("Promedio < 0: el sistema está perdiendo (tuning o filtros urgentes).")
    st.caption("Con pocas operaciones, confirma con más trades antes de cambiar reglas.")

    st.markdown("### 🧠 Pistas rápidas (RR)")
    if (rr_df["rr"] >= 1).mean() < 0.35:
        st.warning("⚠️ Pocos trades llegan a RR≥1. Revisa: entrar tarde, SL muy grande, o TP demasiado corto.")
    if (rr_df["rr"] <= -1).mean() > 0.10:
        st.warning("🚨 Muchas pérdidas de 1R o más (RR ≤ -1). Revisa: slippage, noticias, stops muy ajustados o entradas sin confirmación.")
    if rr_df["rr"].median() > 0.3 and (rr_df["rr"] >= 1).mean() > 0.45:
        st.success("✅ Estructura de RR saludable (según esta muestra). Aun así: valida con más trades.")

    # Captura / Devolución (solo ganadores)
    if "captura_pct" in t.columns and t["captura_pct"].notna().sum() >= 5:
        wincap = t["captura_pct"].dropna()
        wingb = t["devolucion_pct"].dropna()

        c6, c7, c8 = st.columns(3)
        c6.metric("Captura mediana", f"{wincap.median()*100:.0f}%")
        c7.metric("Devolución mediana", f"{wingb.median()*100:.0f}%")
        c8.metric("Ganadores con datos", f"{len(wincap)}")

        fig_cap = px.histogram(t[t["captura_pct"].notna()], x="captura_pct", nbins=25,
                               title="Captura en ganadores (qué % del máximo flotante te quedas)")
        fig_cap.add_vline(x=0.5, line_width=1, line_dash="dash")
        fig_cap.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_cap, use_container_width=True)

        st.markdown("**Qué significa esto (en simple)**")
        st.write(
            "Solo se calcula en trades **ganadores**. \n"
            "- **Máximo flotante** = lo mejor que llegó a ir tu trade antes de cerrar (maxUnreal). \n"
            "- **Captura** = qué % de ese máximo terminaste cobrando. \n"
            "- **Devolución** = qué % devolviste desde el máximo hasta el cierre."
        )
        st.write(
            "Ejemplo: el trade llegó a **+$500** (máximo flotante) pero cerró en **+$200** → "
            "Captura = 200/500 = **40%** y Devolución = **60%**."
        )

        st.markdown("**🧠 Consejos automáticos (salidas / trailing)**")
        cap_med = wincap.median() if not wincap.empty else np.nan
        gb_med = wingb.median() if not wingb.empty else np.nan

        if cap_med == cap_med and cap_med < 0.35:
            st.warning("⚠️ Captura mediana baja (<35%). Estás cerrando muy lejos del mejor punto: revisa trailing, TP, o reglas de salida temprana.")
        elif cap_med == cap_med and cap_med > 0.60:
            st.success("✅ Captura mediana alta (>60%). Buen manejo de salida (en esta muestra).")
        else:
            st.info("🟡 Captura mediana intermedia. Puede estar bien; valida con más trades.")

        if gb_med == gb_med and gb_med > 0.60:
            st.warning("⚠️ Devolución mediana alta (>60%). Estás devolviendo mucho: prueba trailing más agresivo o TP parcial mejor definido.")
        elif gb_med == gb_med and gb_med < 0.35:
            st.success("✅ Devolución mediana baja (<35%). Sueles proteger ganancias a tiempo.")
        else:
            st.info("🟡 Devolución mediana intermedia. Ajusta solo si ves que afecta RR/expectancia.")

        st.caption("Interpretación rápida: Captura alta suele indicar salidas eficientes; Devolución alta suele indicar trailing/TP tarde o dejar correr sin proteger.")

    # RR por plantilla / tipo de orden
    split_cols = []
    # Evitamos 'trigger' porque suele ser demasiado genérico y confunde a usuarios finales.
    for c in ["template", "orderType", "exitReason", "lado"]:
        if c in rr_df.columns and rr_df[c].notna().sum() > 0:
            split_cols.append(c)

    if split_cols:
        pick = st.selectbox("Comparar RR por:", split_cols, index=0)
        rr_tbl = group_metrics(rr_df, pick, min_trades=max(5, min_trades//2), recommended_trades=recommended_trades)
        st.dataframe(rr_tbl, use_container_width=True)
        st.caption("Tip: usa 'Estado' 🟢/🟡/🔴 para no enamorarte de grupos con 3 trades.")

# ============================================================
# Motivos de salida
# ============================================================
st.subheader("🚪 Motivos de salida (qué está cerrando tus trades)")

colA, colB = st.columns(2)

with colA:
    if "exitReason" in t.columns and t["exitReason"].notna().sum() > 0:
        tbl_exit = group_metrics(t, "exitReason", min_trades=min_trades, recommended_trades=recommended_trades)
        st.markdown("**ExitReason**")
        st.dataframe(tbl_exit, use_container_width=True)

        # Pistas
        if not tbl_exit.empty:
            worst = tbl_exit.sort_values("Promedio por trade").iloc[0]
            if float(worst["Promedio por trade"]) < 0 and int(worst["Trades"]) >= min_trades:
                st.warning(f"⚠️ Peor motivo (por promedio): **{worst['Grupo']}**. Revisa ese flujo/condición.")
    else:
        st.info("No hay exitReason en los logs.")

with colB:
    if "forcedCloseReason" in t.columns and t["forcedCloseReason"].notna().sum() > 0:
        tbl_fc = group_metrics(t, "forcedCloseReason", min_trades=max(5, min_trades//2), recommended_trades=recommended_trades)
        st.markdown("**ForcedCloseReason**")
        st.dataframe(tbl_fc, use_container_width=True)

        if not tbl_fc.empty:
            hot = tbl_fc.sort_values("Trades", ascending=False).iloc[0]
            if int(hot["Trades"]) >= min_trades and float(hot["Promedio por trade"]) < 0:
                st.warning(f"🚨 Forced close frecuente y negativo: **{hot['Grupo']}**. Esto suele ser 'regla' o 'protección' mal calibrada.")
    else:
        st.info("No hay forcedCloseReason en los logs.")

st.caption("OJO: el motivo de salida es diagnóstico, no filtro mágico. Úsalo para detectar patrones (stops, daily halt, cierre forzado, etc.).")


# ============================================================
# Long vs Short
# ============================================================
st.subheader("🧭 Compra vs Venta (solo donde hay ENTRY)")

col1, col2, col3 = st.columns(3)
col1.metric("Compras (Long)", int((t["lado"] == "Compra (Long)").sum()))
col2.metric("Ventas (Short)", int((t["lado"] == "Venta (Short)").sum()))
col3.metric("Sin datos (faltó ENTRY)", int((t["lado"].str.startswith("Sin datos")).sum()))

known = t[t["lado"].isin(["Compra (Long)", "Venta (Short)"])].copy()
if known.empty:
    st.info("No hay suficientes trades con ENTRY para separar Compra/Venta.")
else:
    side_tbl = group_metrics(known, "lado", min_trades=max(5, min_trades // 2))
    st.dataframe(side_tbl, use_container_width=True)
    advice_from_table(side_tbl, title="Compra/Venta", min_trades=max(5, min_trades // 2))

# ============================================================
# Tuning factors
# ============================================================
st.subheader("🛠️ Ajuste de filtros (tunear settings con datos reales)")

if known.empty:
    st.info("Estos análisis necesitan ENTRY (para tener ORSize/ATR/EWO/DeltaRatio por trade).")
else:
    df_known = known.copy()
    df_scatter = df_known.sort_values("exit_time")
    if last_n_scatter and last_n_scatter > 0:
        df_scatter = df_scatter.tail(last_n_scatter)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["OR Size", "ATR", "EWO", "Balance C/V", "Presión y Actividad"])

    with tab1:
        with st.expander("📌 OR Size: qué significa y cómo usarlo"):
            st.write("OR Size = tamaño del rango inicial (Opening Range).")
            st.write("- OR grande → mercado más 'movido'. Si tu PF cae aquí, suele ser por whipsaws y entradas tardías.")
            st.write("- OR pequeño → mercado más 'apretado'. Si tu PF cae aquí, suele ser por falta de recorrido.")
            st.write("Consejo: busca rangos con **PF>1**, **promedio > 0** y muestra 🟢 antes de convertirlo en filtro.")
        if "orSize" in df_known.columns and df_known["orSize"].notna().sum() > 30:
            plot_factor_bins(df_known, "orSize", q_bins, min_trades, recommended_trades, "OR Size (tamaño del rango)", show_adv_scatter, df_scatter)
        else:
            st.info("No hay suficientes valores de OR Size en los logs.")

    with tab2:
        with st.expander("📌 ATR: qué significa y cómo usarlo"):
            st.write("ATR = volatilidad (cuánto se mueve el precio en promedio).")
            st.write("- ATR alto → movimientos amplios: necesitas stops/targets coherentes o te saca el ruido.")
            st.write("- ATR bajo → poco recorrido: si tu TP es fijo, puede que no llegue.")
            st.write("Consejo: si tu estrategia tiene ATR engine, aquí verás si te está ayudando o empeorando.")
        if "atr" in df_known.columns and df_known["atr"].notna().sum() > 30:
            plot_factor_bins(df_known, "atr", q_bins, min_trades, recommended_trades, "ATR (volatilidad)", show_adv_scatter, df_scatter)
        else:
            st.info("No hay suficientes valores de ATR en los logs.")

    with tab3:
        with st.expander("📌 EWO: qué significa y cómo usarlo"):
            st.write("EWO (aquí usamos |EWO|) = fuerza de tendencia. Más alto suele implicar tendencia más marcada.")
            st.write("- |EWO| alto → tendencia: suele favorecer continuaciones/breakouts.")
            st.write("- |EWO| bajo → chop/rango: suele castigar entradas por impulso.")
            st.write("Consejo: si ves PF<1 en |EWO| bajo, es un buen candidato a filtro de 'no-trade'.")
        if "ewo" in df_known.columns and df_known["ewo"].notna().sum() > 30:
            df_known2 = df_known.copy()
            df_known2["ewo_abs"] = df_known2["ewo"].abs()

            df_scatter2 = df_scatter.copy()
            df_scatter2["ewo_abs"] = df_scatter2["ewo"].abs()

            plot_factor_bins(df_known2, "ewo_abs", q_bins, min_trades, recommended_trades, "EWO (fuerza de tendencia)", show_adv_scatter, df_scatter2)
        else:
            st.info("No hay suficientes valores de EWO en los logs.")

    with tab4:
        with st.expander("📌 Balance comprador-vendedor: qué significa y cómo usarlo"):
            st.write("Balance C/V = delta/vol (intensidad neta de compras vs ventas).")
            st.write("- Valores altos (en magnitud) → desequilibrio fuerte (posible impulso).")
            st.write("- Cerca de 0 → poca ventaja de flujo (más fácil que el precio se 'devuelva').")
            st.write("Consejo: úsalo para evitar entradas cuando no hay participación real.")
        if "deltaRatio" in df_known.columns and df_known["deltaRatio"].notna().sum() > 30:
            plot_factor_bins(df_known, "deltaRatio", q_bins, min_trades, recommended_trades, "Balance comprador-vendedor (delta/vol)", show_adv_scatter, df_scatter)
        else:
            st.info("No hay suficientes valores de Balance C/V en los logs.")

    with tab5:
        st.markdown("### Qué es esto (en simple)")
        st.write(
            "Usamos los arrays pre-entrada (delta/volumen y OHLC de los últimos N segundos) para medir "
            "**actividad** y **presión** antes de entrar. Esto ayuda a detectar entradas 'con gasolina' "
            "vs entradas 'sin apoyo' o con **absorción**."
        )
        st.caption(
            "⚠️ OJO: con 40 trades esto es exploratorio. Úsalo para generar hipótesis y valida con más muestra."
        )

        needed_any = any(c in df_known.columns for c in ["pre_absorcion", "pre_actividad_rel", "pre_presion_por_vol", "pre_presion_sum"])
        if not needed_any:
            st.info("No veo arrays pre-entrada en tus logs (deltaLastN/volLastN/OHLC). Si están, revisa que se estén logueando en ENTRY.")
        else:
            # Ajuste de bins más conservador cuando la muestra es pequeña
            q_small = min(q_bins, 6)

            colA, colB = st.columns(2)
            with colA:
                if "pre_absorcion" in df_known.columns and df_known["pre_absorcion"].notna().sum() > 20:
                    st.markdown("#### 1) Absorción (presión fuerte, avance pequeño)")
                    plot_factor_bins(df_known, "pre_absorcion", q_small, min_trades, recommended_trades,
                                     "Absorción (presión fuerte + avance pequeño)", show_adv_scatter, df_scatter)
                else:
                    st.info("No hay suficientes valores para 'Absorción'.")

            with colB:
                if "pre_actividad_rel" in df_known.columns and df_known["pre_actividad_rel"].notna().sum() > 20:
                    st.markdown("#### 2) Actividad (volumen relativo antes de entrar)")
                    plot_factor_bins(df_known, "pre_actividad_rel", q_small, min_trades, recommended_trades,
                                     "Actividad relativa (volumen hasta entrada vs mediana reciente)", show_adv_scatter, df_scatter)
                else:
                    st.info("No hay suficientes valores para 'Actividad relativa'.")

            st.markdown("#### 3) Señales sin apoyo (precio avanza, presión va en contra)")
            if "pre_sin_apoyo" in df_known.columns:
                tmp = df_known.copy()
                tmp["Sin apoyo"] = np.where(tmp["pre_sin_apoyo"] == True, "Sí", "No")
                tbl = group_metrics(tmp, "Sin apoyo", min_trades=min_trades, recommended_trades=recommended_trades)
                st.dataframe(tbl, use_container_width=True)

                # Consejos rápidos
                if not tbl.empty and "Grupo" in tbl.columns:
                    row_si = tbl[tbl["Grupo"] == "Sí"]
                    row_no = tbl[tbl["Grupo"] == "No"]
                    if not row_si.empty and not row_no.empty:
                        exp_si = float(row_si.iloc[0]["Promedio por trade"])
                        exp_no = float(row_no.iloc[0]["Promedio por trade"])
                        if exp_si < exp_no:
                            st.warning("⚠️ Tus trades marcados como 'Sin apoyo = Sí' rinden peor en promedio. Buen candidato para FILTRAR.")
                        else:
                            st.info("🟡 Por ahora 'Sin apoyo' no empeora rendimiento, pero confirma con más muestra.")
            else:
                st.info("No se detecta la bandera 'pre_sin_apoyo' (requiere arrays + dir).")

            st.markdown("### Pistas prácticas (tuning)")
            st.write("✅ Si **Actividad alta** + buen PF → suele indicar entradas con participación real.")
            st.write("🚫 Si **Absorción muy alta** + PF<1 → suele ser 'lucha' / absorción antes del giro (candidato a evitar).")
            st.write("⚠️ Si 'Sin apoyo = Sí' aparece mucho en pérdidas → evita perseguir precio cuando el delta no acompaña.")


# ============================================================
# Hours
# ============================================================
st.subheader("⏰ Horarios (justo y confiable)")

hour_tbl = plot_hour_analysis(t, min_trades=min_trades)
plot_heatmap_weekday_hour(t, min_trades=min_trades)

st.markdown("### 🧠 Consejos automáticos (Horarios)")
if hour_tbl is None or hour_tbl.empty:
    st.info("No hay datos suficientes por hora con el mínimo configurado.")
else:
    best = hour_tbl.iloc[0]
    worst = hour_tbl.iloc[-1]

    st.info(
        f"🏆 Hora recomendada: **{best['Grupo']}** | Trades={int(best['Trades'])} | "
        f"{traffic_pf(best['Profit Factor'])} | {traffic_exp(best['Promedio por trade'])}"
    )
    if best["Trades"] < min_trades * 2:
        st.warning("⚠️ La mejor hora aún tiene muestra pequeña. Confirma con más logs.")

    if not np.isnan(worst["Profit Factor"]) and worst["Profit Factor"] < 1.0:
        st.warning(f"🚫 Hora candidata a evitar: **{worst['Grupo']}** (PF < 1 y promedio bajo).")

    st.caption("Regla de oro: prefiere horas con buen Score ponderado + buen PF + buena muestra, no solo winrate.")

# ============================================================
# Resumen final (muy user-friendly)
# ============================================================
st.subheader("🧾 Resumen final y recomendaciones")

st.write(
    "Este bloque resume lo más accionable. No es una verdad absoluta: con muestra pequeña, úsalo para orientar, no para 'sobre-optimizar'."
)

# 1) Salud general
pf_val = summary.get("pf", np.nan)
exp_val = summary.get("expectancy", np.nan)
if pf_val == pf_val and pf_val < 1.0:
    st.error("🚨 Salud general: Profit Factor < 1 (pierde). Prioridad: filtrar condiciones malas antes de ajustar targets.")
elif exp_val == exp_val and exp_val < 0:
    st.warning("⚠️ Salud general: promedio por trade < 0. Revisa filtros (horario/volatilidad/tendencia) y disciplina de salida.")
else:
    st.success("✅ Salud general: no se ve rojo inmediato (según esta muestra). Ahora toca mejorar consistencia.")

# 2) RR y estructura
if "rr" in t.columns and t["rr"].notna().any():
    _rr = t[t["rr"].notna()]["rr"].astype(float)
    rr_median = float(_rr.median())
    rr_mean = float(_rr.mean())
    rr_ge2 = float((_rr >= 2).mean() * 100)
    rr_le_1 = float((_rr <= -1).mean() * 100)

    st.markdown("**Estructura RR**")
    st.write(f"- RR mediana: {rr_median:.2f} | RR promedio: {rr_mean:.2f} | %RR≥2: {rr_ge2:.1f}% | %RR≤-1: {rr_le_1:.1f}%")
    if rr_mean > 0 and rr_median < 0:
        st.info("Promedio > 0 y mediana < 0: dependes de pocos ganadores grandes. Enfócate en reducir stop-outs feos sin matar tus runners.")
    if rr_le_1 > 15:
        st.warning("⚠️ %RR≤-1 alto: estás tomando muchas pérdidas completas. Buen objetivo: mejorar confirmación/evitar chop/horas malas.")

# 3) Manejo de ganadores (captura/devolución)
if "captura_pct" in t.columns and t["captura_pct"].notna().sum() >= 5:
    cap_med = float(t["captura_pct"].dropna().median())
    gb_med = float(t["devolucion_pct"].dropna().median()) if "devolucion_pct" in t.columns and t["devolucion_pct"].notna().any() else np.nan
    st.markdown("**Manejo de ganadores**")
    st.write(f"- Captura mediana: {cap_med*100:.0f}%" + (f" | Devolución mediana: {gb_med*100:.0f}%" if gb_med == gb_med else ""))
    if cap_med < 0.35:
        st.warning("⚠️ Captura baja: estás dejando mucho en la mesa. Ajusta trailing/TP parcial o reglas de salida temprana.")
    if gb_med == gb_med and gb_med > 0.60:
        st.warning("⚠️ Devolución alta: el trade va bien, pero no proteges a tiempo. Prueba trailing más agresivo cuando ya estés en +1R.")

# 4) Motivos de salida (top problema)
if "exitReason" in t.columns and t["exitReason"].notna().any():
    _tbl = group_metrics(t, "exitReason", min_trades=max(5, min_trades//2), recommended_trades=recommended_trades)
    if not _tbl.empty:
        worst = _tbl.sort_values("Promedio por trade").iloc[0]
        if float(worst["Promedio por trade"]) < 0:
            st.markdown("**Motivo de salida a vigilar**")
            st.write(f"- {worst['Grupo']}: promedio {float(worst['Promedio por trade']):.1f} con {int(worst['Trades'])} trades")

# 5) Horarios (si hay)
if hour_tbl is not None and not hour_tbl.empty:
    best = hour_tbl.iloc[0]
    st.markdown("**Horario con mejor Score (ponderado)**")
    st.write(f"- {best['Grupo']} | Trades={int(best['Trades'])} | PF={float(best['Profit Factor']):.2f} | Promedio={float(best['Promedio por trade']):.1f}")
    if int(best["Trades"]) < min_trades * 2:
        st.info("Nota: el mejor horario aún tiene poca muestra. Confirma con más meses antes de convertirlo en regla.")

st.markdown("**Siguientes pasos recomendados (orden)**")
st.write("1) Primero elimina lo rojo: horarios peores + condiciones con PF<1 (muestra 🟢).")
st.write("2) Luego ajusta manejo: reduce devoluciones grandes y evita stop-outs completos recurrentes.")
st.write("3) Al final optimiza targets/trailing: no mates los winners grandes si tu edge depende de ellos.")

# ============================================================
# Trades table
# ============================================================
with st.expander("📄 Tabla de trades (una fila por atmId)", expanded=False):
    cols_show = [c for c in [
        "exit_time", "entry_time", "lado", "outcome", "tradeRealized",
        "maxUnreal", "minUnreal", "exitReason", "forcedCloseReason",
        "orSize", "ewo", "atr", "deltaRatio", "atrSlMult", "tp1R", "tp2R",
        "duration_sec"
    ] if c in t.columns]
    st.dataframe(t[cols_show].sort_values("exit_time", ascending=False), use_container_width=True)

st.caption(
    "Tip: para eliminar “Sin datos (faltó ENTRY)”, asegúrate de cargar meses que contengan los ENTRY y EXIT juntos. "
    "Si quieres 100% robustez, añade `dir` también en el EXIT (en NinjaScript) o guarda logs en bloques por trade."
)
