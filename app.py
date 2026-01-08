import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px

st.set_page_config(page_title="WIC_WLF2 Analizador", layout="wide")

def _best_group_for_categorical(df: pd.DataFrame, col: str, min_trades: int):
    if col not in df.columns:
        return None
    rows = []
    for g, sub in df.groupby(col):
        n = int(len(sub))
        if n < min_trades:
            continue
        exp = float(sub["tradeRealized"].mean())
        pf = profit_factor(sub)
        score = exp * np.log1p(n)
        rows.append((g, n, exp, pf, score))
    if not rows:
        return None
    rows.sort(key=lambda x: (x[4], x[1]), reverse=True)
    g, n, exp, pf, score = rows[0]
    return {"grupo": g, "n": n, "exp": exp, "pf": pf, "score": score}


def _best_range_for_numeric(df: pd.DataFrame, col: str, q: int, min_trades: int):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() < max(15, min_trades * 2):
        return None
    try:
        cuts = pd.qcut(s, q=q, duplicates="drop")
    except Exception:
        return None

    tmp = df.copy()
    tmp["_r"] = cuts.astype(str)

    best = None
    for g, sub in tmp.groupby("_r"):
        n = int(len(sub))
        if n < min_trades:
            continue
        exp = float(sub["tradeRealized"].mean())
        pf = profit_factor(sub)
        score = exp * np.log1p(n)
        cand = {"rango": str(g), "n": n, "exp": exp, "pf": pf, "score": score}
        if (best is None) or (cand["score"] > best["score"]):
            best = cand
    if best is None:
        return None

    # Intentar extraer límites numéricos del rango "(a, b]"
    lo = hi = None
    rg = best["rango"].strip()
    if ("," in rg) and (rg[0] in "([") and (rg[-1] in "])"):
        inner = rg[1:-1]
        parts = inner.split(",")
        if len(parts) == 2:
            try:
                lo = float(parts[0])
                hi = float(parts[1])
            except Exception:
                lo = hi = None
    best["lo"] = lo
    best["hi"] = hi
    return best


def recommend_settings_block(known_df: pd.DataFrame, min_trades: int, recommended_trades: int):
    """
    Resume recomendaciones 'tipo NinjaTrader' basadas en lo que rinde mejor en ESTA muestra.
    No inventa campos: solo usa lo que existe en el JSON.
    """
    if known_df is None or known_df.empty:
        st.info("No hay suficientes trades con ENTRY para sugerir ajustes.")
        return

    n_total = int(len(known_df))
    st.subheader("🎯 Ajustes sugeridos (según esta muestra)")
    st.caption("Esto NO es una optimización científica; es una guía rápida. Con poca muestra, úsalo como hipótesis.")

    if n_total < recommended_trades:
        st.warning(f"Muestra pequeña: {n_total} trades con ENTRY. Recomendado ≥ {recommended_trades} para decisiones fuertes.")

    # 1) Entry: StopMarket vs Limit
    if "orderType" in known_df.columns:
        tmp = known_df.copy()
        ot = tmp["orderType"].astype(str).str.lower()
        tmp["EntryType"] = np.where(ot.str.contains("stop"), "StopMarket", np.where(ot.str.contains("limit"), "Limit", tmp["orderType"].astype(str)))
        best_entry = _best_group_for_categorical(tmp, "EntryType", min_trades=min_trades)
    else:
        best_entry = None

    # 2) UseAtrEngine (bool)
    best_atr_mode = _best_group_for_categorical(known_df, "useAtrEngine", min_trades=min_trades)

    # 3) Rangos numéricos clave (si existen)
    q_small = 3  # con 40 trades: rangos pocos, más estables
    best_or = _best_range_for_numeric(known_df, "orSize", q=q_small, min_trades=min_trades)
    best_ewo = None
    if "ewo" in known_df.columns:
        tmp = known_df.copy()
        tmp["ewoAbs"] = pd.to_numeric(tmp["ewo"], errors="coerce").abs()
        best_ewo = _best_range_for_numeric(tmp, "ewoAbs", q=q_small, min_trades=min_trades)

    best_atr = _best_range_for_numeric(known_df, "atr", q=q_small, min_trades=min_trades)
    best_atr_sl = _best_range_for_numeric(known_df, "atrSlMult", q=q_small, min_trades=min_trades)
    best_tp1 = _best_range_for_numeric(known_df, "tp1R", q=q_small, min_trades=min_trades)
    best_tp2 = _best_range_for_numeric(known_df, "tp2R", q=q_small, min_trades=min_trades)
    best_ts = _best_range_for_numeric(known_df, "tsBehindTP1Atr", q=q_small, min_trades=min_trades)
    best_step = _best_range_for_numeric(known_df, "trailStepTicks", q=q_small, min_trades=min_trades)

    # 4) Render: salida tipo "parámetro -> sugerencia"
    lines = []

    def add_line(name, suggestion, note=None):
        s = f"- **{name}**: {suggestion}"
        if note:
            s += f"  \n  {note}"
        lines.append(s)

    if best_entry:
        use_stop = "True" if str(best_entry["grupo"]).lower() == "stopmarket" else "False"
        add_line("UseStopMarketEntry", use_stop, f"(mejor grupo: {best_entry['grupo']} | n={best_entry['n']} | PF={best_entry['pf']:.2f} | prom={best_entry['exp']:.0f})")

    if best_atr_mode:
        add_line("UseAtrEngine", str(best_atr_mode["grupo"]), f"(n={best_atr_mode['n']} | PF={best_atr_mode['pf']:.2f} | prom={best_atr_mode['exp']:.0f})")

    if best_or and best_or.get("lo") is not None and best_or.get("hi") is not None:
        add_line("MinORSize / MaxORSize", f"{best_or['lo']:.2f}  →  {best_or['hi']:.2f}",
                 f"(mejor rango ORSize | n={best_or['n']} | PF={best_or['pf']:.2f} | prom={best_or['exp']:.0f})")

    if best_ewo and best_ewo.get("lo") is not None:
        add_line("MinEWOMagnitude", f"≈ {best_ewo['lo']:.3f} (o mayor)",
                 f"(basado en |EWO| | n={best_ewo['n']} | PF={best_ewo['pf']:.2f} | prom={best_ewo['exp']:.0f})")

    if best_atr_sl and best_atr_sl.get("lo") is not None and best_atr_sl.get("hi") is not None:
        mid = (best_atr_sl["lo"] + best_atr_sl["hi"]) / 2
        add_line("AtrSlMult", f"≈ {mid:.2f} (rango {best_atr_sl['lo']:.2f}–{best_atr_sl['hi']:.2f})",
                 f"(n={best_atr_sl['n']} | PF={best_atr_sl['pf']:.2f} | prom={best_atr_sl['exp']:.0f})")

    if best_tp1 and best_tp1.get("lo") is not None and best_tp1.get("hi") is not None:
        mid = (best_tp1["lo"] + best_tp1["hi"]) / 2
        add_line("TP1_RMult", f"≈ {mid:.2f} (rango {best_tp1['lo']:.2f}–{best_tp1['hi']:.2f})",
                 f"(n={best_tp1['n']} | PF={best_tp1['pf']:.2f} | prom={best_tp1['exp']:.0f})")

    if best_tp2 and best_tp2.get("lo") is not None and best_tp2.get("hi") is not None:
        mid = (best_tp2["lo"] + best_tp2["hi"]) / 2
        add_line("TP2_RMult", f"≈ {mid:.2f} (rango {best_tp2['lo']:.2f}–{best_tp2['hi']:.2f})",
                 f"(n={best_tp2['n']} | PF={best_tp2['pf']:.2f} | prom={best_tp2['exp']:.0f})")

    if best_ts and best_ts.get("lo") is not None and best_ts.get("hi") is not None:
        mid = (best_ts["lo"] + best_ts["hi"]) / 2
        add_line("TSBehindTP1_ATR", f"≈ {mid:.2f} (rango {best_ts['lo']:.2f}–{best_ts['hi']:.2f})",
                 f"(n={best_ts['n']} | PF={best_ts['pf']:.2f} | prom={best_ts['exp']:.0f})")

    if best_step and best_step.get("lo") is not None and best_step.get("hi") is not None:
        # ticks: mejor entero
        mid = int(round((best_step["lo"] + best_step["hi"]) / 2))
        add_line("TrailStepTicks", f"≈ {mid} (rango {best_step['lo']:.0f}–{best_step['hi']:.0f})",
                 f"(n={best_step['n']} | PF={best_step['pf']:.2f} | prom={best_step['exp']:.0f})")

    if not lines:
        st.info("No hay suficientes campos/muestra para sugerir ajustes automáticos.")
    else:
        st.markdown("\n".join(lines))

    st.caption("⚠️ Consejo: aplica 1 cambio a la vez y vuelve a medir. Evita optimizar 5 parámetros con 40 trades.")


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
def profit_factor(trades) -> float:
    """Profit Factor = suma de ganancias / suma absoluta de pérdidas.

    Acepta:
      - DataFrame con columna 'tradeRealized'
      - Serie de PnL (tradeRealized)
    """
    if trades is None:
        return np.nan

    # Permite pasar directamente la Serie tradeRealized
    if isinstance(trades, pd.Series):
        s = trades.dropna()
    else:
        if not isinstance(trades, pd.DataFrame) or "tradeRealized" not in trades.columns:
            return np.nan
        s = trades["tradeRealized"].dropna()

    wins = float(s[s > 0].sum())
    losses = float(s[s < 0].sum())  # negativo

    loss_abs = abs(losses)
    if loss_abs <= 0:
        # Sin pérdidas: PF infinito si hay ganancias, sino NaN
        return float("inf") if wins > 0 else np.nan

    return wins / loss_abs


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
    """Histograma de PnL realizado (dinero). tradeRealized está en la moneda de la cuenta (normalmente $)."""
    tmp = t.copy()
    tr = pd.to_numeric(tmp.get("tradeRealized"), errors="coerce")
    tmp["Resultado"] = np.where(tr >= 0, "Ganancia", "Pérdida")

    fig = px.histogram(
        tmp,
        x="tradeRealized",
        color="Resultado",
        nbins=40,
        barmode="overlay",
        title="Distribución de PnL ($) por operación",
        labels={"tradeRealized": "PnL ($)"},
    )
    fig.update_traces(opacity=0.75)
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
    """
    Panel por rangos (cuantiles) para un factor (OR/ATR/EWO/etc).
    - Colores consistentes: verde = mejor, rojo = peor.
    - Sin 'bins' en UI: todo se llama 'rangos'.
    - Etiquetas humanas para rangos (ej: 60–94).
    """
    def nice_range_label(s: str) -> str:
        if not isinstance(s, str):
            return str(s)
        s2 = s.strip()
        # Formatos típicos: "(60.0, 94.0]" o "[14.5, 60.0)"
        if ("," in s2) and (s2[0] in "([") and (s2[-1] in "])"):
            inner = s2[1:-1]
            parts = inner.split(",")
            if len(parts) == 2:
                try:
                    a = float(parts[0])
                    b = float(parts[1])
                    # Redondeo “humano”
                    if abs(a) >= 100 or abs(b) >= 100:
                        return f"{a:.0f}–{b:.0f}"
                    if abs(a) >= 10 or abs(b) >= 10:
                        return f"{a:.1f}–{b:.1f}"
                    return f"{a:.2f}–{b:.2f}"
                except Exception:
                    return s2
        return s2

    bins = make_bins_quantiles(df_known, col, q)
    if bins is None:
        st.info(f"No hay suficiente data para crear rangos en **{title}**.")
        return

    tmp = df_known.copy()
    tmp["_range"] = bins.astype(str)

    tbl = group_metrics(tmp, "_range", min_trades=min_trades, recommended_trades=recommended_trades)
    if tbl.empty:
        st.info(f"En **{title}** no hay data para agrupar.")
        return

    # Etiquetas humanas (más fáciles de leer)
    _orig = tbl["Grupo"].astype(str).tolist()
    _nice = [nice_range_label(x) for x in _orig]

    def _parse_left(rg: str):
        rg = str(rg).strip()
        if ("," in rg) and (rg[0] in "([") and (rg[-1] in "])"):
            inner = rg[1:-1]
            parts = inner.split(",")
            if len(parts) == 2:
                try:
                    return float(parts[0])
                except Exception:
                    return np.nan
        return np.nan

    lefts = [(_parse_left(o), i) for i, o in enumerate(_orig)]
    order = [i for _, i in sorted(lefts, key=lambda x: (np.nan_to_num(x[0], nan=1e18), x[1]))]

    k = len(_nice)
    if k == 3:
        names = ["Bajo", "Medio", "Alto"]
    elif k == 4:
        names = ["Bajo", "Medio-bajo", "Medio-alto", "Alto"]
    elif k == 5:
        names = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
    else:
        names = [f"R{i+1}" for i in range(k)]

    label_map = {}
    for rank, idx in enumerate(order):
        base = names[rank] if rank < len(names) else f"R{rank+1}"
        label_map[_orig[idx]] = f"{base} ({_nice[idx]})"

    tbl["Grupo"] = tbl["Grupo"].astype(str).map(label_map).fillna(tbl["Grupo"].astype(str))

    # Leyenda de muestra
    n_ok = int((tbl["Estado"] == "🟢 Suficiente").sum())
    n_small = int((tbl["Estado"] == "🟡 Muestra pequeña").sum())
    n_bad = int((tbl["Estado"] == "🔴 No concluyente").sum())
    st.caption(
        f"Muestra por rango: 🟢{n_ok} 🟡{n_small} 🔴{n_bad}  | "
        f"Mínimo para mirar: {min_trades}  | Recomendado para decidir: {recommended_trades}"
    )

    # Para gráficos: solo rangos con muestra mínima
    tbl_chart = tbl[tbl["Trades"] >= min_trades].copy()

    if not tbl_chart.empty:
        # Promedio por trade: verde = mejor, rojo = peor. Centramos en 0 (negativos quedan rojos).
        fig_exp = px.bar(
            tbl_chart,
            x="Grupo",
            y="Promedio por trade",
            color="Promedio por trade",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title=f"{title} → Promedio por trade (rangos)",
            text="Promedio por trade",
        )
        fig_exp.update_traces(texttemplate="%{text:.0f}", textposition="inside")
        fig_exp.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=50, b=10),
            coloraxis_colorbar_title="Promedio",
        )
        fig_exp.add_hline(y=0, line_width=1, line_dash="dash")

        # Profit Factor: verde = alto (mejor), rojo = bajo (peor). Punto neutro en 1.0.
        fig_pf = px.bar(
            tbl_chart,
            x="Grupo",
            y="Profit Factor",
            color="Profit Factor",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=1.0,
            title=f"{title} → Profit Factor (rangos)",
            text="Profit Factor",
        )
        fig_pf.update_traces(texttemplate="%{text:.2f}", textposition="inside")
        fig_pf.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=50, b=10),
            coloraxis_colorbar_title="PF",
        )
        fig_pf.add_hline(y=1.0, line_width=1, line_dash="dash")

        st.plotly_chart(fig_exp, use_container_width=True)
        st.plotly_chart(fig_pf, use_container_width=True)

        st.caption("Profit Factor (PF) = Ganancias totales / Pérdidas totales. PF>1 indica que el conjunto gana. "
                   "PF>1.5 suele ser sólido si la muestra es suficiente (🟢).")

        advice_from_table(tbl_chart, title, min_trades=min_trades)
    else:
        st.info("⚠️ Con el mínimo actual, todos los rangos quedan con poca muestra. Úsalo como idea, no como regla.")

    st.dataframe(tbl, use_container_width=True)

    if show_scatter:
        st.markdown("**Scatter (solo para ver dispersión / outliers)**")
        if col in scatter_df.columns and scatter_df[col].notna().sum() >= max(20, min_trades):
            plot_scatter_advanced(scatter_df, col, title=f"{title} → Scatter")
        else:
            st.info("No hay suficiente data para scatter en este factor.")

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

    # Compatibilidad: algunas versiones de Plotly no soportan text_auto en imshow.
    try:
        fig = px.imshow(
            pivot,
            aspect="auto",
            title=f"Heatmap: Promedio por trade (Día x Hora) | solo celdas con ≥ {min_trades} trades",
            origin="lower",
            color_continuous_scale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
            color_continuous_midpoint=0,
            text_auto=True,
        )
    except TypeError:
        fig = px.imshow(
            pivot,
            aspect="auto",
            title=f"Heatmap: Promedio por trade (Día x Hora) | solo celdas con ≥ {min_trades} trades",
            origin="lower",
            color_continuous_scale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
            color_continuous_midpoint=0,
        )
        # Añadimos etiquetas manuales (valores)
        for yi, day in enumerate(pivot.index.astype(str).tolist()):
            for xi, hour in enumerate(pivot.columns.tolist()):
                v = pivot.loc[pivot.index[yi], hour]
                if pd.notna(v):
                    fig.add_annotation(
                        x=hour,
                        y=day,
                        text=f"{v:.0f}",
                        showarrow=False,
                        font=dict(size=11, color="black"),
                    )

    fig.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Azul = promedio negativo, rojo = promedio positivo. Si casi todo queda vacío, baja el mínimo (muestra pequeña).")

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
st.caption("PnL ($) = dinero realizado por trade (tradeRealized). Si quieres verlo en ticks/points, se puede derivar con tickSize/pointValue, pero aquí mostramos la realidad de la cuenta.")

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

    rr_tmp = rr_df.copy()
    rr_tmp["Resultado"] = np.where(rr_tmp["rr"] >= 0, "Ganancia", "Pérdida")
    fig_rr = px.histogram(
        rr_tmp,
        x="rr",
        color="Resultado",
        nbins=30,
        barmode="overlay",
        title="Distribución de RR (Ganancia/Riesgo)",
        labels={"rr": "RR"},
    )
    fig_rr.update_traces(opacity=0.75)
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
    pct_rr1 = (rr_df["rr"] >= 1).mean()
    pct_stop = (rr_df["rr"] <= -1).mean()
    pct_small_loss = ((rr_df["rr"] < 0) & (rr_df["rr"] > -0.5)).mean()

    if pct_rr1 < 0.35:
        st.warning("⚠️ Pocos trades llegan a RR≥1. Revisa: entrar tarde, SL muy grande, o TP demasiado corto.")
    if pct_stop > 0.10:
        st.warning("🚨 Muchas pérdidas de 1R o más (RR ≤ -1). Revisa: slippage, noticias, stops muy ajustados o entradas sin confirmación.")
    if pct_small_loss < 0.05 and pct_stop > 0.15:
        st.info("🟡 Casi no hay pérdidas pequeñas: cuando pierdes, sueles perder ~1R completo. Eso suele indicar 'stop-out' frecuente (poco margen para salir antes). Buen candidato: mejorar filtros de entrada/horario o invalidación temprana.")

    if rr_df["rr"].median() > 0.3 and pct_rr1 > 0.45:
        st.success("✅ Estructura de RR saludable (según esta muestra). Aun así: valida con más trades.")

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

        # Mostrar en % (0–100). Si ves valores >100%, suele ser inconsistencia de log (maxUnreal no capturó el pico real).
        raw_cap = t["captura_pct"].dropna()
        n_weird = int((raw_cap > 1.05).sum())
        if n_weird > 0:
            st.warning("⚠️ Veo algunos valores de 'captura' > 100%. Eso suele pasar cuando maxUnreal no capturó el pico real (o por parciales). "
                       "Para no confundir, el gráfico se limita a 0–100%.")

        cap_plot = (raw_cap.clip(0, 1) * 100.0)
        cap_df = pd.DataFrame({"Captura (%)": cap_plot})

        fig_cap = px.histogram(
            cap_df,
            x="Captura (%)",
            nbins=20,
            title="Qué tan cerca del pico cierras (solo ganadores)",
        )
        fig_cap.add_vline(x=50, line_width=1, line_dash="dash")
        fig_cap.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_cap, use_container_width=True)

        st.markdown("**Qué significa esto (en simple)**")
        st.write(
            "Solo se calcula en trades **ganadores**. "
            "La idea es responder: **¿cerré cerca del mejor punto, o dejé devolver mucho?**"
        )
        st.write(
            "- **Pico del trade** = el mejor PnL flotante que tuvo antes de cerrar (*maxUnreal*).\n"
            "- **Te quedas (captura)** = qué % de ese pico terminaste cobrando al cerrar.\n"
            "- **Devolviste (devolución)** = qué % retrocedió desde el pico hasta el cierre."
        )
        st.write(
            "Ejemplo: el trade llegó a **+$500** (pico) y cerró en **+$475** → "
            "Te quedas ≈ **95%** y devolviste ≈ **5%**."
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
# ============================================================
# Entradas por día y orden del trade (tuning realista)
# ============================================================
st.subheader("📅 Entradas por día y orden del trade")
st.caption("Aquí miramos **cuántas entradas haces por día** y si el **trade #1 / #2 / #3...** suele ser mejor o peor. "
           "Sirve para decisiones realistas como **MaxSetupsPerSession**, evitar el 'revenge trade' y cortar días malos.")

if "entry_time" in t.columns and t["entry_time"].notna().any():
    te = t.copy()
    te = te[te["entry_time"].notna()].copy()
    te = te.sort_values("entry_time").reset_index(drop=True)
    te["entry_date"] = te["entry_time"].dt.date
    te["entry_hour"] = te["entry_time"].dt.hour

    # Trades por día
    per_day = (te.groupby("entry_date", as_index=False)
                 .agg(trades=("tradeRealized","size"),
                      pnl=("tradeRealized","sum"),
                      winrate=("tradeRealized", lambda s: float((s>0).mean()*100) if len(s) else np.nan)))
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Promedio trades / día", f"{per_day['trades'].mean():.2f}" if len(per_day) else "N/A")
    colB.metric("Máximo trades en un día", f"{int(per_day['trades'].max())}" if len(per_day) else "N/A")
    colC.metric("Días con 3+ trades", f"{int((per_day['trades']>=3).sum())}" if len(per_day) else "0")
    colD.metric("Días totales", f"{len(per_day)}")

    c1, c2 = st.columns([1,1])
    n_days = len(per_day)
    with c1:
        if n_days > 90:
            dist = per_day.groupby("trades", as_index=False).size().rename(columns={"size":"dias"})
            fig = px.bar(dist, x="trades", y="dias", title="Distribución: # trades por día")
            fig.update_layout(xaxis_title="# trades en el día", yaxis_title="Días")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Con muchos días, esta distribución suele ser más útil que una barra por fecha.")
        else:
            fig = px.bar(per_day, x="entry_date", y="trades", title="Trades por día")
            fig.update_layout(xaxis_title="Día", yaxis_title="Trades")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(per_day, x="trades", y="pnl",
                         title="¿Más trades = mejor día? (PnL día vs # trades)",
                         hover_data=["entry_date","winrate"])
        fig.update_layout(xaxis_title="# trades del día", yaxis_title="PnL del día ($)")
        st.plotly_chart(fig, use_container_width=True)

        by_n = (per_day.groupby("trades", as_index=False)
                    .agg(dias=("entry_date","size"),
                         pnl_prom=("pnl","mean"),
                         pnl_med=("pnl","median")))
        fig2 = px.bar(by_n, x="trades", y="pnl_prom", color="pnl_prom",
                      color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                      title="Promedio PnL del día por # trades")
        fig2.update_layout(xaxis_title="# trades del día", yaxis_title="Promedio PnL del día ($)")
        st.plotly_chart(fig2, use_container_width=True)

        try:
            mean_12 = by_n[by_n["trades"].isin([1,2])]["pnl_prom"].mean()
            mean_3p = by_n[by_n["trades"]>=3]["pnl_prom"].mean()
            n_3p_days = int(by_n[by_n["trades"]>=3]["dias"].sum()) if len(by_n[by_n["trades"]>=3]) else 0
            if pd.notna(mean_12) and pd.notna(mean_3p) and n_3p_days >= 5 and mean_3p < mean_12:
                st.warning("⚠️ En esta muestra, los días con **3+ trades** rinden peor en promedio. "
                           "Prueba en el Lab **Máx trades/día = 2** o **MaxSetupsPerSession=2**.")
        except Exception:
            pass

    # Orden del trade dentro del día
    te["trade_num_day"] = te.groupby("entry_date").cumcount() + 1
    te["orden_trade"] = te["trade_num_day"].apply(lambda n: "1º" if n==1 else ("2º" if n==2 else ("3º" if n==3 else "4º+")))

    # Resumen por orden
    def _pf(sub):
        return profit_factor(sub) if len(sub) else np.nan

    by_order = (te.groupby("orden_trade", as_index=False)
                  .agg(trades=("tradeRealized","size"),
                       pnl_prom=("tradeRealized","mean"),
                       pnl_med=("tradeRealized","median"),
                       winrate=("tradeRealized", lambda s: float((s>0).mean()*100) if len(s) else np.nan)))
    # PF por orden
    pf_map = {k:_pf(v) for k,v in te.groupby("orden_trade")}
    by_order["pf"] = by_order["orden_trade"].map(pf_map)

    # Orden de presentación
    orden_cat = ["1º","2º","3º","4º+"]
    by_order["orden_trade"] = pd.Categorical(by_order["orden_trade"], categories=orden_cat, ordered=True)
    by_order = by_order.sort_values("orden_trade")

    st.markdown("### 🔎 ¿Cuál trade del día suele ser el peor?")
    st.write("Si ves que **#3 o #4+** baja fuerte (PF<1 o promedio<0), suele ser señal de **sobre-trading**.")
    st.dataframe(by_order, use_container_width=True, hide_index=True)

    # Consejos rápidos por orden (con muestra mínima)
    try:
        bo = by_order.copy()
        bo["orden_trade"] = bo["orden_trade"].astype(str)
        # Elegimos "peor" por promedio (y PF si existe)
        bo_sort = bo.sort_values(["pnl_prom"], ascending=True)
        worst_row = bo_sort.iloc[0] if len(bo_sort) else None
        if worst_row is not None and int(worst_row["trades"]) >= max(5, min_trades):
            if float(worst_row["pnl_prom"]) < 0:
                st.warning(f"⚠️ El **{worst_row['orden_trade']} trade** es el más flojo en esta muestra "
                           f"(prom={float(worst_row['pnl_prom']):.0f}$, PF={float(worst_row['pf']):.2f}). "
                           "Candidato a **limitar setups** o a subir confirmación después de 1–2 trades.")
        # Si 4º+ existe y es negativo con algo de muestra
        row4 = bo[bo["orden_trade"]=="4º+"]
        if len(row4) and int(row4.iloc[0]["trades"]) >= max(5, min_trades) and float(row4.iloc[0]["pnl_prom"]) < 0:
            st.warning("🚫 Los trades **4º+** salen negativos en promedio con muestra suficiente: "
                       "muy típico de **sobre-trading**. Prueba **MaxSetupsPerSession=2** o **Máx trades/día=2** en el Lab.")
    except Exception:
        pass

    fig = px.bar(by_order, x="orden_trade", y="pnl_prom", color="pnl_prom",
                 color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                 title="Promedio por trade según orden en el día")
    fig.update_layout(xaxis_title="Orden del trade en el día", yaxis_title="Promedio PnL por trade ($)")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(by_order, x="orden_trade", y="pf", color="pf",
                 color_continuous_scale="RdYlGn", color_continuous_midpoint=1.0,
                 title="Profit Factor según orden en el día")
    fig.add_hline(y=1.0, line_dash="dash")
    fig.update_layout(xaxis_title="Orden del trade en el día", yaxis_title="Profit Factor")
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap: hora vs orden (promedio y conteo)
    st.markdown("### 🕒 Hora de entrada vs orden del trade")
    st.write("Útil para detectar cosas tipo: **'el 2º trade después de cierta hora es malo'**.")
    tmp = te.copy()
    tmp["h"] = tmp["entry_hour"].astype("Int64")
    tmp["ord"] = tmp["orden_trade"].astype(str)
    pivot_mean = tmp.pivot_table(index="ord", columns="h", values="tradeRealized", aggfunc="mean")
    pivot_n = tmp.pivot_table(index="ord", columns="h", values="tradeRealized", aggfunc="size")

    # Solo mostramos si hay algo
    if pivot_mean.notna().sum().sum() > 0:
        # Formatear horas como HH:00 para que el eje sea legible (y no salga 9.5, 10.5, etc.)
        pivot_mean = pivot_mean.sort_index(axis=1)
        pivot_n = pivot_n.reindex(columns=pivot_mean.columns)

        hour_labels = [f"{int(h):02d}:00" for h in pivot_mean.columns]
        pivot_mean.columns = hour_labels
        pivot_n.columns = hour_labels

        # Texto dentro de cada celda: "promedio (n=..)"
        text = pivot_mean.copy()
        for r in text.index:
            for c in text.columns:
                a = pivot_mean.loc[r, c]
                n = pivot_n.loc[r, c]
                if pd.isna(a) or pd.isna(n) or n == 0:
                    text.loc[r, c] = ""
                else:
                    text.loc[r, c] = f"{a:.0f} (n={int(n)})"

        try:
            fig = px.imshow(
                pivot_mean,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                aspect="auto",
            )
            fig.update_traces(
                text=text,
                texttemplate="%{text}",
                customdata=pivot_n.values,
                hovertemplate="Hora: %{x}<br>Orden: %{y}<br>Promedio: %{z:.0f}$<br>n=%{customdata}<extra></extra>",
            )
            fig.update_layout(
                title="Heatmap (promedio PnL $) — muestra por celda",
                xaxis_title="Hora (entrada)",
                yaxis_title="Orden del trade",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Tu versión de Plotly no soporta este heatmap con texto. (No afecta el análisis.)")
    else:
        st.info("No hay suficientes datos con hora de entrada para el heatmap.")
else:
    st.info("No hay entry_time en los datos (faltan logs ENTRY). Esta sección necesita ENTRY.")

# ============================================================
# 🧪 Lab: Simulador de reglas diarias y filtros (tuning realista)
# ============================================================
st.subheader("🧪 Lab: Simulador de reglas diarias y filtros")
st.caption("Simula reglas reales: **MaxLoss/MaxProfit por día**, **máximo trades por día**, **racha de pérdidas**, y filtros (OR/EWO/ATR/Absorción/Actividad). "
           "La simulación es 'what-if': te dice cómo habría cambiado tu curva con esas reglas.")

def _hour_label(h: int) -> str:
    # 12h friendly
    if h == 0:
        return "12 AM"
    if 1 <= h <= 11:
        return f"{h} AM"
    if h == 12:
        return "12 PM"
    return f"{h-12} PM"


def _finite_minmax(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce")
    s2 = s2[np.isfinite(s2)]
    if s2.empty:
        return None, None
    return float(s2.min()), float(s2.max())


def _apply_filters(base: pd.DataFrame):
    """
    Filtros del Lab (intuitivos):
    - Por defecto NO filtra nada (equivale al Resumen rápido).
    - Un filtro "se activa" solo si cambias el control (no está en todo el rango / todo seleccionado).
    """
    df = base.copy()
    notes = []

    # Para horas/orden del día: usa ENTRY si existe, si no usa EXIT como respaldo (y lo avisamos).
    time_col = None
    used_fallback = False
    if "entry_time" in df.columns and df["entry_time"].notna().any():
        time_col = "entry_time"
    elif "exit_time" in df.columns and df["exit_time"].notna().any():
        time_col = "exit_time"
        used_fallback = True

    # -------------------- Horas permitidas --------------------
    if time_col is not None:
        df["_lab_hour"] = pd.to_datetime(df[time_col], errors="coerce").dt.hour
        hour_labels = [_hour_label(h) for h in range(24)]
        label_to_hour = {hour_labels[h]: h for h in range(24)}

        key_hours = "lab_hours_allowed"
        if key_hours not in st.session_state:
            st.session_state[key_hours] = hour_labels[:]  # all

        horas_sel_labels = st.multiselect(
            "Horas permitidas (entrada)",
            options=hour_labels,
            default=st.session_state[key_hours],
            key=key_hours,
            help="Por defecto: todas. Si faltan ENTRY, se usa EXIT como respaldo solo para este filtro."
        )
        horas_sel = [label_to_hour[x] for x in horas_sel_labels] if horas_sel_labels else []
        if len(horas_sel) < 24:
            df = df[df["_lab_hour"].isin(horas_sel)].copy()
            notes.append(f"Horas ({len(horas_sel)}/24)")
        if used_fallback:
            st.caption("ℹ️ Nota: algunos trades no tienen ENTRY; para el filtro de horas se usa EXIT como respaldo.")
    else:
        st.caption("ℹ️ No hay timestamps (ENTRY/EXIT) para filtrar por hora.")

    # Helper: activar filtro solo si cambia respecto a rango completo
    def _range_filter(col, key_rng, label, unit_hint=None):
        nonlocal df, notes
        if col not in df.columns:
            return
        lohi = _finite_minmax(df[col])
        if lohi == (None, None):
            return
        mn, mx = lohi
        if mn == mx:
            st.caption(f"ℹ️ {label}: solo un valor en la muestra ({mn:g}).")
            return

        if key_rng not in st.session_state:
            st.session_state[key_rng] = (mn, mx)  # por defecto: no filtra

        rng = st.slider(
            label,
            min_value=float(mn),
            max_value=float(mx),
            value=st.session_state[key_rng],
            key=key_rng,
            help=("Por defecto: todo el rango." + (f" Unidad: {unit_hint}." if unit_hint else ""))
        )
        # Activa solo si NO es todo el rango
        if (rng[0] > mn) or (rng[1] < mx):
            s = pd.to_numeric(df[col], errors="coerce")
            df = df[s.between(rng[0], rng[1], inclusive="both")].copy()
            notes.append(label.split(" ")[0])

    def _min_filter(col, key_thr, label, prefer_abs=False):
        nonlocal df, notes
        if col not in df.columns:
            return
        s = pd.to_numeric(df[col], errors="coerce")
        if prefer_abs:
            s = s.abs()
        s2 = s[np.isfinite(s)]
        if s2.empty:
            return
        mn, mx = float(s2.min()), float(s2.max())
        if mn == mx:
            st.caption(f"ℹ️ {label}: solo un valor en la muestra ({mn:g}).")
            return

        if key_thr not in st.session_state:
            st.session_state[key_thr] = mn  # por defecto: no filtra

        thr = st.slider(label, min_value=float(mn), max_value=float(mx), value=float(st.session_state[key_thr]), key=key_thr)
        if thr > mn:
            df = df[s >= thr].copy()
            notes.append(label.split(" ")[0])

    def _max_filter(col, key_thr, label):
        nonlocal df, notes
        if col not in df.columns:
            return
        s = pd.to_numeric(df[col], errors="coerce")
        s2 = s[np.isfinite(s)]
        if s2.empty:
            return
        mn, mx = float(s2.min()), float(s2.max())
        if mn == mx:
            st.caption(f"ℹ️ {label}: solo un valor en la muestra ({mn:g}).")
            return

        if key_thr not in st.session_state:
            st.session_state[key_thr] = mx  # por defecto: no filtra

        thr = st.slider(label, min_value=float(mn), max_value=float(mx), value=float(st.session_state[key_thr]), key=key_thr)
        if thr < mx:
            df = df[s <= thr].copy()
            notes.append(label.split(" ")[0])

    # -------------------- OR / EWO / ATR / Pressure --------------------
    _range_filter("orSize", "lab_or_rng", "OR Size permitido", unit_hint="puntos (según tu log)")
    _min_filter("ewo", "lab_ewo_thr", "|EWO| mínimo", prefer_abs=True)
    _range_filter("atr", "lab_atr_rng", "ATR permitido")

    _max_filter("pre_absorcion", "lab_abs_max", "Absorción máxima (evitar extremos)")
    _min_filter("pre_actividad_rel", "lab_act_min", "Actividad mínima (volumen relativo antes de entrar)")

    # Sin apoyo (divergencia) -> bool toggle, por defecto OFF
    if "pre_sin_apoyo" in df.columns:
        key_sa = "lab_avoid_sin_apoyo"
        if key_sa not in st.session_state:
            st.session_state[key_sa] = False
        avoid = st.checkbox("Evitar 'sin apoyo' (presión vs avance en contra)", value=st.session_state[key_sa], key=key_sa)
        if avoid:
            df = df[~df["pre_sin_apoyo"].fillna(False)].copy()
            notes.append("Sin apoyo")

    return df, notes
def _simulate_daily_rules(df: pd.DataFrame, max_loss: float, max_profit: float, max_trades: int,
                          max_consec_losses: int, stop_big_loss: bool, stop_big_win: bool):
    if df is None or df.empty:
        return df, pd.DataFrame()

    sim = df.copy()
    sim = sim[sim["entry_time"].notna()].copy() if "entry_time" in sim.columns else sim.copy()
    sim = sim.sort_values("entry_time" if "entry_time" in sim.columns else "exit_time").reset_index()
    sim["entry_date"] = (sim["entry_time"] if "entry_time" in sim.columns else sim["exit_time"]).dt.date

    kept_rows = []
    stops = []

    for day, sub in sim.groupby("entry_date"):
        sub = sub.sort_values("entry_time" if "entry_time" in sub.columns else "exit_time")
        pnl_day = 0.0
        consec = 0
        taken = 0
        halted = False
        stop_reason = None

        for _, row in sub.iterrows():
            if halted:
                break
            if max_trades > 0 and taken >= max_trades:
                halted = True
                stop_reason = "Máx trades/día"
                break
            if max_loss > 0 and pnl_day <= -abs(max_loss):
                halted = True
                stop_reason = "Max pérdida/día"
                break
            if max_profit > 0 and pnl_day >= abs(max_profit):
                halted = True
                stop_reason = "Max ganancia/día"
                break

            # Ejecuta trade
            kept_rows.append(int(row["index"]))
            tr = float(row.get("tradeRealized", 0) or 0)
            pnl_day += tr
            taken += 1
            if tr < 0:
                consec += 1
            else:
                consec = 0

            rr = row.get("rr", np.nan)
            if stop_big_loss and rr is not None and not (isinstance(rr, float) and np.isnan(rr)):
                if float(rr) <= -1.0:
                    halted = True
                    stop_reason = "Stop-out fuerte (≤ -1R)"
            if (not halted) and stop_big_win and rr is not None and not (isinstance(rr, float) and np.isnan(rr)):
                if float(rr) >= 2.0:
                    halted = True
                    stop_reason = "Cierre tras ganador grande (≥ 2R)"
            if (not halted) and max_consec_losses > 0 and consec >= max_consec_losses:
                halted = True
                stop_reason = f"{max_consec_losses} pérdidas seguidas"
            if (not halted) and max_loss > 0 and pnl_day <= -abs(max_loss):
                halted = True
                stop_reason = "Max pérdida/día"
            if (not halted) and max_profit > 0 and pnl_day >= abs(max_profit):
                halted = True
                stop_reason = "Max ganancia/día"

        if stop_reason is not None:
            stops.append({
                "fecha": day,
                "motivo": stop_reason,
                "pnl_hasta_stop": pnl_day,
                "trades_ejecutados": taken,
                "trades_totales_dia": int(len(sub)),
                "trades_filtrados_por_stop": int(max(0, len(sub) - taken))
            })

    kept = df.loc[kept_rows].copy()
    stops_df = pd.DataFrame(stops)
    return kept, stops_df

# Inputs del Lab
def _reset_lab_state(base_df: pd.DataFrame):
    # Reglas
    st.session_state["lab_max_loss"] = 600.0
    st.session_state["lab_max_profit"] = 4500.0
    st.session_state["lab_max_trades"] = 3
    st.session_state["lab_max_consec_losses"] = 2
    st.session_state["lab_stop_big_loss"] = False
    st.session_state["lab_stop_big_win"] = False

    # Filtros (por defecto: NO filtran)
    # Horas (todas)
    hour_labels = [_hour_label(h) for h in range(24)]
    st.session_state["lab_hours_allowed"] = hour_labels[:]
    st.session_state["lab_avoid_sin_apoyo"] = False

    # Rangos numéricos dinámicos
    if base_df is None or base_df.empty:
        return

    def _set_rng(col, key):
        if col in base_df.columns:
            mn, mx = _finite_minmax(base_df[col])
            if mn is not None and mx is not None and mn != mx:
                st.session_state[key] = (mn, mx)

    def _set_min_abs(col, key):
        if col in base_df.columns:
            s = pd.to_numeric(base_df[col], errors="coerce").abs()
            s = s[np.isfinite(s)]
            if not s.empty:
                st.session_state[key] = float(s.min())

    def _set_max(col, key):
        if col in base_df.columns:
            s = pd.to_numeric(base_df[col], errors="coerce")
            s = s[np.isfinite(s)]
            if not s.empty:
                st.session_state[key] = float(s.max())

    def _set_min(col, key):
        if col in base_df.columns:
            s = pd.to_numeric(base_df[col], errors="coerce")
            s = s[np.isfinite(s)]
            if not s.empty:
                st.session_state[key] = float(s.min())

    _set_rng("orSize", "lab_or_rng")
    _set_rng("atr", "lab_atr_rng")
    _set_min_abs("ewo", "lab_ewo_thr")
    _set_max("pre_absorcion", "lab_abs_max")
    _set_min("pre_actividad_rel", "lab_act_min")


reset_col, reset_info = st.columns([1, 3])
with reset_col:
    if st.button("🔄 Reset experimento", help="Vuelve al estado inicial: mismas cifras que Resumen rápido + sin filtros."):
        _reset_lab_state(t)
with reset_info:
    st.caption("Idea: el Lab **empieza igual que el Resumen rápido**. Luego ajustas reglas/filtros para ver el impacto (what‑if).")

lab_left, lab_right = st.columns([1, 1])

with lab_left:
    st.markdown("**Reglas diarias**")
    max_loss = st.number_input(
        "Max pérdida por día ($)",
        min_value=0.0, value=float(st.session_state.get("lab_max_loss", 600.0)), step=50.0,
        key="lab_max_loss",
        help="Si el día llega a -MaxLoss, se corta el trading del día."
    )
    max_profit = st.number_input(
        "Max ganancia por día ($)",
        min_value=0.0, value=float(st.session_state.get("lab_max_profit", 4500.0)), step=100.0,
        key="lab_max_profit",
        help="Si el día llega a +MaxProfit, se corta el trading del día."
    )
    max_trades = st.slider("Máx trades por día (0 = sin límite)", min_value=0, max_value=10,
                           value=int(st.session_state.get("lab_max_trades", 3)), key="lab_max_trades")
    max_consec_losses = st.slider("Máx pérdidas seguidas (0 = sin límite)", min_value=0, max_value=10,
                                  value=int(st.session_state.get("lab_max_consec_losses", 2)), key="lab_max_consec_losses")
    stop_big_loss = st.checkbox("Cortar el día tras un stop-out fuerte (RR ≤ -1)", value=bool(st.session_state.get("lab_stop_big_loss", False)),
                                key="lab_stop_big_loss")
    stop_big_win = st.checkbox("Cortar el día tras un ganador grande (RR ≥ 2)", value=bool(st.session_state.get("lab_stop_big_win", False)),
                               key="lab_stop_big_win")

with lab_right:
    st.markdown("**Filtros (opcional)**")
    base_for_lab = t.copy()  # <-- mismo universo que Resumen rápido

    # Aviso si faltan ENTRY (solo afecta a filtros que dependan de ENTRY/OR/EWO/ATR)
    missing_entry = int(base_for_lab["entry_time"].isna().sum()) if ("entry_time" in base_for_lab.columns) else 0
    if missing_entry > 0:
        st.warning(f"{missing_entry} operaciones no tienen ENTRY. En esas no se conoce Compra/Venta ni OR/EWO/ATR/DeltaRatio; "
                   "solo se podrán filtrar por hora usando EXIT como respaldo.", icon="⚠️")

    filtered, filter_notes = _apply_filters(base_for_lab)

    st.write(f"Trades tras filtros: **{len(filtered)}** (de {len(base_for_lab)})")
    if filter_notes:
        st.caption("Filtros activos: " + ", ".join(filter_notes))
    else:
        st.caption("Filtros activos: ninguno (equivale al Resumen rápido).")
# Simulación
if filtered is None or filtered.empty:
    st.info("Con los filtros actuales no quedan trades para simular.")
else:
    sim_kept, stops_df = _simulate_daily_rules(filtered, max_loss, max_profit, max_trades, max_consec_losses, stop_big_loss, stop_big_win)

    # Comparativa: base filtrada vs simulado (reglas)
    st.markdown("### 📊 Resultados del Lab (base filtrada vs simulado)")
    c1, c2, c3, c4, c5 = st.columns(5)
    base_pf = profit_factor(filtered) if len(filtered) else np.nan
    sim_pf = profit_factor(sim_kept) if len(sim_kept) else np.nan

    c1.metric("Trades (base filtrada)", f"{len(filtered)}")
    c2.metric("Trades (sim)", f"{len(sim_kept)}", delta=f"{len(sim_kept)-len(filtered)}")
    c3.metric("PnL base filtrada ($)", f"{filtered['tradeRealized'].sum():.0f}")
    c4.metric("PnL simulado ($)", f"{sim_kept['tradeRealized'].sum():.0f}", delta=f"{(sim_kept['tradeRealized'].sum()-filtered['tradeRealized'].sum()):.0f}")
    c5.metric("PF base / sim", f"{fmt(base_pf,2)} / {fmt(sim_pf,2)}")

    # Drawdown
    def _equity_df(df_):
        z = df_.copy()
        z = z.sort_values("exit_time" if "exit_time" in z.columns else "entry_time")
        z["equity"] = z["tradeRealized"].fillna(0).cumsum()
        z["equity_peak"] = z["equity"].cummax()
        z["drawdown"] = z["equity"] - z["equity_peak"]
        return z

    base_eq = _equity_df(filtered)
    sim_eq = _equity_df(sim_kept)

    dd_base = float(base_eq["drawdown"].min()) if len(base_eq) else np.nan
    dd_sim = float(sim_eq["drawdown"].min()) if len(sim_eq) else np.nan
    colx, coly, colz = st.columns(3)
    dd_base_mag = abs(dd_base) if not np.isnan(dd_base) else np.nan
    dd_sim_mag = abs(dd_sim) if not np.isnan(dd_sim) else np.nan

    colx.metric("Máx caída base ($)", f"{dd_base_mag:.0f}" if not np.isnan(dd_base_mag) else "N/A")
    coly.metric("Máx caída sim ($)", f"{dd_sim_mag:.0f}" if not np.isnan(dd_sim_mag) else "N/A",
                delta=f"{(dd_sim_mag-dd_base_mag):.0f}" if (not np.isnan(dd_sim_mag) and not np.isnan(dd_base_mag)) else None,
                delta_color="inverse")
    colz.metric("Días 'cortados' (reglas)", f"{len(stops_df)}")
    st.caption("Máx caída (Drawdown) = peor retroceso desde el pico de la curva de equity. **Más bajo = mejor**.")

    # Equity curve
    st.markdown("#### Curva de equity (base vs simulado)")
    # Alineamos por índice temporal (exit_time)
    base_plot = base_eq[["exit_time","equity"]].rename(columns={"equity":"Equity base"})
    sim_plot = sim_eq[["exit_time","equity"]].rename(columns={"equity":"Equity sim"})
    plot_df = pd.merge_asof(base_plot.sort_values("exit_time"),
                            sim_plot.sort_values("exit_time"),
                            on="exit_time", direction="nearest", tolerance=pd.Timedelta("1D"))
    plot_df = plot_df.sort_values("exit_time")
    fig = px.line(plot_df, x="exit_time", y=["Equity base","Equity sim"], title="Equity ($)")
    st.plotly_chart(fig, use_container_width=True)

    # Impacto de las reglas (en vez de solo frecuencia)
    if stops_df is not None and not stops_df.empty:
        st.markdown("#### ¿Qué regla detuvo el día y qué impacto tuvo?")

        # Construimos, por día, la secuencia de PnL en el orden real de entrada
        _lab_base = filtered.copy()
        _time_col = "entry_time" if "entry_time" in _lab_base.columns else "exit_time"
        _lab_base = _lab_base[_lab_base[_time_col].notna()].copy()
        _lab_base = _lab_base.sort_values(_time_col)
        _lab_base["lab_day"] = _lab_base[_time_col].dt.date

        def _day_dd_mag(pnls):
            if pnls is None or len(pnls) == 0:
                return 0.0
            c = np.cumsum(pnls)
            peak = np.maximum.accumulate(c)
            dd = c - peak
            return float(abs(np.min(dd))) if len(dd) else 0.0

        day_to_pnls = {}
        for d, subd in _lab_base.groupby("lab_day"):
            pnls = subd["tradeRealized"].fillna(0).astype(float).tolist()
            day_to_pnls[d] = pnls

        enriched = stops_df.copy()

        # Enriquecemos con "qué PnL quedó fuera" y "mejora de DD intradía" (aprox)
        pnl_total_base = []
        pnl_omitido = []
        delta_pnl = []
        dd_day_base = []
        dd_day_sim = []
        delta_dd = []
        stop_ahorra = []

        for _, r in enriched.iterrows():
            d = r["fecha"]
            taken = int(r.get("trades_ejecutados", 0) or 0)
            pnls = day_to_pnls.get(d, [])
            base_total = float(np.sum(pnls)) if pnls else 0.0
            kept_total = float(np.sum(pnls[:taken])) if pnls else 0.0
            skipped = base_total - kept_total              # lo que "se pierde" por cortar
            d_pnl = kept_total - base_total                # impacto vs base (positivo = mejora)

            dd_b = _day_dd_mag(pnls)
            dd_s = _day_dd_mag(pnls[:taken])
            d_dd = dd_b - dd_s                             # positivo = reduce caída intradía

            pnl_total_base.append(base_total)
            pnl_omitido.append(skipped)
            delta_pnl.append(d_pnl)
            dd_day_base.append(dd_b)
            dd_day_sim.append(dd_s)
            delta_dd.append(d_dd)
            stop_ahorra.append(skipped < 0)                # si lo omitido era negativo, el corte evitó pérdidas

        enriched["pnl_total_dia_base"] = pnl_total_base
        enriched["pnl_omitido_por_corte"] = pnl_omitido
        enriched["delta_pnl_vs_base"] = delta_pnl
        enriched["dd_dia_base"] = dd_day_base
        enriched["dd_dia_sim"] = dd_day_sim
        enriched["mejora_dd_dia"] = delta_dd
        enriched["evito_perdidas"] = stop_ahorra

        # Resumen por motivo (lo realmente útil)
        impact = (
            enriched.groupby("motivo", as_index=False)
            .agg(
                dias=("fecha", "count"),
                trades_omitidos=("trades_filtrados_por_stop", "sum"),
                delta_pnl_total=("delta_pnl_vs_base", "sum"),
                delta_pnl_prom=("delta_pnl_vs_base", "mean"),
                mejora_dd_total=("mejora_dd_dia", "sum"),
                mejora_dd_prom=("mejora_dd_dia", "mean"),
                pct_evito_perdidas=("evito_perdidas", "mean"),
            )
        )
        impact["pct_evito_perdidas"] = (impact["pct_evito_perdidas"] * 100).round(0)

        # Gráficos de impacto (PnL y DD) — mucho más interpretables que "frecuencia"
        leftg, rightg = st.columns(2)

        with leftg:
            st.markdown("**Impacto en PnL al cortar el día**")
            figp = px.bar(
                impact.sort_values("delta_pnl_total"),
                x="motivo",
                y="delta_pnl_total",
                color="delta_pnl_total",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                text=impact.sort_values("delta_pnl_total")["delta_pnl_total"].round(0).astype(int),
                hover_data={"dias": True, "trades_omitidos": True, "pct_evito_perdidas": True},
                title="ΔPnL total vs base ($) — positivo = mejora",
            )
            figp.update_layout(xaxis_title="Regla", yaxis_title="ΔPnL ($)")
            st.plotly_chart(figp, use_container_width=True)

        with rightg:
            st.markdown("**Impacto en Drawdown intradía (aprox.)**")
            figd = px.bar(
                impact.sort_values("mejora_dd_total"),
                x="motivo",
                y="mejora_dd_total",
                color="mejora_dd_total",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                text=impact.sort_values("mejora_dd_total")["mejora_dd_total"].round(0).astype(int),
                hover_data={"dias": True, "trades_omitidos": True, "pct_evito_perdidas": True},
                title="Mejora DD total ($) — positivo = reduce caídas",
            )
            figd.update_layout(xaxis_title="Regla", yaxis_title="Mejora DD ($)")
            
            st.plotly_chart(figd, use_container_width=True)

        # Relación entre reglas: trade-off PnL vs reducción de drawdown (visión de cuadrantes)
        if len(impact) >= 1:
            impact_sc = impact.copy()
            def _quad(row):
                dp = float(row.get("delta_pnl_total", 0) or 0)
                dd = float(row.get("mejora_dd_total", 0) or 0)
                if dp >= 0 and dd >= 0:
                    return "Mejora PnL y reduce DD"
                if dp < 0 and dd >= 0:
                    return "Reduce DD pero cuesta PnL"
                if dp >= 0 and dd < 0:
                    return "Mejora PnL pero empeora DD"
                return "Empeora ambos"
            impact_sc["cuadrante"] = impact_sc.apply(_quad, axis=1)

            st.markdown("**Mapa PnL vs Drawdown por regla (trade‑off)**")
            figs = px.scatter(
                impact_sc,
                x="delta_pnl_total",
                y="mejora_dd_total",
                size="dias" if "dias" in impact_sc.columns else None,
                color="cuadrante",
                hover_name="motivo",
                hover_data={
                    "dias": True,
                    "trades_omitidos": True,
                    "pct_evito_perdidas": True,
                    "delta_pnl_total": ":.0f",
                    "mejora_dd_total": ":.0f",
                },
                labels={
                    "delta_pnl_total": "ΔPnL vs base ($)",
                    "mejora_dd_total": "Mejora DD total ($)",
                    "dias": "Días activada",
                },
                title="Cada punto = una regla. Derecha/arriba = mejor. Tamaño = días activada.",
            )
            figs.add_vline(x=0, line_dash="dash", opacity=0.4)
            figs.add_hline(y=0, line_dash="dash", opacity=0.4)
            figs.update_layout(xaxis_title="ΔPnL ($)  (positivo = mejora)", yaxis_title="Mejora DD ($)  (positivo = reduce caídas)")
            st.plotly_chart(figs, use_container_width=True)

            if len(impact_sc) >= 3:
                try:
                    r = float(np.corrcoef(impact_sc["delta_pnl_total"], impact_sc["mejora_dd_total"])[0,1])
                    st.caption(f"Correlación entre ΔPnL y Mejora DD (entre reglas): r = {r:.2f}. "
                               "Con pocas reglas, úsalo solo como referencia visual.")
                except Exception:
                    pass

        st.caption(

            "Interpretación: **ΔPnL** positivo significa que cortar evitó más pérdidas de las que dejó pasar. "
            "**Mejora DD** positiva significa menor caída intradía en los días donde se activó esa regla. "
            "Esto es una aproximación por día (no reemplaza el DD global)."
        )

        # Tabla detallada (para quien quiera auditar)
        
        # ------------------------------------------------------------
        # Extra: "What-if" por regla (una regla a la vez)
        # Esto evita confusión cuando varias reglas están activas a la vez.
        # ------------------------------------------------------------
        st.markdown("#### 🧩 What‑if por regla (una regla a la vez)")
        st.caption("Aquí se simula **cada regla por separado** (manteniendo los filtros) y se compara contra la base filtrada. "
                   "Sirve para ver el **trade‑off PnL vs Drawdown** sin mezclar reglas.")

        def _dd_mag_from_df(df_):
            if df_ is None or df_.empty:
                return 0.0
            z = df_.copy()
            z = z.sort_values("exit_time" if "exit_time" in z.columns else "entry_time")
            eq = z["tradeRealized"].fillna(0).astype(float).cumsum()
            peak = eq.cummax()
            dd = eq - peak
            return float(abs(dd.min())) if len(dd) else 0.0

        pnl_base_f = float(filtered["tradeRealized"].fillna(0).astype(float).sum()) if (filtered is not None and not filtered.empty) else 0.0
        dd_base_f  = _dd_mag_from_df(filtered)
        pf_base_f  = profit_factor(filtered["tradeRealized"]) if (filtered is not None and not filtered.empty and "tradeRealized" in filtered.columns) else np.nan

        # Construye escenarios individuales en función de lo que el usuario activó
        scenarios = []
        if max_loss > 0:
            scenarios.append(("Max pérdida/día", dict(max_loss=max_loss, max_profit=0.0, max_trades=0, max_consec_losses=0, stop_big_loss=False, stop_big_win=False)))
        if max_profit > 0:
            scenarios.append(("Max ganancia/día", dict(max_loss=0.0, max_profit=max_profit, max_trades=0, max_consec_losses=0, stop_big_loss=False, stop_big_win=False)))
        if max_trades > 0:
            scenarios.append(("Máx trades/día", dict(max_loss=0.0, max_profit=0.0, max_trades=max_trades, max_consec_losses=0, stop_big_loss=False, stop_big_win=False)))
        if max_consec_losses > 0:
            scenarios.append((f"{max_consec_losses} pérdidas seguidas", dict(max_loss=0.0, max_profit=0.0, max_trades=0, max_consec_losses=max_consec_losses, stop_big_loss=False, stop_big_win=False)))
        if stop_big_loss:
            scenarios.append(("Stop‑out fuerte (≤ -1R)", dict(max_loss=0.0, max_profit=0.0, max_trades=0, max_consec_losses=0, stop_big_loss=True, stop_big_win=False)))
        if stop_big_win:
            scenarios.append(("Cierre tras ganador grande (≥ 2R)", dict(max_loss=0.0, max_profit=0.0, max_trades=0, max_consec_losses=0, stop_big_loss=False, stop_big_win=True)))

        rows = []
        for name, params in scenarios:
            sim_one, stops_one = _simulate_daily_rules(filtered, **params)
            pnl_sim_one = float(sim_one["tradeRealized"].fillna(0).astype(float).sum()) if (sim_one is not None and not sim_one.empty) else 0.0
            dd_sim_one  = _dd_mag_from_df(sim_one)
            pf_sim_one  = profit_factor(sim_one["tradeRealized"]) if (sim_one is not None and not sim_one.empty and "tradeRealized" in sim_one.columns) else np.nan

            rows.append({
                "regla": name,
                "trades_sim": int(len(sim_one)) if sim_one is not None else 0,
                "dias_cortados": int(len(stops_one)) if stops_one is not None else 0,
                "delta_pnl": pnl_sim_one - pnl_base_f,
                "mejora_dd": dd_base_f - dd_sim_one,
                "pf_sim": pf_sim_one,
            })

        # Punto "Combinado" (todas las reglas activas) si hay más de 1 regla
        if len(scenarios) >= 2:
            pnl_base_comb = float(filtered["tradeRealized"].fillna(0).sum()) if (filtered is not None and not filtered.empty and "tradeRealized" in filtered.columns) else 0.0
            pnl_sim_comb  = float(sim_kept["tradeRealized"].fillna(0).sum()) if (sim_kept is not None and not sim_kept.empty and "tradeRealized" in sim_kept.columns) else 0.0
            rows.append({
                "regla": "Combinado (todas)",
                "trades_sim": int(len(sim_kept)) if sim_kept is not None else 0,
                "dias_cortados": int(len(stops_df)) if stops_df is not None else 0,
                "delta_pnl": (float(sim_kept["tradeRealized"].sum()) if (sim_kept is not None and len(sim_kept)>0 and "tradeRealized" in sim_kept.columns) else 0.0) - (float(filtered["tradeRealized"].sum()) if (filtered is not None and len(filtered)>0 and "tradeRealized" in filtered.columns) else 0.0),
                "mejora_dd": dd_base_mag - dd_sim_mag,
                "pf_sim": sim_pf,
            })

        if rows:
            wf = pd.DataFrame(rows)
            def _quad2(r):
                dp = float(r["delta_pnl"])
                ddm = float(r["mejora_dd"])
                if dp >= 0 and ddm >= 0: return "✅ Mejora PnL y DD"
                if dp < 0 and ddm >= 0:  return "🟡 Reduce DD pero cuesta PnL"
                if dp >= 0 and ddm < 0:  return "🟠 Mejora PnL pero empeora DD"
                return "❌ Empeora ambos"
            wf["cuadrante"] = wf.apply(_quad2, axis=1)

            # Gráfico trade-off (más estable con muchas reglas)
            figwf = px.scatter(
                wf,
                x="delta_pnl",
                y="mejora_dd",
                size="dias_cortados",
                color="cuadrante",
                hover_name="regla",
                hover_data={"trades_sim": True, "dias_cortados": True, "pf_sim": True},
                title="Mapa (what‑if) por regla — ΔPnL vs mejora DD (cada regla por separado)",
            )
            figwf.add_vline(x=0, line_dash="dash", opacity=0.4)
            figwf.add_hline(y=0, line_dash="dash", opacity=0.4)

            # Asegura que el 0 siempre sea visible (evita ejes “pegados” a un lado)
            try:
                xmin = float(wf["delta_pnl"].min()); xmax = float(wf["delta_pnl"].max())
                ymin = float(wf["mejora_dd"].min()); ymax = float(wf["mejora_dd"].max())
                padx = max(100.0, (xmax - xmin) * 0.15)
                pady = max(100.0, (ymax - ymin) * 0.15)
                figwf.update_xaxes(range=[min(xmin, 0.0) - padx, max(xmax, 0.0) + padx])
                figwf.update_yaxes(range=[min(ymin, 0.0) - pady, max(ymax, 0.0) + pady])
            except Exception:
                pass

            figwf.update_layout(xaxis_title="ΔPnL vs base ($)  (positivo = mejor)", yaxis_title="Mejora DD ($)  (positivo = reduce caídas)")
            st.plotly_chart(figwf, use_container_width=True)

            with st.expander("Ver tabla (what‑if por regla)"):
                wf2 = wf.sort_values(["cuadrante","mejora_dd","delta_pnl"], ascending=[True, False, False]).copy()
                wf2["delta_pnl"] = wf2["delta_pnl"].round(0).astype(int)
                wf2["mejora_dd"] = wf2["mejora_dd"].round(0).astype(int)
                st.dataframe(wf2[["regla","delta_pnl","mejora_dd","dias_cortados","trades_sim","pf_sim","cuadrante"]], use_container_width=True, hide_index=True)

            # Consejo rápido automático
            best = wf[wf["regla"] != "Combinado (todas)"].copy()
            if not best.empty and (best["dias_cortados"].sum() > 0):
                cand = best.sort_values(["mejora_dd","delta_pnl"], ascending=False).iloc[0]
                st.success(f"✅ Regla con mejor equilibrio (en esta muestra): **{cand['regla']}** "
                           f"(ΔPnL {cand['delta_pnl']:.0f}$, mejora DD {cand['mejora_dd']:.0f}$). "
                           "Ojo: valida con más trades y no optimices 5 cosas a la vez.")
        else:
            st.info("Activa al menos una regla diaria para ver el 'what‑if por regla'.")

        with st.expander("Ver detalle por día (qué se omitió al cortar)"):
            show_cols = [
                "fecha", "motivo",
                "trades_ejecutados", "trades_totales_dia", "trades_filtrados_por_stop",
                "pnl_total_dia_base", "pnl_hasta_stop", "pnl_omitido_por_corte", "delta_pnl_vs_base",
                "dd_dia_base", "dd_dia_sim", "mejora_dd_dia",
            ]
            st.dataframe(
                enriched[show_cols].sort_values(["fecha"]),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Con estas reglas, no se activó ningún 'corte del día'.")
    # Consejos automáticos del Lab
    st.markdown("### 💡 Consejos automáticos (Lab)")
    if len(filtered) < 30:
        st.warning("Muestra pequeña tras filtros: toma estas conclusiones como hipótesis, no como regla.")
    if len(sim_kept) < max(10, int(0.25*len(filtered))):
        st.warning("La simulación está eliminando demasiados trades. Revisa si estás filtrando/cortando en exceso.")

    # OJO: dd_base/dd_sim son negativos (caída desde el pico). Para comparar "mejor/peor" usamos magnitudes.
    if (not np.isnan(dd_base_mag)) and (not np.isnan(dd_sim_mag)) and dd_sim_mag < dd_base_mag:
        st.success("✅ Menor drawdown con reglas: buen candidato para un 'Daily Guard' realista.")
    if (not np.isnan(dd_base_mag)) and (not np.isnan(dd_sim_mag)) and dd_sim_mag > dd_base_mag:
        st.warning("⚠️ El drawdown empeoró en la simulación: estas reglas podrían estar cortando recuperación o dejando pérdidas grandes.")

    if (not np.isnan(base_pf)) and (not np.isnan(sim_pf)) and sim_pf > base_pf + 0.2 and (not np.isnan(dd_base_mag)) and (not np.isnan(dd_sim_mag)) and dd_sim_mag > dd_base_mag:
        st.info("Mejoró PF pero empeoró DD: típico cuando eliminas muchos trades pequeños pero te quedas con pérdidas grandes. Revisa filtros/horarios.")
    if (not np.isnan(base_pf)) and (not np.isnan(sim_pf)) and sim_pf < base_pf and (not np.isnan(dd_base_mag)) and (not np.isnan(dd_sim_mag)) and dd_sim_mag < dd_base_mag:
        st.info("Bajó PF pero también bajó el DD: a veces vale la pena si tu objetivo es estabilidad (menos estrés / menos reset).")
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

recommend_settings_block(known, min_trades=min_trades, recommended_trades=recommended_trades)

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
