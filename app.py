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
    # Soporta UTF-8 con o sin BOM (utf-8-sig) y evita romper por caracteres raros
    txt = None
    for enc in ("utf-8-sig", "utf-8"):
        try:
            txt = b.decode(enc, errors="replace")
            break
        except Exception:
            continue
    if txt is None:
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
# Helpers: arrays (vol/delta/OHLC) & features pre-entrada
# ============================================================
def _as_list(x):
    """Convierte a lista si viene como list o como string tipo '[...]'."""
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


def add_pressure_activity_features(t: pd.DataFrame) -> pd.DataFrame:
    """Añade métricas de presión/actividad usando arrays *pre-entrada* (LastN) si existen."""
    if t is None or t.empty:
        return t
    t = t.copy()

    # Asegura numéricos base
    for c in ["tickSize", "pointValue", "slTicks", "qtyTP1", "qtyRunner", "volEntrySoFar", "deltaEntrySoFar"]:
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
                "pre_pressure_sum": np.nan,
                "pre_pressure_abs": np.nan,
                "pre_vol_sum": np.nan,
                "pre_price_net": np.nan,
                "pre_price_range": np.nan,
                "pre_move_ticks": np.nan,
                "pre_absorption": np.nan,
                "pre_activity_rel": np.nan,
                "pre_pressure_per_vol": np.nan,
                "pre_divergence": False,
            })

        pressure_sum = float(np.nansum(dl))
        pressure_abs = float(abs(pressure_sum))
        vol_sum = float(np.nansum(vl))

        # Movimiento de precio pre-entrada
        try:
            price_net = float(cl[-1] - ol[0])
        except Exception:
            price_net = np.nan

        try:
            price_range = float(np.nanmax(hl) - np.nanmin(ll))
        except Exception:
            price_range = np.nan

        tick_size = r.get("tickSize")
        if tick_size is None or (isinstance(tick_size, float) and np.isnan(tick_size)) or tick_size == 0:
            move_ticks = np.nan
        else:
            move_ticks = price_range / float(tick_size) if price_range == price_range else np.nan

        # Absorción: mucha presión pero poco avance
        if move_ticks == move_ticks:
            absorption = pressure_abs / max(move_ticks, 1.0)
        else:
            absorption = np.nan

        # Actividad relativa: volumen acumulado hasta entrada vs mediana del volumen reciente
        med_vol = float(np.nanmedian(vl)) if len(vl) else np.nan
        vol_entry = r.get("volEntrySoFar")
        if med_vol and med_vol == med_vol and med_vol > 0 and vol_entry == vol_entry:
            activity_rel = float(vol_entry / med_vol)
        else:
            activity_rel = np.nan

        # Presión por volumen (0..1 aprox si normalizado)
        if vol_sum and vol_sum > 0 and vol_sum == vol_sum:
            pressure_per_vol = float(pressure_abs / vol_sum)
        else:
            pressure_per_vol = np.nan

        # Divergencia (precio avanza pero delta no acompaña) según dirección
        d = r.get("dir")
        divergence = False
        if d == d and price_net == price_net:
            if d > 0 and price_net > 0 and pressure_sum < 0:
                divergence = True
            elif d < 0 and price_net < 0 and pressure_sum > 0:
                divergence = True

        return pd.Series({
            "pre_pressure_sum": pressure_sum,
            "pre_pressure_abs": pressure_abs,
            "pre_vol_sum": vol_sum,
            "pre_price_net": price_net,
            "pre_price_range": price_range,
            "pre_move_ticks": move_ticks,
            "pre_absorption": absorption,
            "pre_activity_rel": activity_rel,
            "pre_pressure_per_vol": pressure_per_vol,
            "pre_divergence": divergence,
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

def traffic_expectancy(exp: float):
    if exp is None or np.isnan(exp):
        return "⚪ Expectancia N/A"
    if exp < 0:    return "🔴 Expectancia < 0"
    if exp < 10:   return "🟡 Expectancia baja"
    if exp < 30:   return "🟢 Expectancia buena"
    return "🟣 Expectancia alta"

def hour_bucket_label(h):
    if pd.isna(h):
        return "Sin hora"
    h = int(h)
    return f"{h:02d}:00–{h:02d}:59"


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
    for i, x in enumerate(outcomes):
        if x == target:
            cur += 1
            if cur > best_len:
                best_len = cur
                best_end = i
        else:
            cur = 0
    if best_len == 0:
        return 0, None, None
    best_start = best_end - best_len + 1
    return best_len, best_start, best_end


def drawdown_details(trades: pd.DataFrame):
    if trades.empty:
        return 0.0, None, None
    equity = trades["tradeRealized"].fillna(0).cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = float(dd.min())
    trough_idx = dd.idxmin()
    peak_idx = equity.loc[:trough_idx].idxmax()
    return max_dd, trades.loc[peak_idx, "exit_time"], trades.loc[trough_idx, "exit_time"]


def summarize(trades: pd.DataFrame) -> dict:
    t = trades.copy()
    t = t.dropna(subset=["tradeRealized"])
    n = len(t)
    if n == 0:
        return {
            "n": 0, "wins": 0, "losses": 0, "win_rate": np.nan,
            "pnl_total": 0.0, "pf": np.nan, "expectancy": np.nan,
            "max_dd": 0.0, "dd_peak_time": None, "dd_trough_time": None,
            "max_win": np.nan, "max_loss": np.nan,
            "best_win_streak": 0, "best_loss_streak": 0,
        }

    wins = int((t["tradeRealized"] > 0).sum())
    losses = int((t["tradeRealized"] <= 0).sum())
    win_rate = wins / n * 100

    pf = profit_factor(t)
    expectancy = float(t["tradeRealized"].mean())

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


def group_metrics(df: pd.DataFrame, group_col: str, min_trades: int):
    """Métricas por grupo con etiqueta de tamaño de muestra.
    - No oculta grupos pequeños: los marca como 'Muestra pequeña' o 'No concluyente'.
    - Para gráficos/decisiones, filtra luego por Trades >= min_trades o >= recommended_trades.
    """
    rec = globals().get("recommended_trades", max(60, min_trades))
    rows = []
    for g, sub in df.groupby(group_col):
        n = int(len(sub))
        wins = int((sub["tradeRealized"] > 0).sum()) if "tradeRealized" in sub.columns else 0
        wr = wins / n * 100 if n > 0 else np.nan
        wr_adj = (wins + 1) / (n + 2) * 100 if n > 0 else np.nan  # suavizado

        pf = profit_factor(sub) if "tradeRealized" in sub.columns else np.nan
        exp = float(sub["tradeRealized"].mean()) if "tradeRealized" in sub.columns else np.nan
        pnl = float(sub["tradeRealized"].sum()) if "tradeRealized" in sub.columns else np.nan
        score = exp * np.log1p(n) if exp == exp else np.nan  # pondera tamaño de muestra

        if n >= rec:
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
            "WinRate (suav.) %": wr_adj,
            "Profit Factor": pf,
            "Promedio por trade": exp,
            "PnL Total": pnl,
            "Score (ponderado)": score,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Grupo", "Trades", "Estado", "WinRate %", "WinRate (suav.) %", "Profit Factor",
            "Promedio por trade", "PnL Total", "Score (ponderado)"
        ])

    tbl = pd.DataFrame(rows)
    tbl = tbl.sort_values(["Score (ponderado)", "Trades"], ascending=[False, False], na_position="last")
    return tbl


def advice_from_table(tbl: pd.DataFrame, title: str, min_trades: int):
    if tbl is None or tbl.empty:
        st.info(f"En **{title}** no hay grupos con muestra mínima (min={min_trades}). Para decisiones, recomendado ≥ {globals().get('recommended_trades', max(60, min_trades))} trades por grupo.")
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


def plot_hist_pnl(t: pd.DataFrame):
    fig = px.histogram(t, x="tradeRealized", nbins=40, title="Distribución de PnL por operación")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    fig.add_vline(x=0, line_width=1, line_dash="dash")
    return fig


def plot_scatter_advanced(df_known: pd.DataFrame, xcol: str, title: str,
                          trend_mode: str = "Regresión (OLS)", lowess_frac: float = 0.25):
    # Import aquí para que si alguien no instaló statsmodels, el resto de la app igual cargue.
    try:
        import statsmodels.api as sm
    except Exception:
        sm = None

    tmp = df_known[[xcol, "tradeRealized", "exit_time", "lado", "exitReason"]].dropna().copy()
    tmp["Resultado"] = np.where(tmp["tradeRealized"] >= 0, "Ganancia", "Pérdida")

    fig = px.scatter(
        tmp,
        x=xcol,
        y="tradeRealized",
        color="Resultado",
        color_discrete_map={"Ganancia": "green", "Pérdida": "red"},
        hover_data=["exit_time", "lado", "exitReason"],
        title=title
    )

    # Línea de tendencia
    if trend_mode == "Regresión (OLS)":
        if sm is not None and len(tmp) >= 5:
            x = tmp[xcol].astype(float).values
            y = tmp["tradeRealized"].astype(float).values
            X = sm.add_constant(x)
            model = sm.OLS(y, X, missing="drop").fit()
            xline = np.linspace(np.nanmin(x), np.nanmax(x), 80)
            yline = model.params[0] + model.params[1] * xline
            df_line = pd.DataFrame({xcol: xline, "tradeRealized": yline})
            fig.add_traces(px.line(df_line, x=xcol, y="tradeRealized").data)
            fig.add_annotation(
                text=f"OLS: y={model.params[0]:.1f}+{model.params[1]:.2f}x | R²={model.rsquared:.2f}",
                xref="paper", yref="paper", x=0.01, y=0.98,
                showarrow=False
            )
        else:
            st.info("No se pudo dibujar OLS (falta statsmodels o hay poca data).")

    elif trend_mode == "Suavizado (LOWESS)":
        if sm is not None and len(tmp) >= 10:
            x = tmp[xcol].astype(float).values
            y = tmp["tradeRealized"].astype(float).values
            low = sm.nonparametric.lowess(y, x, frac=lowess_frac, return_sorted=True)
            df_line = pd.DataFrame({xcol: low[:, 0], "tradeRealized": low[:, 1]})
            fig.add_traces(px.line(df_line, x=xcol, y="tradeRealized").data)
            fig.add_annotation(
                text=f"LOWESS (suavidad={lowess_frac})",
                xref="paper", yref="paper", x=0.01, y=0.98,
                showarrow=False
            )
        else:
            st.info("No se pudo dibujar LOWESS (falta statsmodels o hay poca data).")

    fig.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Cómo leerlo: verde = PnL ≥ 0, rojo = PnL < 0. "
        "OLS muestra tendencia lineal; LOWESS muestra forma real si hay curva."
    )


def plot_factor_bins(df_known: pd.DataFrame, col: str, q: int, min_trades: int, title: str,
                     show_scatter: bool, trend_mode: str, lowess_frac: float, scatter_df: pd.DataFrame):
    bins = make_bins_quantiles(df_known, col, q)
    if bins is None:
        st.info(f"No hay suficiente data para crear rangos por cuantiles en **{title}**.")
        return

    tmp = df_known.copy()
    tmp["_bin"] = bins.astype(str)

    tbl = group_metrics(tmp, "_bin", min_trades=min_trades)
    if tbl.empty:
        st.info(f"En **{title}** no hay datos para agrupar.")
        return

    rec = globals().get("recommended_trades", max(60, min_trades))
    tbl_valid = tbl[tbl["Trades"] >= min_trades].copy()

    # Aviso de muestra
    n_ok = int((tbl["Estado"] == "🟢 Suficiente").sum()) if "Estado" in tbl.columns else 0
    n_small = int((tbl["Estado"] == "🟡 Muestra pequeña").sum()) if "Estado" in tbl.columns else 0
    n_bad = int((tbl["Estado"] == "🔴 No concluyente").sum()) if "Estado" in tbl.columns else 0
    st.caption(f"Mínimo para mostrar: {min_trades} trades/bin. Recomendado para decidir: ≥ {rec} trades/bin. "
               f"Bins: 🟢{n_ok} 🟡{n_small} 🔴{n_bad}")

    if not tbl_valid.empty:
        fig_exp = px.bar(tbl_valid, x="Grupo", y="Promedio por trade", title=f"{title} → Promedio por trade (bins)")
        fig_exp.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_exp, use_container_width=True)

        fig_pf = px.bar(tbl_valid, x="Grupo", y="Profit Factor", title=f"{title} → Profit Factor (bins)")
        fig_pf.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_pf, use_container_width=True)

        fig_wr = px.bar(tbl_valid, x="Grupo", y="WinRate (suav.) %", title=f"{title} → WinRate suavizado (bins)")
        fig_wr.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_wr, use_container_width=True)

        fig_pnl = px.bar(tbl_valid, x="Grupo", y="PnL Total", title=f"{title} → PnL Total (bins)")
        fig_pnl.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_pnl, use_container_width=True)

        advice_from_table(tbl_valid, title, min_trades=min_trades)
    else:
        st.info(f"⚠️ En **{title}** todo queda con muestra pequeña. Úsalo como idea, no como regla.")

    st.dataframe(tbl, use_container_width=True)

    if show_scatter:
        st.markdown("**Scatter (para ver dispersión real)**")
        if col in scatter_df.columns and scatter_df[col].notna().sum() >= max(20, min_trades):
            plot_scatter_advanced(scatter_df, col, title=f"{title} → Scatter", trend_mode=trend_mode, lowess_frac=lowess_frac)
        else:
            st.info("No hay suficiente data para scatter en este factor.")


def plot_hour_analysis(t: pd.DataFrame, min_trades: int):
    tbl = group_metrics(t, "exit_hour_label", min_trades=min_trades)
    if tbl.empty:
        st.info(f"No hay data por hora para min_trades={min_trades}.")
        return None

    tbl_valid = tbl[tbl["Trades"] >= min_trades].copy()
    if tbl_valid.empty:
        st.info("⚠️ Por hora hay muy poca muestra. Se muestra tabla completa con avisos.")
    else:
        fig = px.bar(tbl_valid, x="Grupo", y="Score (ponderado)",
                     title="Horas más prometedoras (Score ponderado por tamaño de muestra)")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(tbl, use_container_width=True)
    return tbl


def plot_heatmap_weekday_hour(t: pd.DataFrame, min_trades: int):
    if t.empty:
        return

    tmp = t.copy()
    tmp["weekday"] = tmp["exit_time"].dt.day_name()
    tmp["hour"] = tmp["exit_time"].dt.hour

    # Tabla por celda (weekday-hour)
    g = tmp.groupby(["weekday", "hour"]).agg(
        trades=("tradeRealized", "size"),
        avg=("tradeRealized", "mean"),
        pnl=("tradeRealized", "sum"),
        wins=("tradeRealized", lambda s: (s > 0).sum()),
    ).reset_index()
    g["wr"] = g["wins"] / g["trades"] * 100

    # Filtra celdas con poca muestra
    g.loc[g["trades"] < min_trades, ["avg", "pnl", "wr"]] = np.nan

    pivot = g.pivot(index="weekday", columns="hour", values="avg")

    fig = px.imshow(
        pivot,
        aspect="auto",
        title=f"Heatmap Weekday x Hour → Promedio por trade (NaN = < {min_trades} trades)",
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Lectura: celdas positivas = mejor promedio/trade. Vacías = poca muestra (no concluyente).")


# ============================================================
# Pairing ENTRY+EXIT -> 1 fila por atmId
# ============================================================
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
        "snapN",
        "volLastN", "deltaLastN",
        "openLastN", "highLastN", "lowLastN", "closeLastN"
    ]
    entry_cols = [c for c in entry_cols if c in entries.columns]

    if len(entry_cols) == 0:
        e1 = pd.DataFrame(columns=["atmId", "entry_time"])
    else:
        e1 = entries.sort_values("ts_parsed").groupby("atmId", as_index=False).first()[entry_cols]
        e1 = e1.rename(columns={"ts_parsed": "entry_time"})

    exit_cols = [
        "atmId", "ts_parsed",
        "tradeRealized", "dayRealized", "maxUnreal", "minUnreal",
        "exitReason", "forcedCloseReason", "outcome",
        "tickSize", "pointValue", "qtyTP1", "qtyRunner", "slTicks", "avgEntry",
        "instrument",
        "deltaBar", "deltaPrevBar", "cvd"
    ]
    exit_cols = [c for c in exit_cols if c in exits.columns]
    if len(exit_cols) == 0:
        x1 = pd.DataFrame(columns=["atmId", "exit_time", "tradeRealized"])
    else:
        x1 = exits.sort_values("ts_parsed").groupby("atmId", as_index=False).last()[exit_cols]
        x1 = x1.rename(columns={"ts_parsed": "exit_time"})

    t = pd.merge(x1, e1, on="atmId", how="left")

    for c in [
        "tradeRealized", "dayRealized", "maxUnreal", "minUnreal",
        "orSize", "atr", "ewo", "deltaRatio", "dir",
        "atrSlMult", "tp1R", "tp2R", "trailStepTicks"
    ]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")


    # ------------------------------------------------------------
    # Métricas base de tuning (si hay datos suficientes en el log)
    # ------------------------------------------------------------
    for c in ["tickSize", "pointValue", "slTicks", "qtyTP1", "qtyRunner", "avgEntry",
              "maxUnreal", "minUnreal", "volEntrySoFar", "deltaEntrySoFar"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    # Valor de tick y riesgo estimado ($) usando SL ticks y qty total
    if all(c in t.columns for c in ["tickSize", "pointValue"]):
        t["tick_value"] = t["tickSize"] * t["pointValue"]
    else:
        t["tick_value"] = np.nan

    if "qtyTP1" in t.columns or "qtyRunner" in t.columns:
        q1 = t["qtyTP1"] if "qtyTP1" in t.columns else 0
        qr = t["qtyRunner"] if "qtyRunner" in t.columns else 0
        t["qty_total"] = pd.to_numeric(q1, errors="coerce").fillna(0) + pd.to_numeric(qr, errors="coerce").fillna(0)
    else:
        t["qty_total"] = np.nan

    if all(c in t.columns for c in ["slTicks", "tick_value", "qty_total"]):
        t["risk_$"] = t["slTicks"] * t["tick_value"] * t["qty_total"]
        t.loc[t["risk_$"] <= 0, "risk_$"] = np.nan
    else:
        t["risk_$"] = np.nan

    if "tradeRealized" in t.columns:
        t["ganancia_vs_riesgo"] = t["tradeRealized"] / t["risk_$"]
    else:
        t["ganancia_vs_riesgo"] = np.nan

    # Cuánto capturé / devolví (solo ganadores con maxUnreal válido)
    if all(c in t.columns for c in ["tradeRealized", "maxUnreal"]):
        winners = t["tradeRealized"] > 0
        t["capture_pct"] = np.where(winners & (t["maxUnreal"] > 0), t["tradeRealized"] / t["maxUnreal"], np.nan)
        t["giveback_pct"] = np.where(winners & (t["maxUnreal"] > 0),
                                     (t["maxUnreal"] - t["tradeRealized"]) / t["maxUnreal"], np.nan)
    else:
        t["capture_pct"] = np.nan
        t["giveback_pct"] = np.nan

    # Presión / actividad (usa arrays LastN si existen)
    t = add_pressure_activity_features(t)

    t["has_entry"] = t.get("entry_time", pd.NaT).notna()

    # outcome robusto (si falta 'outcome' en EXIT, lo calculamos por PnL)
    calc_outcome = np.where(t["tradeRealized"].fillna(0) >= 0, "WIN", "LOSS")
    if "outcome" not in t.columns:
        t["outcome"] = calc_outcome
    else:
        t["outcome"] = t["outcome"].where(t["outcome"].notna(), calc_outcome)

    # duration
    if "entry_time" in t.columns and "exit_time" in t.columns:
        t["duration_sec"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds()
    else:
        t["duration_sec"] = np.nan

    t = t.sort_values("exit_time").reset_index(drop=True)

    # equity & drawdown
    t["equity"] = t["tradeRealized"].fillna(0).cumsum()
    t["equity_peak"] = t["equity"].cummax()
    t["drawdown"] = t["equity"] - t["equity_peak"]

    # exit hour label
    t["exit_hour"] = t["exit_time"].dt.hour
    t["exit_hour_label"] = t["exit_hour"].apply(hour_bucket_label)

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

    return t


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

# Load all
all_recs = []
bad_total = 0
for uf in uploaded:
    recs, bad = parse_jsonl_bytes(uf.getvalue())
    bad_total += bad
    all_recs.extend(recs)

df = pd.DataFrame(all_recs)
df = normalize_columns(df)

if bad_total > 0:
    st.warning(f"Se ignoraron {bad_total} líneas inválidas (JSON roto o incompleto).")

t = pair_trades(df)

if t.empty:
    st.error("No se encontraron trades (ENTRY/EXIT) en los archivos cargados.")
    st.stop()

# Missing ENTRY warning
missing_entry = int((~t["has_entry"]).sum())
if missing_entry > 0:
    st.warning(
        f"⚠️ **{missing_entry} operaciones no tienen ENTRY** en los archivos cargados. "
        "En esas, no se puede saber Compra/Venta ni ORSize/ATR/EWO/DeltaRatio. "
        "Se muestran como: “Sin datos (faltó ENTRY)”."
    )

# Sidebar controls
st.sidebar.subheader("⚙️ Ajustes")
min_trades = st.sidebar.slider("Mínimo trades por grupo (para mostrar métricas)", 5, 120, 30, 5)
recommended_trades = st.sidebar.slider("Trades recomendados por grupo (para decidir)", 20, 200, 60, 10)
q_bins = st.sidebar.slider("Número de rangos (bins por cuantiles)", 3, 12, 5, 1)

st.sidebar.markdown("**Presión / Actividad (pre-entrada)**")
activity_low = st.sidebar.slider("Actividad baja (relativa)", 0.20, 1.50, 0.70, 0.05)
activity_high = st.sidebar.slider("Actividad alta (relativa)", 0.50, 3.00, 1.30, 0.05)

show_adv_scatter = st.sidebar.checkbox("Mostrar scatters (modo avanzado)", value=True)
trend_mode = st.sidebar.selectbox(
    "Línea de tendencia (scatter)",
    ["Ninguna", "Regresión (OLS)", "Suavizado (LOWESS)"],
    index=0
)
lowess_frac = st.sidebar.slider("LOWESS suavidad (solo si LOWESS)", 0.05, 0.60, 0.25, 0.05)
last_n_scatter = st.sidebar.slider("Scatters: últimos N trades (0=todo)", 0, 3000, 800, 100)

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
    st.write("**Promedio por trade (Expectancia)**: lo que ganas/pierdes en promedio por operación.")
    st.write("**Profit Factor**: suma de ganancias / suma de pérdidas (ideal > 1.2; excelente > 1.5).")
    st.write("**Max Drawdown**: peor caída desde un pico de equity (si es muy grande, revisa filtro/SL/horarios).")
    st.write("⚠️ **Importante:** si la muestra es pequeña, cualquier número puede engañar. Usa el slider de mínimos.")

st.markdown("---")

# ============================================================
# Time filters
# ============================================================
st.subheader("⏱️ Filtros de tiempo (exit_time)")

min_dt = t["exit_time"].min()
max_dt = t["exit_time"].max()
colA, colB = st.columns(2)
date_from = colA.date_input("Desde", value=min_dt.date() if pd.notna(min_dt) else None)
date_to = colB.date_input("Hasta", value=max_dt.date() if pd.notna(max_dt) else None)

mask = (t["exit_time"].dt.date >= date_from) & (t["exit_time"].dt.date <= date_to)
t_f = t.loc[mask].copy()

summary_f = summarize(t_f)

st.info(
    f"Rango seleccionado: **{date_from}** → **{date_to}** | "
    f"Trades={summary_f['n']} | PF={fmt(summary_f['pf'],2)} | "
    f"Expectancia={fmt(summary_f['expectancy'],1)} | PnL={fmt(summary_f['pnl_total'],0)}"
)

# ============================================================
# Charts quick
# ============================================================
st.subheader("📈 PnL y drawdown")

cL, cR = st.columns([1, 1])
with cL:
    fig_hist = plot_hist_pnl(t_f)
    st.plotly_chart(fig_hist, use_container_width=True)

with cR:
    fig_eq = px.line(t_f, x="exit_time", y="equity", title="Equity (cumsum PnL)")
    fig_eq.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_eq, use_container_width=True)

    fig_dd = px.area(t_f, x="exit_time", y="drawdown", title="Drawdown (equity - peak)")
    fig_dd.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_dd, use_container_width=True)

# ============================================================
# Hour analysis
# ============================================================
st.subheader("🕒 Horas (exit_time)")

plot_hour_analysis(t_f, min_trades=min_trades)

with st.expander("🗓️ Heatmap Weekday x Hour", expanded=False):
    plot_heatmap_weekday_hour(t_f, min_trades=min_trades)

# ============================================================
# Known (tiene ENTRY)
# ============================================================
known = t_f[t_f["has_entry"]].copy()

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["OR Size", "ATR", "EWO", "DeltaRatio", "Presión y actividad"])

    with tab1:
        if "orSize" in df_known.columns and df_known["orSize"].notna().sum() > 30:
            plot_factor_bins(df_known, "orSize", q_bins, min_trades, "OR Size",
                             show_adv_scatter, trend_mode, lowess_frac, df_scatter)
        else:
            st.info("No hay suficiente data de OR Size.")

    with tab2:
        if "atr" in df_known.columns and df_known["atr"].notna().sum() > 30:
            plot_factor_bins(df_known, "atr", q_bins, min_trades, "ATR",
                             show_adv_scatter, trend_mode, lowess_frac, df_scatter)
        else:
            st.info("No hay suficiente data de ATR.")

    with tab3:
        if "ewo" in df_known.columns and df_known["ewo"].notna().sum() > 30:
            plot_factor_bins(df_known, "ewo", q_bins, min_trades, "EWO",
                             show_adv_scatter, trend_mode, lowess_frac, df_scatter)
        else:
            st.info("No hay suficiente data de EWO.")

    with tab4:
        if "deltaRatio" in df_known.columns and df_known["deltaRatio"].notna().sum() > 30:
            plot_factor_bins(df_known, "deltaRatio", q_bins, min_trades, "Agresión neta (delta/vol)",
                             show_adv_scatter, trend_mode, lowess_frac, df_scatter)
        else:
            st.info("No hay suficiente data de DeltaRatio.")

    with tab5:
        st.write("Estas métricas usan el **contexto pre-entrada** guardado en arrays (LastN): volumen, delta y OHLC.")
        cols_need = ["pre_absorption", "pre_activity_rel", "pre_pressure_per_vol", "pre_pressure_sum", "pre_pressure_abs", "pre_move_ticks", "pre_price_net"]
        have = [c for c in cols_need if c in df_known.columns]
        if len(have) < 3 or ("pre_absorption" in df_known.columns and df_known["pre_absorption"].notna().sum() < 20):
            st.info("No hay suficiente data de arrays (vol/delta/OHLC) para este panel.")
        else:
            if "pre_absorption" in df_known.columns and df_known["pre_absorption"].notna().sum() >= max(30, min_trades):
                plot_factor_bins(df_known, "pre_absorption", q_bins, min_trades, "Absorción (presión alta, avance bajo)",
                                 show_adv_scatter, trend_mode, lowess_frac, df_scatter)
            else:
                st.info("Absorción: falta muestra.")

            if "pre_activity_rel" in df_known.columns and df_known["pre_activity_rel"].notna().sum() >= max(30, min_trades):
                plot_factor_bins(df_known, "pre_activity_rel", q_bins, min_trades, "Actividad relativa (volumen hasta entrada vs reciente)",
                                 show_adv_scatter, trend_mode, lowess_frac, df_scatter)
            else:
                st.info("Actividad relativa: falta muestra.")

            if "pre_pressure_per_vol" in df_known.columns and df_known["pre_pressure_per_vol"].notna().sum() >= max(30, min_trades):
                plot_factor_bins(df_known, "pre_pressure_per_vol", q_bins, min_trades, "Presión por volumen (delta abs / volumen)",
                                 show_adv_scatter, trend_mode, lowess_frac, df_scatter)
            else:
                st.info("Presión por volumen: falta muestra.")

            tmp = df_known.copy()
            p_thr = tmp["pre_pressure_abs"].quantile(0.75) if "pre_pressure_abs" in tmp.columns else np.nan
            m_thr = tmp["pre_move_ticks"].quantile(0.25) if "pre_move_ticks" in tmp.columns else np.nan

            def classify(r):
                if r.get("pre_divergence") is True:
                    return "Sin apoyo (divergencia)"
                if (r.get("pre_pressure_abs") == r.get("pre_pressure_abs")) and (r.get("pre_move_ticks") == r.get("pre_move_ticks")):
                    if r["pre_pressure_abs"] >= p_thr and r["pre_move_ticks"] <= m_thr:
                        return "Absorción"
                d = r.get("dir")
                if d == d and r.get("pre_pressure_sum") == r.get("pre_pressure_sum") and r.get("pre_price_net") == r.get("pre_price_net"):
                    ar = r.get("pre_activity_rel")
                    ar_ok = (ar == ar) and (ar >= 1.0)
                    if d > 0 and r["pre_pressure_sum"] > 0 and r["pre_price_net"] > 0 and ar_ok:
                        return "Impulso limpio"
                    if d < 0 and r["pre_pressure_sum"] < 0 and r["pre_price_net"] < 0 and ar_ok:
                        return "Impulso limpio"
                return "Mixto"

            tmp["Estado pre-entrada"] = tmp.apply(classify, axis=1)

            st.subheader("📋 Estado pre-entrada (impulso / absorción / sin apoyo)")
            tbl_state = group_metrics(tmp, "Estado pre-entrada", min_trades=min_trades)
            st.dataframe(tbl_state, use_container_width=True)

            # Actividad (baja/normal/alta) según umbrales del sidebar
            if "pre_activity_rel" in tmp.columns:
                tmp["Actividad"] = np.select(
                    [
                        tmp["pre_activity_rel"] < activity_low,
                        tmp["pre_activity_rel"] > activity_high,
                    ],
                    [
                        "Baja actividad",
                        "Alta actividad",
                    ],
                    default="Actividad normal"
                )
                st.subheader("📋 Actividad (según volumen relativo)")
                tbl_act = group_metrics(tmp, "Actividad", min_trades=min_trades)
                st.dataframe(tbl_act, use_container_width=True)

            # Divergencia simple
            if "pre_divergence" in tmp.columns:
                n_div = int(tmp["pre_divergence"].sum())
                st.caption(f"⚠️ Divergencia (precio avanza sin apoyo de delta): {n_div} trades en la muestra.")

            st.caption("⚠️ Nota: esto describe el mercado **antes** de la entrada (no lo que pasó durante el trade).")

# ============================================================
with st.expander("📄 Tabla de trades (una fila por atmId)", expanded=False):
    cols_show = [c for c in [
        "exit_time", "entry_time", "lado", "outcome", "tradeRealized",
        "risk_$", "ganancia_vs_riesgo", "capture_pct", "giveback_pct",
        "maxUnreal", "minUnreal", "exitReason", "forcedCloseReason",
        "orSize", "ewo", "atr", "deltaRatio", "atrSlMult", "tp1R", "tp2R",
        "pre_absorption", "pre_activity_rel", "pre_pressure_per_vol", "pre_divergence",
        "duration_sec"
    ] if c in t.columns]

    df_show = t[cols_show].sort_values("exit_time", ascending=False).copy()
    df_show = df_show.rename(columns={
        "tradeRealized": "PnL",
        "risk_$": "Riesgo estimado ($)",
        "ganancia_vs_riesgo": "Ganancia/Riesgo (R)",
        "capture_pct": "% capturado (solo wins)",
        "giveback_pct": "% devuelto (solo wins)",
        "deltaRatio": "Agresión neta (delta/vol)",
        "pre_absorption": "Absorción (pre)",
        "pre_activity_rel": "Actividad relativa (pre)",
        "pre_pressure_per_vol": "Presión/Vol (pre)",
        "pre_divergence": "Divergencia (pre)",
        "duration_sec": "Duración (s)",
    })
    st.dataframe(df_show, use_container_width=True)

st.caption(
    "Tip: para eliminar “Sin datos (faltó ENTRY)”, asegúrate de cargar meses que contengan los ENTRY y EXIT juntos. "
    "Si quieres 100% robustez, añade `dir` también en el EXIT (en NinjaScript) o guarda logs en bloques por trade."
)
