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
        "template", "orderType", "trigger",
        "orHigh", "orLow", "orSize",
        "ewo", "atr", "useAtrEngine", "atrSlMult",
        "tp1R", "tp2R", "tsBehindTP1Atr", "trailStepTicks",
        "deltaRatio", "dailyPnL"
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
        "maxUnreal", "minUnreal", "forcedCloseReason", "dailyHalt"
    ]
    exit_cols = [c for c in exit_cols if c in exits.columns]

    xlast = (exits.sort_values("ts_parsed")
                  .groupby("atmId", as_index=False)[exit_cols]
                  .last()
                  .rename(columns={"ts_parsed": "exit_time"}))

    t = xlast.merge(e1, on="atmId", how="left")

    for c in [
        "tradeRealized", "dayRealized", "maxUnreal", "minUnreal",
        "orSize", "atr", "ewo", "deltaRatio", "dir",
        "atrSlMult", "tp1R", "tp2R", "trailStepTicks"
    ]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

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

    # time buckets
    t["exit_date"] = pd.to_datetime(t["exit_time"]).dt.date
    t["exit_hour"] = pd.to_datetime(t["exit_time"]).dt.hour
    t["exit_hour_label"] = t["exit_hour"].apply(hour_bucket_label)

    # weekday (keep english internal, easier for ordering)
    t["weekday"] = pd.to_datetime(t["exit_time"]).dt.day_name()

    # side (Compra/Venta)
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

    # Backward-compat aliases (older code may expect *_ts)
    if "exit_time" in t.columns and "exit_ts" not in t.columns:
        t["exit_ts"] = t["exit_time"]
    if "entry_time" in t.columns and "entry_ts" not in t.columns:
        t["entry_ts"] = t["entry_time"]

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


def group_metrics(df: pd.DataFrame, group_col: str, min_trades: int):
    rows = []
    for g, sub in df.groupby(group_col):
        n = len(sub)
        if n < min_trades:
            continue
        wins = int((sub["tradeRealized"] > 0).sum())
        wr = wins / n * 100
        wr_adj = (wins + 1) / (n + 2) * 100  # suavizado
        pf = profit_factor(sub)
        exp = float(sub["tradeRealized"].mean())
        pnl = float(sub["tradeRealized"].sum())
        score = exp * np.log1p(n)  # pondera tamaño de muestra
        rows.append({
            "Grupo": str(g),
            "Trades": n,
            "WinRate%": wr,
            "WinRate Ajustado%": wr_adj,
            "Profit Factor": pf,
            "Promedio por trade": exp,
            "PnL Total": pnl,
            "Score (ponderado)": score,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Score (ponderado)", ascending=False).reset_index(drop=True)
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
        return ["ℹ️ No hay suficiente data para detectar zonas por bins."]

    tmp = df_known.copy()
    tmp["_bin"] = bins.astype(str)
    tbl = group_metrics(tmp, "_bin", min_trades=min_trades)

    if tbl.empty:
        return [f"ℹ️ No hay bins con ≥ {min_trades} trades para {title}."]

    # zonas malas
    bad = tbl[(tbl["Promedio por trade"] < 0) & (tbl["Profit Factor"] < 1.0)].copy()
    good = tbl[(tbl["Promedio por trade"] > 0) & (tbl["Profit Factor"] >= 1.2)].copy()

    insights = []
    if not bad.empty:
        b = bad.sort_values("Score (ponderado)").iloc[0]
        insights.append(f"🚫 Zona peligrosa: **{b['Grupo']}** → promedio<0 y PF<1 (candidato a EVITAR).")
    else:
        insights.append("✅ No se detectan bins claramente peligrosos (con muestra suficiente).")

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
        hover_data=["exit_time", "lado", "exitReason", "tradeRealized"],
        title=f"{title}"
    )
    fig.update_traces(marker=dict(size=6, opacity=0.55))
    fig.add_hline(y=0, line_width=1, line_dash="dash")

    # Trendline
    if sm is not None and trend_mode != "Ninguna":
        x = pd.to_numeric(tmp[xcol], errors="coerce")
        y = pd.to_numeric(tmp["tradeRealized"], errors="coerce")
        m = (~x.isna()) & (~y.isna())
        x = x[m].values
        y = y[m].values

        if trend_mode == "Regresión (OLS)" and len(x) >= 10:
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            xs = np.linspace(np.min(x), np.max(x), 140)
            ys = model.predict(sm.add_constant(xs))

            line = pd.DataFrame({xcol: xs, "tradeRealized": ys})
            fig2 = px.line(line, x=xcol, y="tradeRealized")
            fig2.update_traces(line=dict(width=3), showlegend=False)
            for tr in fig2.data:
                fig.add_trace(tr)

            r2 = model.rsquared
            fig.add_annotation(
                text=f"OLS: R²={r2:.3f}",
                xref="paper", yref="paper", x=0.01, y=0.98,
                showarrow=False
            )

        if trend_mode == "Suavizado (LOWESS)" and len(x) >= 25:
            z = sm.nonparametric.lowess(y, x, frac=float(lowess_frac), return_sorted=True)
            line = pd.DataFrame({xcol: z[:, 0], "tradeRealized": z[:, 1]})
            fig2 = px.line(line, x=xcol, y="tradeRealized")
            fig2.update_traces(line=dict(width=3), showlegend=False)
            for tr in fig2.data:
                fig.add_trace(tr)

            fig.add_annotation(
                text=f"LOWESS (suavidad={lowess_frac})",
                xref="paper", yref="paper", x=0.01, y=0.98,
                showarrow=False
            )

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
        st.info(f"En **{title}** no hay bins con mínimo {min_trades} trades.")
        return

    fig_exp = px.bar(tbl, x="Grupo", y="Promedio por trade", title=f"{title} → Promedio por trade (bins)")
    fig_exp.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    fig_exp.add_hline(y=0, line_width=1, line_dash="dash")

    fig_pf = px.bar(tbl, x="Grupo", y="Profit Factor", title=f"{title} → Profit Factor (bins)")
    fig_pf.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    fig_pf.add_hline(y=1.0, line_width=1, line_dash="dash")

    st.plotly_chart(fig_exp, use_container_width=True)
    st.plotly_chart(fig_pf, use_container_width=True)

    st.dataframe(tbl, use_container_width=True)
    advice_from_table(tbl, title=title, min_trades=min_trades)

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
        plot_scatter_advanced(scatter_df, col, f"{title}: Scatter PnL vs {col}",
                              trend_mode=trend_mode, lowess_frac=lowess_frac)


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
    advice_from_table(tbl, title="Hora (bucket)", min_trades=min_trades)
    return tbl


def plot_heatmap_weekday_hour(t: pd.DataFrame, min_trades: int):
    tmp = t.copy()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    tmp["weekday_order"] = pd.Categorical(tmp["weekday"], categories=weekday_order, ordered=True)

    agg = tmp.groupby(["weekday_order", "exit_hour"]).agg(
        Trades=("tradeRealized", "size"),
        Promedio=("tradeRealized", "mean"),
    ).reset_index()

    agg.loc[agg["Trades"] < min_trades, "Promedio"] = np.nan
    pivot = agg.pivot(index="weekday_order", columns="exit_hour", values="Promedio")

    fig = px.imshow(
        pivot,
        aspect="auto",
        title=f"Heatmap: Promedio por trade (Día x Hora) | solo celdas con ≥ {min_trades} trades",
        origin="lower"
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lectura: celdas positivas = mejor promedio/trade. Vacías = poca muestra (no concluyente).")


# ============================================================
# UI
# ============================================================
st.title("📊 WIC_WLF2 Analizador (Next-level, Español, tuning-friendly)")

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
min_trades = st.sidebar.slider("Mínimo trades por grupo (confiable)", 5, 120, 30, 5)
q_bins = st.sidebar.slider("Número de rangos (bins por cuantiles)", 3, 12, 5, 1)

show_adv_scatter = st.sidebar.checkbox("Mostrar scatters (modo avanzado)", value=True)
trend_mode = st.sidebar.selectbox(
    "Línea de tendencia (scatter)",
    ["Ninguna", "Suavizado (LOWESS)"],
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
    st.write("**Promedio por trade (Expectancia)**: lo que ganas/pierdes en promedio por operación. Si es positivo, bien.")
    st.write("**Profit Factor**: ganancias totales / pérdidas totales. PF > 1.0 indica ventaja. PF > 1.2 suele ser más sólido.")
    st.write("**Drawdown**: la peor caída desde el máximo de tu equity; representa el “dolor máximo” del sistema.")
    st.write("**Rachas**: cuántas operaciones ganadas/perdidas seguidas (útil para guardias diarias y sizing).")

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

    tab1, tab2, tab3, tab4 = st.tabs(["OR Size", "ATR", "EWO", "DeltaRatio"])

    with tab1:
        if "orSize" in df_known.columns and df_known["orSize"].notna().sum() > 30:
            plot_factor_bins(df_known, "orSize", q_bins, min_trades, "OR Size",
                             show_adv_scatter, trend_mode, lowess_frac, df_scatter)
        else:
            st.info("No hay suficientes valores de OR Size en los logs.")

    with tab2:
        if "atr" in df_known.columns and df_known["atr"].notna().sum() > 30:
            plot_factor_bins(df_known, "atr", q_bins, min_trades, "ATR",
                             show_adv_scatter, trend_mode, lowess_frac, df_scatter)
        else:
            st.info("No hay suficientes valores de ATR en los logs.")

    with tab3:
        if "ewo" in df_known.columns and df_known["ewo"].notna().sum() > 30:
            df_known2 = df_known.copy()
            df_known2["ewo_abs"] = df_known2["ewo"].abs()

            df_scatter2 = df_scatter.copy()
            df_scatter2["ewo_abs"] = df_scatter2["ewo"].abs()

            plot_factor_bins(df_known2, "ewo_abs", q_bins, min_trades, "EWO (magnitud |abs|)",
                             show_adv_scatter, trend_mode, lowess_frac, df_scatter2)
        else:
            st.info("No hay suficientes valores de EWO en los logs.")

    with tab4:
        if "deltaRatio" in df_known.columns and df_known["deltaRatio"].notna().sum() > 30:
            plot_factor_bins(df_known, "deltaRatio", q_bins, min_trades, "DeltaRatio",
                             show_adv_scatter, trend_mode, lowess_frac, df_scatter)
        else:
            st.info("No hay suficientes valores de DeltaRatio en los logs.")

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
