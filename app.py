import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go

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
# Helpers
# ============================================================
def parse_jsonl_bytes(b: bytes):
    txt = b.decode("utf-8", errors="replace")
    recs = []
    bad = 0
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

    # Compat keys (por si logs viejos cambiaron nombres)
    rename_map = {
        "slAtrMult": "atrSlMult",
        "tpR": "tp1R",
        "useAtrOverride": "useAtrEngine",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "type" not in df.columns:
        df["type"] = ""

    return df


def profit_factor(trades: pd.DataFrame) -> float:
    wins = trades.loc[trades["tradeRealized"] > 0, "tradeRealized"].sum()
    losses = trades.loc[trades["tradeRealized"] < 0, "tradeRealized"].sum()
    if losses == 0:
        return np.nan
    return float(wins / abs(losses))


def max_streak(outcomes: pd.Series, target: str):
    best_len = 0
    cur = 0
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


def pair_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 fila por trade (atmId):
    - EXIT: último por atmId
    - ENTRY: primero por atmId
    Si falta ENTRY -> lado = "Sin datos (faltó ENTRY)" (NO inventamos)
    """
    entries = df[df["type"] == "ENTRY"].copy()
    exits = df[df["type"] == "EXIT"].copy()

    # ENTRY (primero por atmId)
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

    # EXIT (último por atmId)
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

    # JOIN
    t = xlast.merge(e1, on="atmId", how="left")

    # Numeric
    for c in [
        "tradeRealized", "dayRealized", "maxUnreal", "minUnreal",
        "orSize", "atr", "ewo", "deltaRatio", "dir",
        "atrSlMult", "tp1R", "tp2R", "trailStepTicks"
    ]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    t["has_entry"] = t.get("entry_time", pd.NaT).notna()

    # outcome robusto
    calc_outcome = pd.Series(np.where(t["tradeRealized"].fillna(0) >= 0, "WIN", "LOSS"), index=t.index)
    if "outcome" not in t.columns:
        t["outcome"] = calc_outcome
    else:
        t["outcome"] = t["outcome"].where(t["outcome"].notna(), calc_outcome)

    # duration
    if "entry_time" in t.columns and "exit_time" in t.columns:
        t["duration_sec"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds()
    else:
        t["duration_sec"] = np.nan

    # sort
    t = t.sort_values("exit_time").reset_index(drop=True)

    # equity & drawdown
    t["equity"] = t["tradeRealized"].fillna(0).cumsum()
    t["equity_peak"] = t["equity"].cummax()
    t["drawdown"] = t["equity"] - t["equity_peak"]

    # time buckets
    t["exit_date"] = pd.to_datetime(t["exit_time"]).dt.date
    t["exit_hour"] = pd.to_datetime(t["exit_time"]).dt.hour
    t["weekday"] = pd.to_datetime(t["exit_time"]).dt.day_name()

    # side
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

    wlen, ws, we = max_streak(t["outcome"], "WIN")
    llen, ls, le = max_streak(t["outcome"], "LOSS")

    return {
        "n": n, "wins": wins, "losses": losses, "win_rate": win_rate,
        "pnl_total": float(t["tradeRealized"].sum()),
        "pf": pf, "expectancy": expectancy,
        "max_dd": max_dd, "dd_peak_time": dd_peak_time, "dd_trough_time": dd_trough_time,
        "max_win": max_win, "max_loss": max_loss,
        "best_win_streak": wlen, "best_loss_streak": llen,
        "best_win_streak_idx": (ws, we),
        "best_loss_streak_idx": (ls, le),
    }


def metric_tag_ok(value, good_when="high"):
    # Pequeña ayuda visual para consejos
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "⚪"
    if good_when == "high":
        return "🟢" if value >= 1.2 else ("🟡" if value >= 1.0 else "🔴")
    else:
        return "🟢" if value <= -500 else ("🟡" if value <= -300 else "🔴")


def make_bins_quantiles(df: pd.DataFrame, col: str, q: int):
    s = df[col].dropna()
    if len(s) < q * 10:
        return None  # poca data
    try:
        bins = pd.qcut(df[col], q=q, duplicates="drop")
        return bins
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
        wr_adj = (wins + 1) / (n + 2) * 100  # smoothing (evita 2 trades=100% “perfecto”)
        pf = profit_factor(sub)
        exp = float(sub["tradeRealized"].mean())
        pnl = float(sub["tradeRealized"].sum())
        # Score ponderado: prioriza exp + tamaño de muestra
        score = exp * np.log1p(n)
        rows.append({
            "Grupo": str(g),
            "Trades": n,
            "WinRate%": wr,
            "WinRate Ajustado%": wr_adj,
            "PF": pf,
            "Expectancia": exp,
            "PnL Total": pnl,
            "Score (ponderado)": score,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Score (ponderado)", ascending=False).reset_index(drop=True)
    return out


def advice_from_table(tbl: pd.DataFrame, title: str, min_trades: int):
    if tbl is None or tbl.empty:
        st.info(f"En **{title}** no hay suficiente muestra (mínimo {min_trades} trades por grupo).")
        return

    best = tbl.iloc[0]
    worst = tbl.iloc[-1]

    st.markdown("**Consejos automáticos (basados en los datos):**")
    st.write(
        f"✅ Mejor grupo: **{best['Grupo']}** | Trades={int(best['Trades'])} | "
        f"PF={best['PF']:.2f} | Exp={best['Expectancia']:.1f} | PnL={best['PnL Total']:.0f}"
    )
    st.write(
        f"⚠️ Peor grupo: **{worst['Grupo']}** | Trades={int(worst['Trades'])} | "
        f"PF={worst['PF']:.2f} | Exp={worst['Expectancia']:.1f} | PnL={worst['PnL Total']:.0f}"
    )

    # Reglas simples para texto
    if not np.isnan(best["PF"]) and best["PF"] < 1.0:
        st.warning("⚠️ Incluso el “mejor” grupo tiene PF < 1.0 → faltan filtros o el sistema no tiene edge con estos datos.")
    if not np.isnan(worst["PF"]) and worst["PF"] < 1.0:
        st.warning("🧨 Hay grupos con PF < 1.0 → considera filtrarlos o ajustar SL/TP/horarios.")
    if best["Trades"] < min_trades * 2:
        st.info("ℹ️ El mejor grupo tiene muestra justa. Ideal: acumular más trades para confirmar.")


def plot_equity_drawdown(t: pd.DataFrame):
    fig1 = px.line(t, x="exit_time", y="equity", title="Equity (curva de capital)")
    fig1.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))

    fig2 = px.line(t, x="exit_time", y="drawdown", title="Drawdown (caída desde el pico)")
    fig2.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
    fig2.add_hline(y=0, line_width=1, line_dash="dash")

    return fig1, fig2


def plot_pnl_hist(t: pd.DataFrame):
    fig = px.histogram(t, x="tradeRealized", nbins=40, title="Distribución de PnL por trade")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    fig.add_vline(x=0, line_width=1, line_dash="dash")
    return fig


def plot_factor_bins(df_known: pd.DataFrame, col: str, q: int, min_trades: int, title: str):
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

    # Gráficos
    fig_exp = px.bar(tbl, x="Grupo", y="Expectancia", title=f"{title} → Expectancia por rango (bins)")
    fig_exp.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    fig_exp.add_hline(y=0, line_width=1, line_dash="dash")

    fig_pf = px.bar(tbl, x="Grupo", y="PF", title=f"{title} → Profit Factor por rango (bins)")
    fig_pf.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    fig_pf.add_hline(y=1.0, line_width=1, line_dash="dash")

    st.plotly_chart(fig_exp, use_container_width=True)
    st.plotly_chart(fig_pf, use_container_width=True)

    st.dataframe(tbl, use_container_width=True)
    advice_from_table(tbl, title=title, min_trades=min_trades)


def plot_scatter_advanced(df_known: pd.DataFrame, xcol: str, title: str):
    # Scatter más legible: verde arriba de 0, rojo abajo
    tmp = df_known[[xcol, "tradeRealized", "exit_time", "lado", "exitReason"]].dropna().copy()
    tmp["Resultado"] = np.where(tmp["tradeRealized"] >= 0, "Ganancia", "Pérdida")

    fig = px.scatter(
        tmp,
        x=xcol,
        y="tradeRealized",
        color="Resultado",
        color_discrete_map={"Ganancia": "green", "Pérdida": "red"},
        hover_data=["exit_time", "lado", "exitReason", "tradeRealized"],
        title=f"{title} (modo avanzado)"
    )
    fig.update_traces(marker=dict(size=6, opacity=0.55))
    fig.add_hline(y=0, line_width=1, line_dash="dash")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Cómo leerlo: si en cierto rango de X hay mayoría de puntos verdes y pocos rojos grandes → ese rango suele ser mejor.")


def plot_hour_analysis(t: pd.DataFrame, min_trades: int):
    # Tabla por hora: usa winrate ajustado + score ponderado (evita 2 trades = 100% engañoso)
    tbl = group_metrics(t, "exit_hour", min_trades=min_trades)
    if tbl.empty:
        st.info(f"No hay suficientes trades por hora para min_trades={min_trades}.")
        return

    fig = px.bar(tbl, x="Grupo", y="Score (ponderado)", title="Horas con mejor score (ponderado por muestra)")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(tbl, use_container_width=True)
    advice_from_table(tbl, title="Hora de salida (exit_hour)", min_trades=min_trades)


def plot_heatmap_weekday_hour(t: pd.DataFrame, min_trades: int):
    # Heatmap de Expectancia por weekday x hour, pero oculta celdas con pocos trades
    tmp = t.copy()
    tmp["weekday_order"] = pd.Categorical(
        tmp["weekday"],
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True
    )

    agg = tmp.groupby(["weekday_order", "exit_hour"]).agg(
        Trades=("tradeRealized", "size"),
        Expectancia=("tradeRealized", "mean"),
        PnL=("tradeRealized", "sum"),
    ).reset_index()

    agg.loc[agg["Trades"] < min_trades, "Expectancia"] = np.nan

    pivot = agg.pivot(index="weekday_order", columns="exit_hour", values="Expectancia")
    fig = px.imshow(
        pivot,
        aspect="auto",
        title=f"Heatmap: Expectancia por Día x Hora (solo celdas con ≥ {min_trades} trades)",
        origin="lower"
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Cómo leerlo: celdas más altas (más positivas) = mejores combinaciones Día/Hora. Celdas vacías = poca muestra.")


# ============================================================
# UI
# ============================================================
st.title("📊 WIC_WLF2 Analizador (Plotly)")

uploaded = st.file_uploader(
    "📤 Sube uno o varios archivos .jsonl (meses)",
    type=["jsonl"],
    accept_multiple_files=True
)

if not uploaded:
    st.stop()

all_records = []
bad_total = 0
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
        f"⚠️ **{missing_entry} trades no tienen ENTRY** en los archivos cargados. "
        "Para esos trades, no podemos saber Compra/Venta y aparecerán como “Sin datos (faltó ENTRY)”."
    )

summary = summarize(t)

# Settings sidebar
st.sidebar.subheader("⚙️ Ajustes del análisis")
min_trades = st.sidebar.slider("Mínimo trades por grupo (para considerar confiable)", 5, 80, 30, 5)
q_bins = st.sidebar.slider("Número de rangos (bins) por cuantiles", 3, 10, 5, 1)
show_adv_scatter = st.sidebar.checkbox("Mostrar scatters (modo avanzado)", value=False)
last_n_scatter = st.sidebar.slider("Si scatters: mostrar solo últimos N trades (0=todo)", 0, 2000, 800, 100)

# Summary metrics
st.subheader("✅ Resumen rápido (lo más importante)")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Operaciones", f"{summary['n']}")
c2.metric("Ganadas", f"{summary['wins']}")
c3.metric("Perdidas", f"{summary['losses']}")
c4.metric("% Acierto", f"{summary['win_rate']:.1f}%")
c5.metric("PnL Total", f"{summary['pnl_total']:.0f}")
c6.metric("Profit Factor", f"{summary['pf']:.2f}" if not np.isnan(summary["pf"]) else "N/A")

c7, c8, c9, c10, c11, c12 = st.columns(6)
c7.metric("Expectancia", f"{summary['expectancy']:.1f}")
c8.metric("Max Drawdown", f"{summary['max_dd']:.0f}")
c9.metric("Mejor racha (wins)", f"{summary['best_win_streak']}")
c10.metric("Peor racha (losses)", f"{summary['best_loss_streak']}")
c11.metric("Mayor win", f"{summary['max_win']:.1f}")
c12.metric("Mayor loss", f"{summary['max_loss']:.1f}")

with st.expander("🧠 Consejos globales (basados en métricas)", expanded=True):
    pf = summary["pf"]
    exp = summary["expectancy"]
    dd = summary["max_dd"]

    if not np.isnan(pf):
        tag = metric_tag_ok(pf, "high")
        if pf < 1.0:
            st.warning(f"{tag} Profit Factor {pf:.2f} → a largo plazo este set tiende a perder. Requiere filtros/ajustes.")
        elif pf < 1.2:
            st.info(f"{tag} Profit Factor {pf:.2f} → ligera ventaja, pero frágil. Prioriza control de riesgo y filtros.")
        else:
            st.success(f"{tag} Profit Factor {pf:.2f} → ventaja estadística (edge) más sólida en estos datos.")

    if exp < 0:
        st.warning(f"🔴 Expectancia {exp:.1f} → promedio negativo. Ajusta filtros (OR/ATR/EWO/DeltaRatio) u horarios.")
    else:
        st.info(f"🟢 Expectancia {exp:.1f} → promedio positivo. Ahora enfócate en reducir drawdown y rachas.")

    if not np.isnan(dd) and dd < 0:
        st.info(f"ℹ️ Max Drawdown {dd:.0f} → úsalo para guardias diarias y sizing. Si duele demasiado, filtra peores grupos.")


# Main charts
st.subheader("📈 Gráficos principales (claros)")

fig_eq, fig_dd = plot_equity_drawdown(t)
colA, colB = st.columns(2)
with colA:
    st.plotly_chart(fig_eq, use_container_width=True)
with colB:
    st.plotly_chart(fig_dd, use_container_width=True)

st.plotly_chart(plot_pnl_hist(t), use_container_width=True)

# Side summary
st.subheader("🧭 Compra vs Venta (solo donde hay ENTRY)")
col1, col2, col3 = st.columns(3)
col1.metric("Compras (Long)", int((t["lado"] == "Compra (Long)").sum()))
col2.metric("Ventas (Short)", int((t["lado"] == "Venta (Short)").sum()))
col3.metric("Sin datos (faltó ENTRY)", int((t["lado"].str.startswith("Sin datos")).sum()))

known = t[t["lado"].isin(["Compra (Long)", "Venta (Short)"])].copy()
if known.empty:
    st.info("No hay suficientes trades con ENTRY para separar Compra/Venta. (Se arregla logueando dir en EXIT).")
else:
    side_tbl = group_metrics(known, "lado", min_trades=max(5, min_trades // 2))
    st.dataframe(side_tbl, use_container_width=True)
    advice_from_table(side_tbl, "Compra/Venta", max(5, min_trades // 2))


# ============================================================
# Tuning: ORSize / ATR / EWO / DeltaRatio
# ============================================================
st.subheader("🛠️ Ajuste de filtros (lo que realmente sirve para tunear)")

if known.empty:
    st.info("Estos análisis necesitan ENTRY (para tener ORSize/ATR/EWO/DeltaRatio por trade).")
else:
    # Reduce ruido si user quiere
    df_known = known.copy()
    if last_n_scatter and last_n_scatter > 0:
        df_known = df_known.sort_values("exit_time").tail(last_n_scatter)

    tab1, tab2, tab3, tab4 = st.tabs(["OR Size", "ATR", "EWO", "DeltaRatio"])

    with tab1:
        if "orSize" in known.columns and known["orSize"].notna().sum() > 30:
            plot_factor_bins(known, "orSize", q_bins, min_trades, "OR Size")
            if show_adv_scatter:
                plot_scatter_advanced(df_known, "orSize", "OR Size vs PnL")
        else:
            st.info("No hay suficientes valores de OR Size en los logs.")

    with tab2:
        if "atr" in known.columns and known["atr"].notna().sum() > 30:
            plot_factor_bins(known, "atr", q_bins, min_trades, "ATR")
            if show_adv_scatter:
                plot_scatter_advanced(df_known, "atr", "ATR vs PnL")
        else:
            st.info("No hay suficientes valores de ATR en los logs.")

    with tab3:
        if "ewo" in known.columns and known["ewo"].notna().sum() > 30:
            # usamos magnitud abs(ewo) porque suele ser lo útil
            known2 = known.copy()
            known2["ewo_abs"] = known2["ewo"].abs()
            plot_factor_bins(known2, "ewo_abs", q_bins, min_trades, "EWO (magnitud |abs|)")
            if show_adv_scatter:
                plot_scatter_advanced(df_known.assign(ewo_abs=df_known["ewo"].abs()), "ewo_abs", "EWO |abs| vs PnL")
        else:
            st.info("No hay suficientes valores de EWO en los logs.")

    with tab4:
        if "deltaRatio" in known.columns and known["deltaRatio"].notna().sum() > 30:
            plot_factor_bins(known, "deltaRatio", q_bins, min_trades, "DeltaRatio")
            if show_adv_scatter:
                plot_scatter_advanced(df_known, "deltaRatio", "DeltaRatio vs PnL")
        else:
            st.info("No hay suficientes valores de DeltaRatio en los logs.")


# ============================================================
# Time analysis (fair): Hour + Heatmap weekday/hour
# ============================================================
st.subheader("⏰ Horarios (justo y confiable)")

plot_hour_analysis(t, min_trades=min_trades)
plot_heatmap_weekday_hour(t, min_trades=min_trades)

with st.expander("📄 Tabla de trades (una fila por atmId)", expanded=False):
    cols_show = [c for c in [
        "exit_time", "entry_time", "lado", "outcome", "tradeRealized",
        "maxUnreal", "minUnreal", "exitReason", "forcedCloseReason",
        "orSize", "ewo", "atr", "deltaRatio", "atrSlMult", "tp1R", "tp2R"
    ] if c in t.columns]
    st.dataframe(t[cols_show].sort_values("exit_time", ascending=False), use_container_width=True)

st.caption(
    "Tip: para eliminar totalmente “Sin datos (faltó ENTRY)”, lo ideal es que el NinjaScript escriba `dir` también dentro del `EXIT`."
)
