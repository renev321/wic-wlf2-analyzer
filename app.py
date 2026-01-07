# app.py — WIC_WLF2 Analizador JSONL (Streamlit)
# UI en español, pensado para “lectura fácil” (equipo / no-tech).
#
# Requisitos: streamlit, pandas, numpy
# Ejecutar: streamlit run app.py

import os
import json
import math
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -------------------------
# Seguridad simple (opcional)
# -------------------------
def auth_gate():
    """
    Si defines APP_PASSWORD como variable de entorno en Streamlit Cloud,
    pedirá una contraseña. Si no existe, la app queda pública (sin login).
    """
    pwd = os.getenv("APP_PASSWORD", "").strip()
    if not pwd:
        return True

    st.sidebar.markdown("### 🔐 Acceso")
    user_in = st.sidebar.text_input(
        "Usuario (solo informativo)",
        value="",
        help="No se valida. Solo para que sepan quién entró.",
    )
    pass_in = st.sidebar.text_input("Contraseña", type="password")
    if st.sidebar.button("Entrar"):
        if pass_in == pwd:
            st.session_state["auth_ok"] = True
            st.session_state["auth_user"] = user_in or "usuario"
        else:
            st.sidebar.error("Contraseña incorrecta.")
            st.session_state["auth_ok"] = False

    if st.session_state.get("auth_ok"):
        st.sidebar.success(f"Sesión: {st.session_state.get('auth_user','usuario')}")
        return True

    st.stop()


# -------------------------
# Lectura JSONL
# -------------------------
def parse_jsonl_bytes(file_bytes: bytes) -> Tuple[List[Dict[str, Any]], int]:
    """Devuelve (records, invalid_lines)."""
    records: List[Dict[str, Any]] = []
    invalid = 0
    text = file_bytes.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            invalid += 1
    return records, invalid


def to_datetime_safe(s: Any) -> pd.Timestamp:
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # columnas típicas (si faltan, las creamos)
    for col in [
        "type",
        "ts",
        "atmId",
        "dir",
        "instrument",
        "tickSize",
        "pointValue",
        "tradeRealized",
        "maxUnreal",
        "minUnreal",
        "outcome",
        "useAtrEngine",
        "orSize",
        "atr",
        "ewo",
        "deltaRatio",
        "avgEntry",
        "slTicks",
        "tp1Ticks",
        "tp2Ticks",
        "exitReason",
        "template",
        "orderType",
        "trigger",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["type"] = df["type"].astype(str).str.upper()
    df["ts_dt"] = df["ts"].apply(to_datetime_safe)

    # numéricos
    num_cols = [
        "dir",
        "tickSize",
        "pointValue",
        "tradeRealized",
        "maxUnreal",
        "minUnreal",
        "orSize",
        "atr",
        "ewo",
        "deltaRatio",
        "avgEntry",
        "slTicks",
        "tp1Ticks",
        "tp2Ticks",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # booleanos
    if "useAtrEngine" in df.columns:
        df["useAtrEngine"] = df["useAtrEngine"].map(
            lambda x: bool(x) if pd.notna(x) else np.nan
        )

    # atmId vacío => NaN
    df["atmId"] = df["atmId"].replace("", np.nan)

    return df


# -------------------------
# Emparejar operaciones (EXIT base + ENTRY opcional)
# -------------------------
def pair_trades(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    1 operación = 1 atmId (tomamos el ÚLTIMO EXIT por atmId).
    Luego añadimos info de ENTRY si existe (primer ENTRY por atmId).
    """
    if df_all.empty:
        return pd.DataFrame()

    exits = df_all[df_all["type"] == "EXIT"].copy()
    entries = df_all[df_all["type"] == "ENTRY"].copy()

    exits = exits[exits["atmId"].notna()].copy()
    if exits.empty:
        return pd.DataFrame()

    # último EXIT por atmId
    exits = exits.sort_values(["atmId", "ts_dt"])
    ex1 = exits.groupby("atmId", as_index=False).tail(1)

    # primer ENTRY por atmId
    e1 = pd.DataFrame()
    if not entries.empty:
        entries = (
            entries[entries["atmId"].notna()]
            .copy()
            .sort_values(["atmId", "ts_dt"])
        )
        e1 = entries.groupby("atmId", as_index=False).head(1)

    entry_cols = [
        "atmId",
        "ts_dt",
        "dir",
        "instrument",
        "tickSize",
        "pointValue",
        "avgEntry",
        "useAtrEngine",
        "template",
        "orderType",
        "trigger",
        "orSize",
        "atr",
        "ewo",
        "deltaRatio",
        "slTicks",
        "tp1Ticks",
        "tp2Ticks",
    ]
    entry_cols = [c for c in entry_cols if c in e1.columns]

    if not e1.empty:
        t = ex1.merge(e1[entry_cols], on="atmId", how="left", suffixes=("_exit", "_entry"))
        t = t.rename(columns={"ts_dt_exit": "exit_ts", "ts_dt_entry": "entry_ts"})
    else:
        t = ex1.copy()
        t = t.rename(columns={"ts_dt": "exit_ts"})
        t["entry_ts"] = pd.NaT

    # outcome
    tr = pd.to_numeric(t["tradeRealized"], errors="coerce").fillna(0.0)
    out = t["outcome"].astype(str).str.upper().replace({"NAN": ""})
    t["outcome"] = np.where(out == "", np.where(tr >= 0, "WIN", "LOSS"), out)

    # dirección: preferimos ENTRY.dir; si no existe, usamos EXIT.dir si lo tienes
    dir_entry = t["dir_entry"] if "dir_entry" in t.columns else pd.Series([np.nan] * len(t))
    dir_exit = t["dir_exit"] if "dir_exit" in t.columns else pd.Series([np.nan] * len(t))
    t["dir_final"] = dir_entry.fillna(dir_exit)

    def side_label(v: Any) -> str:
        if pd.isna(v):
            return "Desconocido"
        try:
            v = int(v)
        except Exception:
            return "Desconocido"
        if v == 1:
            return "Compra (Long)"
        if v == -1:
            return "Venta (Short)"
        return "Desconocido"

    t["lado"] = t["dir_final"].apply(side_label)

    # modo riesgo ATR vs Fijo
    if "useAtrEngine" in t.columns:
        t["modo_riesgo"] = np.where(
            t["useAtrEngine"] == True,
            "ATR",
            np.where(t["useAtrEngine"] == False, "Fijo", "Desconocido"),
        )
    else:
        t["modo_riesgo"] = "Desconocido"

    return t.sort_values("exit_ts").reset_index(drop=True)


# -------------------------
# Métricas
# -------------------------
def profit_factor(trade_realized: pd.Series) -> float:
    wins = trade_realized[trade_realized > 0].sum()
    losses = trade_realized[trade_realized < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / abs(losses))


def max_consecutive(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def equity_and_drawdown(trade_realized: pd.Series):
    eq = trade_realized.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    max_dd = float(dd.min()) if len(dd) else 0.0
    return eq, dd, max_dd


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """Límite inferior Wilson para proporción (winrate). Penaliza muestras pequeñas."""
    if n == 0:
        return 0.0
    phat = wins / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (center - margin) / denom


# -------------------------
# Grupos / rangos por factor
# -------------------------
def _fmt_num(x: float) -> str:
    if pd.isna(x):
        return ""
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax < 1:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    if ax < 100:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return f"{x:.0f}"


def make_groups(series: pd.Series, n_groups: int = 5) -> pd.Series:
    """
    Devuelve rangos tipo "a–b" usando quantiles.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < n_groups:
        return pd.Series(["Sin datos"] * len(series), index=series.index)

    try:
        cats = pd.qcut(s, q=n_groups, duplicates="drop")
    except Exception:
        return pd.Series(["Sin datos"] * len(series), index=series.index)

    def interval_to_str(iv) -> str:
        if not hasattr(iv, "left"):
            return str(iv)
        return f"{_fmt_num(iv.left)}–{_fmt_num(iv.right)}"

    return cats.map(interval_to_str)


def groups_table(trades: pd.DataFrame, factor_col: str, n_groups: int = 5, min_trades: int = 5) -> pd.DataFrame:
    if factor_col not in trades.columns:
        return pd.DataFrame()

    df = trades.copy()
    df["grupo"] = make_groups(df[factor_col], n_groups=n_groups)
    df["tradeRealized"] = pd.to_numeric(df["tradeRealized"], errors="coerce").fillna(0.0)
    df["is_win"] = df["tradeRealized"] >= 0

    g = df.groupby("grupo", dropna=False)
    out = g.agg(
        Operaciones=("atmId", "count"),
        Ganadas=("is_win", "sum"),
        PnL_Total=("tradeRealized", "sum"),
        PnL_Promedio=("tradeRealized", "mean"),
        WinRate=("is_win", "mean"),
        Mayor_Win=("tradeRealized", "max"),
        Mayor_Loss=("tradeRealized", "min"),
    ).reset_index()

    # Profit Factor por grupo
    pf_list = []
    for grp in out["grupo"]:
        tr = df.loc[df["grupo"] == grp, "tradeRealized"]
        pf_list.append(profit_factor(tr))
    out["ProfitFactor"] = pf_list

    out["WinRate(%)"] = (out["WinRate"] * 100).round(1)
    out = out.drop(columns=["WinRate"])

    out = out.sort_values(["Operaciones", "ProfitFactor"], ascending=[False, False]).reset_index(drop=True)
    out["⚠️"] = np.where(out["Operaciones"] < min_trades, "Pocas ops", "")
    return out


def advice_for_groups(tbl: pd.DataFrame, min_trades: int = 10) -> List[str]:
    if tbl.empty:
        return ["No hay suficientes datos para este factor."]
    t_ok = tbl[tbl["Operaciones"] >= min_trades].copy()
    if t_ok.empty:
        return [f"Hay datos, pero ningún grupo tiene ≥ {min_trades} operaciones. Carga más logs o baja el mínimo."]

    best = t_ok.sort_values("ProfitFactor", ascending=False).iloc[0]
    worst = t_ok.sort_values("ProfitFactor", ascending=True).iloc[0]

    tips = []
    tips.append(f"✅ Mejor grupo por PF: **{best['grupo']}** (PF={best['ProfitFactor']:.2f}, ops={int(best['Operaciones'])}, PnL prom={best['PnL_Promedio']:.0f}).")
    tips.append(f"⚠️ Peor grupo por PF: **{worst['grupo']}** (PF={worst['ProfitFactor']:.2f}, ops={int(worst['Operaciones'])}, PnL prom={worst['PnL_Promedio']:.0f}).")

    bad_big = t_ok[(t_ok["ProfitFactor"] < 1.0)].sort_values("Operaciones", ascending=False)
    if not bad_big.empty:
        g0 = bad_big.iloc[0]
        tips.append(f"🚨 Grupo grande con PF<1: **{g0['grupo']}** (PF={g0['ProfitFactor']:.2f}, ops={int(g0['Operaciones'])}). Candidato a filtro.")
    return tips


# -------------------------
# Vega-Lite charts (colores claros)
# -------------------------
def vega_bar_avg_pnl(tbl: pd.DataFrame, title: str) -> Dict[str, Any]:
    d = tbl[["grupo", "PnL_Promedio", "Operaciones"]].copy()
    d = d.rename(columns={"PnL_Promedio": "pnl_prom", "Operaciones": "ops"})
    return {
        "data": {"values": d.to_dict(orient="records")},
        "title": title,
        "width": "container",
        "height": 220,
        "layer": [
            {
                "mark": {"type": "bar"},
                "encoding": {
                    "x": {"field": "grupo", "type": "ordinal", "sort": None, "title": "Grupo / rango"},
                    "y": {"field": "pnl_prom", "type": "quantitative", "title": "PnL promedio por operación"},
                    "color": {"condition": {"test": "datum.pnl_prom >= 0", "value": "green"}, "value": "red"},
                    "tooltip": [
                        {"field": "grupo", "type": "ordinal"},
                        {"field": "ops", "type": "quantitative", "title": "Operaciones"},
                        {"field": "pnl_prom", "type": "quantitative", "title": "PnL prom"},
                    ],
                },
            },
            {"mark": {"type": "rule", "strokeDash": [6, 6]}, "encoding": {"y": {"datum": 0}}},
        ],
    }


def vega_bar_pf(tbl: pd.DataFrame, title: str) -> Dict[str, Any]:
    d = tbl[["grupo", "ProfitFactor", "Operaciones"]].copy()
    d = d.rename(columns={"ProfitFactor": "pf", "Operaciones": "ops"})
    return {
        "data": {"values": d.to_dict(orient="records")},
        "title": title,
        "width": "container",
        "height": 220,
        "layer": [
            {
                "mark": {"type": "bar"},
                "encoding": {
                    "x": {"field": "grupo", "type": "ordinal", "sort": None, "title": "Grupo / rango"},
                    "y": {"field": "pf", "type": "quantitative", "title": "Profit Factor"},
                    "color": {"condition": {"test": "datum.pf >= 1.0", "value": "green"}, "value": "red"},
                    "tooltip": [
                        {"field": "grupo", "type": "ordinal"},
                        {"field": "ops", "type": "quantitative", "title": "Operaciones"},
                        {"field": "pf", "type": "quantitative", "title": "PF"},
                    ],
                },
            },
            {"mark": {"type": "rule", "strokeDash": [6, 6]}, "encoding": {"y": {"datum": 1}}},
        ],
    }


def vega_scatter(trades: pd.DataFrame, x_col: str, title: str) -> Dict[str, Any]:
    df = trades[[x_col, "tradeRealized"]].copy().dropna()
    df = df.rename(columns={x_col: "x", "tradeRealized": "pnl"})
    df["resultado"] = np.where(df["pnl"] >= 0, "Ganada", "Perdida")
    return {
        "data": {"values": df.to_dict(orient="records")},
        "title": title,
        "width": "container",
        "height": 260,
        "mark": {"type": "circle", "opacity": 0.65},
        "encoding": {
            "x": {"field": "x", "type": "quantitative", "title": x_col},
            "y": {"field": "pnl", "type": "quantitative", "title": "PnL por operación"},
            "color": {"field": "resultado", "type": "nominal", "scale": {"domain": ["Ganada", "Perdida"], "range": ["green", "red"]}},
            "tooltip": [{"field": "x", "type": "quantitative", "title": x_col}, {"field": "pnl", "type": "quantitative", "title": "PnL"}],
        },
    }


# -------------------------
# UI principal
# -------------------------
def main():
    st.set_page_config(page_title="WIC_WLF2 Analizador", layout="wide")
    auth_gate()

    st.title("📊 WIC_WLF2 Analizador")
    st.caption("Sube uno o varios archivos **.jsonl** → métricas claras, tablas legibles, gráficos y consejos prácticos.")

    with st.expander("🧠 Cómo leer esto (muy simple)"):
        st.markdown(
            """
- **Operaciones**: número de trades (1 trade = 1 `atmId` con un **EXIT**).
- **Ganadas / Perdidas**: según `tradeRealized` (PnL) o `outcome` si viene en el log.
- **PnL promedio por operación (expectancia)**: PnL total / #operaciones.
- **Profit Factor (PF)**: ganancias totales / pérdidas totales.
- **Equity**: suma acumulada del PnL.
- **Drawdown**: caída máxima desde un pico de equity.
- **Rachas**: máximas ganadas seguidas / perdidas seguidas.
            """
        )

    with st.expander("🔒 Privacidad / sesiones"):
        st.markdown(
            """
- Cada persona que abre la web crea su **propia sesión**.
- Los archivos no se comparten con otros usuarios.
- Para poner contraseña: define `APP_PASSWORD` en Streamlit Cloud.
            """
        )

    files = st.file_uploader(
        "Sube archivos .jsonl (puedes subir varios meses)",
        type=["jsonl", "txt"],
        accept_multiple_files=True,
    )

    if not files:
        st.info("Sube al menos un archivo .jsonl para empezar.")
        return

    all_records: List[Dict[str, Any]] = []
    invalid_total = 0
    for f in files:
        recs, inv = parse_jsonl_bytes(f.getvalue())
        all_records.extend(recs)
        invalid_total += inv

    if not all_records:
        st.error("No se pudo leer ningún JSON válido.")
        return

    df_all = normalize_df(pd.DataFrame(all_records))
    trades = pair_trades(df_all)

    if invalid_total > 0:
        st.warning(f"Líneas inválidas ignoradas: {invalid_total}")

    if trades.empty:
        st.error("No se encontraron operaciones (EXIT) con atmId válido.")
        return

    unknown_n = int((trades["lado"] == "Desconocido").sum())
    if unknown_n > 0:
        st.warning(
            f"⚠️ {unknown_n} operaciones sin dirección (Compra/Venta). "
            "Normalmente falta `ENTRY` para ese atmId o falta `dir` en `EXIT`. "
            "Solución: loguear `dir` también en EXIT."
        )

    tr = pd.to_numeric(trades["tradeRealized"], errors="coerce").fillna(0.0)
    n_ops = int(len(trades))
    wins = int((tr >= 0).sum())
    losses = int((tr < 0).sum())
    winrate = wins / n_ops if n_ops else 0.0
    pf = profit_factor(tr)
    expectancy = float(tr.mean()) if n_ops else 0.0
    eq, dd, max_dd = equity_and_drawdown(tr)

    is_win = (trades.sort_values("exit_ts")["tradeRealized"].fillna(0.0) >= 0).to_numpy()
    best_streak = max_consecutive(is_win)
    worst_streak = max_consecutive(~is_win)

    long_n = int((trades["lado"] == "Compra (Long)").sum())
    short_n = int((trades["lado"] == "Venta (Short)").sum())

    atr_n = int((trades["modo_riesgo"] == "ATR").sum())
    fijo_n = int((trades["modo_riesgo"] == "Fijo").sum())

    st.markdown("## ✅ Lo más importante (resumen rápido)")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Operaciones", f"{n_ops}")
    c2.metric("Compras (Long)", f"{long_n}")
    c3.metric("Ventas (Short)", f"{short_n}")
    c4.metric("Ganadas", f"{wins}")
    c5.metric("Perdidas", f"{losses}")
    c6.metric("% Acierto", f"{winrate*100:.1f}%")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("PnL Total", f"{tr.sum():.0f}")
    c2.metric("Profit Factor", f"{pf:.2f}" if math.isfinite(pf) else "∞")
    c3.metric("Expectancia (PnL prom/op)", f"{expectancy:.1f}")
    c4.metric("Max Drawdown", f"{max_dd:.0f}")
    c5.metric("Mejor racha (wins)", f"{best_streak}")
    c6.metric("Peor racha (losses)", f"{worst_streak}")

    st.caption(f"Modo de riesgo: **ATR {atr_n} ops ({(atr_n/n_ops*100 if n_ops else 0):.1f}%)** · **Fijo {fijo_n} ops ({(fijo_n/n_ops*100 if n_ops else 0):.1f}%)**")

    # Consejos rápidos
    if pf < 1.0:
        st.error(f"🚨 PF {pf:.2f} (<1.0): con estos datos, el sistema pierde dinero.")
    elif pf < 1.2:
        st.warning(f"⚠️ PF {pf:.2f}: ventaja débil. Necesitas filtros/gestión.")
    else:
        st.success(f"✅ PF {pf:.2f}: se ve ventaja estadística con estos datos.")

    if worst_streak >= 8:
        st.warning(f"⚠️ Racha perdedora máx = {worst_streak}. Plan recomendado: daily stop / reducción tamaño / filtro de condiciones malas.")

    # Rendimiento por lado
    st.markdown("## 🧭 Rendimiento por lado (Compra vs Venta)")
    tmp = trades.copy()
    tmp["is_win"] = tmp["tradeRealized"].fillna(0.0) >= 0
    g = tmp.groupby("lado", dropna=False)
    side = g.agg(
        Trades=("atmId", "count"),
        Wins=("is_win", "sum"),
        PnL_Total=("tradeRealized", "sum"),
        PnL_Prom=("tradeRealized", "mean"),
        Mayor_Win=("tradeRealized", "max"),
        Mayor_Loss=("tradeRealized", "min"),
    ).reset_index()
    side["WinRate(%)"] = (side["Wins"] / side["Trades"] * 100).round(1)
    side["PF"] = [profit_factor(tmp.loc[tmp["lado"] == lado, "tradeRealized"].fillna(0.0)) for lado in side["lado"]]
    st.dataframe(side, use_container_width=True)

    # Instrumentos
    st.markdown("## 🎯 Instrumentos")
    if "instrument_entry" in trades.columns and trades["instrument_entry"].notna().any():
        inst_df = trades.copy()
        inst_df["instrument"] = inst_df["instrument_entry"].fillna("Desconocido")
        inst = inst_df.groupby("instrument").agg(
            Operaciones=("atmId", "count"),
            PnL_Total=("tradeRealized", "sum"),
            PnL_Prom=("tradeRealized", "mean"),
        ).reset_index().sort_values("Operaciones", ascending=False)
        st.dataframe(inst, use_container_width=True)
    else:
        st.info("No hay `instrument` en estos logs. Si lo logueas en ENTRY/EXIT, podrás ver resultados por instrumento.")

    # Equity + Drawdown
    st.markdown("## 📈 Equity y Drawdown")
    eq_df = pd.DataFrame({"exit_ts": trades["exit_ts"], "equity": eq, "drawdown": dd}).dropna().set_index("exit_ts")
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(eq_df["equity"], height=260)
        st.caption("Equity: debe subir estable. Si oscila demasiado → falta filtro/gestión.")
    with c2:
        st.line_chart(eq_df["drawdown"], height=260)
        st.caption("Drawdown: mientras más negativo, peor. Busca bajarlo sin matar el PF.")

    # Horas (ajustado)
    st.markdown("## 🕒 Rendimiento por hora (ajustado por cantidad)")
    if trades["exit_ts"].notna().any():
        h = trades[trades["exit_ts"].notna()].copy()
        h["hour"] = h["exit_ts"].dt.hour
        h["is_win"] = h["tradeRealized"].fillna(0.0) >= 0
        gh = h.groupby("hour")
        byh = gh.agg(Operaciones=("atmId", "count"), Ganadas=("is_win", "sum")).reset_index()
        byh["WinRate(%)"] = (byh["Ganadas"] / byh["Operaciones"] * 100).round(1)
        byh["WinRate_Ajustado(%)"] = [wilson_lower_bound(int(w), int(n)) * 100 for w, n in zip(byh["Ganadas"], byh["Operaciones"])]
        byh = byh.sort_values("WinRate_Ajustado(%)", ascending=False).reset_index(drop=True)

        min_ops = st.slider("Mínimo de operaciones para confiar en una hora", 1, 50, 10)
        st.dataframe(byh, use_container_width=True)
        st.caption("Una hora con 2 trades al 100% NO es confiable. El winrate ajustado penaliza muestras pequeñas.")

    # Factores
    st.markdown("## 🧪 Factores (OR/ATR/EWO/DeltaRatio)")
    st.caption("Ver en qué rangos un factor mejora o empeora PnL/PF para crear filtros claros.")

    c1, c2, c3 = st.columns(3)
    n_groups = c1.slider("Número de grupos", 3, 10, 5)
    min_tr = c2.slider("Mínimo de operaciones por grupo (para consejos)", 1, 50, 10)
    show_scatter = c3.checkbox("Mostrar scatter (detalle)", value=True)

    factor_map = {"OR Size": "orSize", "ATR": "atr", "EWO": "ewo", "DeltaRatio": "deltaRatio"}
    tabs = st.tabs(list(factor_map.keys()))

    for tab, (label, col) in zip(tabs, factor_map.items()):
        with tab:
            if col not in trades.columns or trades[col].notna().sum() == 0:
                st.info(f"No hay datos para {label} (`{col}`).")
                continue

            tbl = groups_table(trades, col, n_groups=n_groups, min_trades=min_tr)
            st.subheader(f"{label} — análisis por rangos")

            c1, c2 = st.columns(2)
            with c1:
                st.vega_lite_chart(vega_bar_avg_pnl(tbl, f"{label}: PnL promedio por grupo"), use_container_width=True)
            with c2:
                st.vega_lite_chart(vega_bar_pf(tbl, f"{label}: Profit Factor por grupo"), use_container_width=True)

            st.dataframe(tbl, use_container_width=True)

            st.markdown("#### Consejos prácticos")
            for m in advice_for_groups(tbl, min_trades=min_tr):
                st.info(m)

            if show_scatter:
                st.markdown("#### Scatter (detalle de cada operación)")
                st.vega_lite_chart(vega_scatter(trades, col, f"{label}: valor vs PnL por operación"), use_container_width=True)
                st.caption("Verde = ganancia, rojo = pérdida. Útil para ver ruido/outliers.")

    # Export CSV
    st.markdown("## ⬇️ Exportar")
    out = trades.copy()
    out["exit_ts"] = out["exit_ts"].astype(str)
    out["entry_ts"] = out.get("entry_ts", pd.Series([pd.NaT]*len(out))).astype(str)
    st.download_button(
        "Descargar operaciones emparejadas (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="wic_wlf2_trades.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
