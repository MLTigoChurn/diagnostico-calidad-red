"""
eda_diagnostico.py

Genera el analisis exploratorio con enfoque diagnostico: target RECLAMO a nivel
sitio por dia, y umbral de cada KPI a partir del cual sube la tasa de queja.

Salida (una carpeta por corrida, sellada con fecha y hora):
    output/eda/eda_<YYYY-MM-DD_HHMM>/
        *.png                          (figuras)
        eda_reporte_<...>.pdf          (todas las figuras en un PDF)
        resumen.md                     (figuras embebidas + estadística)

Uso:
    python scripts/eda_diagnostico.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
DB = BASE_DIR / "db_files"
# Una carpeta por corrida, sellada con fecha y hora -> no se pisan resultados.
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H%M")
OUTDIR = BASE_DIR / "output" / "eda" / f"eda_{RUN_ID}"
FIG = OUTDIR
RESUMEN = OUTDIR / "resumen.md"
PDF_PATH = OUTDIR / f"eda_reporte_{RUN_ID}.pdf"
OUTDIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# Registro de figuras (fig, slug, título) para PNG + PDF + embebido en el md.
_figs: list[tuple] = []


def register(fig, slug: str, titulo: str) -> None:
    _figs.append((fig, slug, titulo))

# KPIs de calidad accionables (foco del diagnóstico) y confusor de carga.
KPIS_FOCO = ["AVA_4G", "DC_V4G", "THROUGHPUT_4G", "CSFR_V4G"]
CONFUSOR = "USERS_4G"
# KPIs numéricos para correlación/normalidad (los reales del crudo; NO ids).
KPIS_NUM = [
    "MM_TOTAL", "TB_TOTAL", "AVA_4G", "DC_V4G", "CSFR_V4G", "CSFR_D4G",
    "DC_D4G", "USERS_4G", "THROUGHPUT_4G", "AVA_3G", "USERS_3G", "THROUGHPUT_3G",
]

_lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    _lines.append(s)


# ---------------------------------------------------------------------------
# 1. Carga y construcción del panel sitio × día con target RECLAMO
# ---------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    claims = pd.read_csv(DB / "claims.csv", sep="|", encoding="utf-8-sig",
                         dtype=str, low_memory=False)
    location = pd.read_csv(DB / "location.csv", sep=",", dtype=str, low_memory=False)
    kpis = pd.read_csv(DB / "kpis_diario.txt", sep="\t", low_memory=False)
    # \N -> nulo
    claims = claims.replace(r"\N", np.nan)
    location = location.replace(r"\N", np.nan)
    # IDs como string (no perder ceros, no tratar como número)
    for df, cols in [(claims, ["user_id"]), (location, ["user_id", "site_id"])]:
        for c in cols:
            df[c] = df[c].astype("string").str.strip()
    kpis["site_id"] = kpis["site_id"].astype("string").str.strip()
    # Fechas: claims ISO, kpis DD/MM/YYYY (forzar formato o 60% NaT silenciosos)
    claims["dia"] = pd.to_datetime(claims["fecha_creacion"], errors="coerce").dt.normalize()
    kpis["dia"] = pd.to_datetime(kpis["FECHA"], format="%d/%m/%Y", errors="coerce").dt.normalize()
    return claims, location, kpis


def build_panel(claims, location, kpis) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel sitio × día (KPIs + es_reclamo) y claims con site_id asignado (cl)."""
    user_site = location[["user_id", "site_id"]].dropna().drop_duplicates("user_id")
    cl = claims.merge(user_site, on="user_id", how="inner")
    motivo = cl["motivo_atencion"].astype("string").str.upper().str.strip()
    rec = (
        cl.assign(es_reclamo=(motivo == "RECLAMO").astype(int))
        .groupby(["site_id", "dia"], as_index=False)["es_reclamo"].max()
    )
    panel = kpis.merge(rec, on=["site_id", "dia"], how="left")
    panel["es_reclamo"] = panel["es_reclamo"].fillna(0).astype(int)
    return panel, cl


# ---------------------------------------------------------------------------
# 2. Análisis + figuras
# ---------------------------------------------------------------------------

def tasa_por_bucket(panel: pd.DataFrame, kpi: str, bins) -> pd.DataFrame:
    sub = panel[[kpi, "es_reclamo"]].dropna(subset=[kpi])
    sub = sub.assign(bucket=pd.cut(sub[kpi], bins=bins, include_lowest=True))
    g = sub.groupby("bucket", observed=True)["es_reclamo"].agg(["mean", "count"])
    g["mean"] = (g["mean"] * 100).round(3)
    return g.rename(columns={"mean": "tasa_queja_%", "count": "n"})


def fig_target_balance(panel):
    pos = panel["es_reclamo"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Sin reclamo", "Con reclamo"], [1 - pos, pos], color=["#4C72B0", "#C44E52"])
    ax.set_ylabel("Proporción de sitio-días")
    ax.set_title(f"Desbalanceo del target (sitio×día)\npositivos = {pos:.3%} (~1:{round(1/pos)})")
    for i, v in enumerate([1 - pos, pos]):
        ax.text(i, v, f"{v:.3%}", ha="center", va="bottom")
    fig.tight_layout(); register(fig, "01_target_balance", "Desbalanceo del target")
    return pos


def fig_tasa_umbral(panel):
    bins_map = {
        "AVA_4G": [0, 0.90, 0.95, 0.98, 0.99, 1.001],
        "DC_V4G": [0, 0.005, 0.01, 0.02, 0.05, 1.001],
        "THROUGHPUT_4G": [0, 5, 10, 15, 20, 30, 1e9],
        "CSFR_V4G": [0, 0.005, 0.01, 0.02, 0.05, 1.001],
    }
    base = panel["es_reclamo"].mean() * 100
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, kpi in zip(axes.ravel(), KPIS_FOCO):
        if kpi not in panel.columns:
            continue
        g = tasa_por_bucket(panel, kpi, bins_map[kpi])
        ax.bar(range(len(g)), g["tasa_queja_%"], color="#C44E52")
        ax.axhline(base, ls="--", color="gray", lw=1, label=f"baseline {base:.2f}%")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([str(b) for b in g.index], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Tasa de queja por rango de {kpi}")
        ax.set_ylabel("% sitio-días con reclamo")
        ax.legend(fontsize=8)
    fig.suptitle("Pregunta central: ¿desde qué umbral sube la tasa de queja?", fontsize=13)
    fig.tight_layout(); register(fig, "02_tasa_queja_umbral", "Tasa de queja por umbral de KPI")
    return {k: tasa_por_bucket(panel, k, bins_map[k]) for k in KPIS_FOCO if k in panel.columns}


def fig_distribuciones(panel):
    cols = [c for c in ["AVA_4G", "THROUGHPUT_4G", "USERS_4G", "DC_V4G"] if c in panel.columns]
    # AVA/DC están casi todas en su valor sano (1.0 / 0); escala log en Y
    # revela la cola, que es donde vive la señal de queja.
    log_y = {"AVA_4G", "DC_V4G"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, c in zip(axes.ravel(), cols):
        sns.histplot(panel[c].dropna(), bins=60, ax=ax, color="#4C72B0")
        ttl = f"Distribución de {c}"
        if c in log_y:
            ax.set_yscale("log")
            ttl += " (Y log — cola)"
        ax.set_title(ttl)
    fig.suptitle("Distribución de KPIs clave (sesgo + colas)", fontsize=13)
    fig.tight_layout(); register(fig, "03_distribuciones_kpis", "Distribución de KPIs clave")


def fig_confusor(panel):
    if CONFUSOR not in panel.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [panel.loc[panel.es_reclamo == 0, CONFUSOR].dropna(),
            panel.loc[panel.es_reclamo == 1, CONFUSOR].dropna()]
    ax.boxplot(data, label=["Sin reclamo", "Con reclamo"], showfliers=False)
    ax.set_ylabel(CONFUSOR)
    ax.set_title(f"Confusor de carga: {CONFUSOR} por clase\n(sitios con queja tienen más usuarios)")
    fig.tight_layout(); register(fig, "04_confusor_users", "Confusor de carga (USERS_4G)")


def fig_spearman(panel):
    cols = [c for c in KPIS_NUM if c in panel.columns]
    corr = panel[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlación de Spearman entre KPIs")
    fig.tight_layout(); register(fig, "05_spearman_heatmap", "Correlación de Spearman")
    return corr


def fig_temporal(panel):
    s = (panel.assign(semana=panel["dia"].dt.to_period("W").dt.start_time)
         .groupby("semana")["es_reclamo"].sum())
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.index, s.values, marker="o", color="#C44E52")
    ax.set_title("Reclamos por semana (sitio×día agregados)")
    ax.set_ylabel("Nº sitio-días con reclamo")
    fig.tight_layout(); register(fig, "06_temporal", "Reclamos por semana")


# ---------------------------------------------------------------------------
# 3. Estadística (correcta para el scope)
# ---------------------------------------------------------------------------

def figuras_block():
    log("# EDA diagnóstico — TFM Grupo 1\n")
    log(f"_Generado por `scripts/eda_diagnostico.py` el {RUN_ID}. "
        f"Reporte en PDF: `{PDF_PATH.name}`._\n")
    log("## Figuras\n")
    for _, slug, titulo in _figs:
        log(f"### {titulo}\n")
        log(f"![{titulo}]({slug}.png)\n")


def stats_block(panel, claims, location, kpis, cl):

    log("## Volúmenes y cobertura del join\n")
    log(f"- claims: {len(claims)} tickets | user_id válidos (no \\N): {claims['user_id'].notna().sum()}")
    log(f"- location: {len(location)} filas | sitios únicos: {location['site_id'].nunique()}")
    log(f"- kpis: {len(kpis)} filas | sitios únicos: {kpis['site_id'].nunique()} | días: {kpis['dia'].nunique()}")
    cov_claims = cl["site_id"].notna().mean() * 100
    log(f"- claims con site_id asignado vía location: {len(cl)} ({cov_claims:.1f}% de los unidos)\n")

    log("## Desbalanceo del target\n")
    pos_d = panel["es_reclamo"].mean()
    sem = (panel.assign(semana=panel["dia"].dt.to_period("W"))
           .groupby(["site_id", "semana"])["es_reclamo"].max())
    pos_w = sem.mean()
    log(f"- sitio×día: {pos_d:.3%} positivos (~1:{round(1/pos_d)})")
    log(f"- sitio×semana: {pos_w:.3%} positivos (~1:{round(1/pos_w)}) → grano de modelado\n")

    log("## Tasa de queja por umbral (hallazgo central)\n")
    for kpi, g in TASAS.items():
        log(f"**{kpi}**\n")
        log(g.to_markdown())
        log("")

    log("## Medianas por clase + Mann-Whitney U (queja vs sin queja)\n")
    log("Reemplaza al Kruskal-Wallis sobre CLAIMS: aquí el contraste es binario sobre el target RECLAMO real.\n")
    rows = []
    for c in [x for x in KPIS_NUM if x in panel.columns]:
        a = panel.loc[panel.es_reclamo == 1, c].dropna()
        b = panel.loc[panel.es_reclamo == 0, c].dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"KPI": c, "mediana_c/queja": round(a.median(), 4),
                     "mediana_s/queja": round(b.median(), 4),
                     "p_value": f"{p:.2e}", "signif (p<0.05)": "sí" if p < 0.05 else "no"})
    log(pd.DataFrame(rows).to_markdown(index=False))
    log("")

    log("## Normalidad (Kolmogorov-Smirnov) — solo KPIs numéricos\n")
    log("> SITE_ID y user_id son **identificadores**: no se les aplica normalidad ni media (no tienen magnitud).\n")
    rows = []
    for c in [x for x in KPIS_NUM if x in panel.columns]:
        s = panel[c].dropna()
        if len(s) < 20:
            continue
        st = (s - s.mean()) / (s.std() + 1e-12)
        ksst, ksp = stats.kstest(st.sample(min(5000, len(st)), random_state=42), "norm")
        rows.append({"KPI": c, "KS": round(ksst, 4), "p": f"{ksp:.2e}",
                     "normal": "no" if ksp < 0.05 else "sí"})
    log(pd.DataFrame(rows).to_markdown(index=False))
    log("")

    log("## Top correlaciones de Spearman (|rho| ≥ 0.5)\n")
    corr = CORR.where(~np.eye(len(CORR), dtype=bool))
    pairs = (corr.stack().reset_index()
             .rename(columns={"level_0": "A", "level_1": "B", 0: "rho"}))
    pairs = pairs[pairs["A"] < pairs["B"]]
    pairs = pairs[pairs["rho"].abs() >= 0.5].sort_values("rho", key=abs, ascending=False)
    log(pairs.round(3).to_markdown(index=False))
    log("")


def main():
    claims, location, kpis = load()
    panel, cl = build_panel(claims, location, kpis)

    global TASAS, CORR
    pos = fig_target_balance(panel)
    TASAS = fig_tasa_umbral(panel)
    fig_distribuciones(panel)
    fig_confusor(panel)
    CORR = fig_spearman(panel)
    fig_temporal(panel)

    # Guardar PNG (para el md) + PDF (un solo archivo con todas las figuras).
    with PdfPages(PDF_PATH) as pdf:
        for fig, slug, _ in _figs:
            fig.savefig(FIG / f"{slug}.png", dpi=140)
            pdf.savefig(fig)
            plt.close(fig)

    figuras_block()
    stats_block(panel, claims, location, kpis, cl)
    RESUMEN.write_text("\n".join(_lines), encoding="utf-8")

    print(f"\n✓ Carpeta de la corrida: {OUTDIR}")
    print(f"✓ PDF con figuras:       {PDF_PATH.name}")
    print(f"✓ Resumen (md+imágenes): {RESUMEN.name}")


if __name__ == "__main__":
    main()
