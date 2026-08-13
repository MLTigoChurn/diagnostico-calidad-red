"""
eda_claims_location.py — EDA descriptivo de claims y location (TFM Grupo 1)

Complementa `eda_diagnostico.py` (que cubre kpis + target). Aquí el foco es
caracterizar las dos fuentes categóricas:
- claims: ¿cuántos RECLAMO vs CONSULTA? categorías, segmento, temporalidad, recurrencia.
- location: tipo de negocio, tecnología, concentración usuario→sitio, tráfico.

Para variables categóricas se reportan FRECUENCIAS y proporciones (no media/mediana,
que no aplican a nominales). Para tráfico (numérico, muy sesgado) se usan mediana y
percentiles, como en kpis.

Salida (carpeta por corrida, sellada con fecha y hora):
    output/eda/claims_location_<YYYY-MM-DD_HHMM>/
        *.png · reporte_<...>.pdf · resumen.md

Uso:
    python scripts/eda_claims_location.py
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

BASE_DIR = Path(__file__).resolve().parent.parent
DB = BASE_DIR / "db_files"
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H%M")
OUTDIR = BASE_DIR / "output" / "eda" / f"claims_location_{RUN_ID}"
PDF_PATH = OUTDIR / f"reporte_{RUN_ID}.pdf"
RESUMEN = OUTDIR / "resumen.md"
OUTDIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
_figs: list[tuple] = []
_lines: list[str] = []


def register(fig, slug, titulo):
    _figs.append((fig, slug, titulo))


def log(s=""):
    print(s)
    _lines.append(s)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load():
    claims = pd.read_csv(DB / "claims.csv", sep="|", encoding="utf-8-sig",
                         dtype=str, low_memory=False).replace(r"\N", np.nan)
    location = pd.read_csv(DB / "location.csv", sep=",", dtype=str,
                           low_memory=False).replace(r"\N", np.nan)
    claims["user_id"] = claims["user_id"].astype("string").str.strip()
    claims["motivo"] = claims["motivo_atencion"].astype("string").str.upper().str.strip()
    claims["fecha"] = pd.to_datetime(claims["fecha_creacion"], errors="coerce")
    for c in ["user_id", "site_id"]:
        location[c] = location[c].astype("string").str.strip()
    for c in ["trafico_4g", "trafico_3g", "trafico_2g"]:
        location[c] = pd.to_numeric(location[c], errors="coerce")
    return claims, location


def barras(serie_counts, titulo, slug, pct=True, color="#4C72B0", rot=0):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = serie_counts.values
    ax.bar(range(len(vals)), vals, color=color)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(serie_counts.index, rotation=rot, ha="right" if rot else "center", fontsize=9)
    total = vals.sum()
    for i, v in enumerate(vals):
        lab = f"{v:,}\n({v/total:.1%})" if pct else f"{v:,}"
        ax.text(i, v, lab, ha="center", va="bottom", fontsize=8)
    ax.set_title(titulo)
    ax.margins(y=0.15)
    fig.tight_layout(); register(fig, slug, titulo)


# ---------------------------------------------------------------------------
# CLAIMS
# ---------------------------------------------------------------------------

def analiza_claims(claims):
    log("# EDA claims & location — TFM Grupo 1\n")
    log(f"_Generado por `scripts/eda_claims_location.py` el {RUN_ID}. PDF: `{PDF_PATH.name}`._\n")
    log("## CLAIMS\n")
    n = len(claims)
    n_nN = claims["user_id"].isna().sum()
    log(f"- Tickets totales: **{n}**")
    log(f"- `user_id` nulos (\\N): {n_nN} ({n_nN/n:.1%}) → se pierden en el join a sitio")
    log(f"- `nro_ticket` únicos: {claims['nro_ticket'].nunique()} (sin duplicados: {claims['nro_ticket'].nunique()==n})\n")

    # Motivo de atención — el foco
    mot = claims["motivo"].value_counts()
    log("### Motivo de atención (foco)\n")
    log((mot.to_frame("conteo").assign(pct=(mot/n*100).round(1)).to_markdown()))
    log("")
    barras(mot, "Motivo de atención: RECLAMO vs CONSULTA", "c01_motivo", color="#C44E52")

    # Segmento
    seg = claims["segmento"].value_counts().head(10)
    barras(seg, "Distribución por segmento de cliente", "c02_segmento", rot=30)

    # Category_2 (subcategoría)
    cat2 = claims["category_2"].value_counts().head(8)
    barras(cat2, "Top subcategorías (category_2)", "c03_category2", rot=30)

    # Temporal: RECLAMO vs CONSULTA por mes
    tmp = (claims.dropna(subset=["fecha"])
           .assign(mes=lambda d: d["fecha"].dt.to_period("M").astype(str)))
    piv = tmp.pivot_table(index="mes", columns="motivo", values="nro_ticket",
                          aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    piv.plot(kind="bar", stacked=True, ax=ax, color={"RECLAMO": "#C44E52", "CONSULTA": "#4C72B0"})
    ax.set_title("Tickets por mes (RECLAMO vs CONSULTA)")
    ax.set_xlabel(""); ax.set_ylabel("nº tickets")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(); register(fig, "c04_temporal_mes", "Tickets por mes")

    # Patrón por hora del día
    horas = claims["fecha"].dt.hour.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(horas.index, horas.values, color="#4C72B0")
    ax.set_title("Tickets por hora de creación"); ax.set_xlabel("hora"); ax.set_ylabel("nº")
    fig.tight_layout(); register(fig, "c05_hora", "Tickets por hora")

    # Recurrencia: claims por usuario (solo user_id válidos)
    rec = claims.dropna(subset=["user_id"]).groupby("user_id").size()
    log("### Recurrencia (tickets por usuario, user_id válidos)\n")
    log(f"- usuarios únicos: {rec.size} | media: {rec.mean():.2f} | máx: {rec.max()} | con >1 ticket: {(rec>1).mean():.1%}\n")

    # PII en description
    desc = claims["description"].dropna()
    con_ph = desc.str.contains(r"\[", regex=True).mean()
    log("### Texto libre (description)\n")
    log(f"- con placeholder de PII anonimizada (`[TEL]/[NOMBRE]/...`): {con_ph:.1%} de los tickets")
    log(f"- longitud media: {desc.str.len().mean():.0f} caracteres\n")


# ---------------------------------------------------------------------------
# LOCATION
# ---------------------------------------------------------------------------

def analiza_location(location):
    log("## LOCATION\n")
    n = len(location)
    log(f"- Filas (usuario×snapshot): **{n:,}**")
    log(f"- usuarios únicos: {location['user_id'].nunique():,} | sitios únicos: {location['site_id'].nunique():,}\n")

    tn = location["tipo_negocio"].value_counts()
    log("### Tipo de negocio\n")
    log(tn.to_frame("conteo").assign(pct=(tn/n*100).round(1)).to_markdown())
    log("")
    barras(tn, "Distribución por tipo de negocio", "l01_tipo_negocio")

    tec = location["tecnologia_soporta"].value_counts()
    log("### Tecnología soportada\n")
    log(tec.to_frame("conteo").assign(pct=(tec/n*100).round(1)).to_markdown())
    log("")
    barras(tec, "Tecnología soportada por el sitio", "l02_tecnologia", color="#55A868")

    # Concentración usuario -> sitio
    por_user = location.dropna(subset=["user_id"]).groupby("user_id")["site_id"].nunique()
    multi = (por_user > 1).mean()
    log("### Estructura del join usuario → sitio\n")
    log(f"- usuarios con >1 sitio distinto: {multi:.2%} → confirma regla **1 usuario → 1 sitio**")
    usuarios_por_sitio = location.dropna(subset=["site_id"]).groupby("site_id")["user_id"].nunique()
    log(f"- usuarios por sitio — mediana: {usuarios_por_sitio.median():.0f} | "
        f"máx: {usuarios_por_sitio.max():,} (infraestructura concentrada)\n")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(usuarios_por_sitio, bins=60, ax=ax, color="#55A868")
    ax.set_yscale("log")
    ax.set_title("Usuarios únicos por sitio (Y log)"); ax.set_xlabel("usuarios/sitio")
    fig.tight_layout(); register(fig, "l03_usuarios_por_sitio", "Usuarios por sitio")

    # Tráfico por tecnología (numérico, muy sesgado)
    traf = location[["trafico_4g", "trafico_3g", "trafico_2g"]]
    log("### Tráfico por tecnología (muy sesgado → mediana y percentiles)\n")
    log(traf.describe(percentiles=[.5, .9, .99]).round(2).to_markdown())
    log("")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, c in zip(axes, ["trafico_4g", "trafico_3g", "trafico_2g"]):
        s = location[c].dropna()
        s = s[s > 0]
        sns.histplot(s, bins=60, ax=ax, color="#4C72B0", log_scale=(True, False))
        ax.set_title(f"{c} (>0, X log)")
    fig.suptitle("Distribución de tráfico por tecnología", fontsize=12)
    fig.tight_layout(); register(fig, "l04_trafico", "Tráfico por tecnología")


def main():
    claims, location = load()
    analiza_claims(claims)
    analiza_location(location)

    with PdfPages(PDF_PATH) as pdf:
        for fig, slug, _ in _figs:
            fig.savefig(OUTDIR / f"{slug}.png", dpi=140)
            pdf.savefig(fig)
            plt.close(fig)

    # Embeber figuras en el md
    figs_md = ["## Figuras\n"]
    for _, slug, titulo in _figs:
        figs_md += [f"### {titulo}\n", f"![{titulo}]({slug}.png)\n"]
    RESUMEN.write_text("\n".join(_lines) + "\n" + "\n".join(figs_md), encoding="utf-8")

    print(f"\n✓ Carpeta: {OUTDIR}")
    print(f"✓ PDF:     {PDF_PATH.name}")
    print(f"✓ Resumen: {RESUMEN.name}")


if __name__ == "__main__":
    main()
