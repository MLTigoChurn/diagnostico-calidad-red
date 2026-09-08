"""
build_modeling_dataset_v2.py

Construye el dataset mensual usando solo informacion anterior al mes del
objetivo. Corrige cuatro defectos de `src/build_modeling_dataset.py`: el
numerador de la degradacion sin desplazar, `USERS_4G` tomado del propio mes,
las ventanas promediadas dentro del mes y el rolling que cruza entre sitios.
El detalle esta en `src_v2/README.md`.

Uso:
    uv run python src_v2/build_modeling_dataset_v2.py
    uv run python src/score_sites.py \
        --data output_v2/modeling_dataset_monthly_v2.parquet \
        --output-dir output_v2/mensual-pronopro
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

import build_modeling_dataset as b1  # noqa: E402

OUTPUT_PATH_V2 = BASE_DIR / "output_v2" / "modeling_dataset_monthly_v2.parquet"
WINDOW_LARGA = 30
WINDOW_CORTA = 7


def build_monthly_kpis_v2(
    kpis: pd.DataFrame,
    month_anchor: str = b1.MONTH_ANCHOR,
    window: int = WINDOW_LARGA,
) -> pd.DataFrame:
    """Resume los KPI de los `window` dias anteriores a cada sitio-mes."""
    kpis = kpis.sort_values(["site_id", "dia"]).reset_index(drop=True).copy()
    grp = kpis.groupby("site_id", group_keys=False)

    rolling_specs = {
        "avg_AVA_4G_30d": ("AVA_4G", "mean"),
        "avg_CSFR_V4G_30d": ("CSFR_V4G", "mean"),
        "max_DC_V4G_30d": ("DC_V4G", "max"),
        "avg_THP_4G_30d": ("THROUGHPUT_4G", "mean"),
        "USERS_4G": ("USERS_4G", "mean"),
    }

    def rodante(col: str, w: int, how: str) -> pd.Series:
        """Devuelve el rolling desplazado un dia, calculado dentro de cada sitio."""
        return grp[col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).agg(how))

    for new_col, (src_col, how) in rolling_specs.items():
        if src_col not in kpis.columns:
            continue
        kpis[new_col] = rodante(src_col, window, how)

    if "THROUGHPUT_4G" in kpis.columns:
        corta = rodante("THROUGHPUT_4G", WINDOW_CORTA, "mean")
        kpis["degradacion_THP_30d"] = corta / kpis["avg_THP_4G_30d"].replace(0, np.nan)

    cols = [c for c in rolling_specs if c in kpis.columns]
    if "degradacion_THP_30d" in kpis.columns:
        cols.append("degradacion_THP_30d")

    meses = (
        kpis[["site_id", "dia"]]
        .assign(mes=lambda d: d["dia"].dt.to_period(month_anchor))
        .groupby(["site_id", "mes"], as_index=False)
        .size()
        .drop(columns="size")
    )
    meses["inicio"] = meses["mes"].dt.to_timestamp()

    izq = meses.sort_values("inicio")
    der = kpis[["site_id", "dia"] + cols].sort_values("dia")
    out = pd.merge_asof(
        izq, der, left_on="inicio", right_on="dia", by="site_id",
        direction="backward", allow_exact_matches=True,
    )
    return out[["site_id", "mes"] + cols]


def build_monthly_v2(source: str = "files", output_path: Path = OUTPUT_PATH_V2) -> pd.DataFrame:
    """Construye y escribe el dataset mensual corregido."""
    claims, location, kpis = b1.load_sources(source, b1.DB_FILES_DIR)
    claims, location, kpis = b1.normalize(claims, location, kpis)
    b1.validate_integrity(claims, location, kpis)

    kpis_monthly = build_monthly_kpis_v2(kpis)
    target = b1.build_monthly_target(claims, location)

    dataset = kpis_monthly.merge(target, on=["site_id", "mes"], how="left")
    count_cols = ["n_reclamos_mes", "n_claims_mes"]
    if "n_reclamos_confirmados_mes" in dataset.columns:
        count_cols.append("n_reclamos_confirmados_mes")
    for col in count_cols:
        dataset[col] = dataset[col].fillna(0).astype(int)
    dataset["es_reclamo_mensual"] = (dataset["n_reclamos_mes"] > 0).astype(int)
    if "n_reclamos_confirmados_mes" in dataset.columns:
        dataset["es_reclamo_confirmado_mensual"] = (
            dataset["n_reclamos_confirmados_mes"] > 0
        ).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    print(f"\nDataset mensual v2 (site x mes): {dataset.shape[0]} filas, {dataset.shape[1]} columnas")
    print(f"Escrito en: {output_path}")
    return dataset


if __name__ == "__main__":
    build_monthly_v2()
