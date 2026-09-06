"""
build_modeling_dataset_v2.py — TFM Grupo 1

Version corregida del builder mensual. NO reemplaza a `src/build_modeling_dataset.py`,
que es el que produjo los resultados entregados. Existe para medir cuanto cambia el
resultado al eliminar la informacion del propio periodo que detecto el asesor
del TFM (ver `src_v2/README.md`).

Que corrige, respecto del original:

1. `degradacion_THP_30d` usaba `THROUGHPUT_4G` del dia en curso, sin desplazar,
   como numerador (`build_modeling_dataset.py:433`). Aca se reemplaza por el
   cociente entre una ventana corta (7d) y una larga (30d), ambas cerradas
   antes del mes. Se conserva la intencion original (cuanto cayo el sitio
   respecto de su propia normalidad) sin mirar el periodo del objetivo.

2. `USERS_4G` entraba como media del propio mes, sin `shift` ni `rolling`
   (`build_modeling_dataset.py:443`). Aca pasa por la misma ventana previa que
   el resto.

3. El original resume las columnas rolling promediandolas *dentro* del mes, asi
   que la ventana del dia 20 de marzo sigue conteniendo del 1 al 19 de marzo.
   Aca se toma el valor en el primer dia del mes: con `.shift(1).rolling(30)`
   eso equivale exactamente a los 30 dias anteriores al mes, sin un solo dia
   del periodo del objetivo.

`thp_per_user` no se corrige aca porque `score_sites.py` lo deriva de
`avg_THP_4G_30d / USERS_4G`: al quedar limpios sus dos insumos, queda limpio.

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
    """Agregados de KPI estrictamente anteriores al mes del objetivo.

    Para el sitio-mes M, todas las variables resumen los `window` dias que
    terminan el ultimo dia disponible antes de que empiece M. Ningun dia de M
    entra en ninguna variable.
    """
    kpis = kpis.sort_values(["site_id", "dia"]).reset_index(drop=True).copy()
    grp = kpis.groupby("site_id", group_keys=False)

    rolling_specs = {
        "avg_AVA_4G_30d": ("AVA_4G", "mean"),
        "avg_CSFR_V4G_30d": ("CSFR_V4G", "mean"),
        "max_DC_V4G_30d": ("DC_V4G", "max"),
        "avg_THP_4G_30d": ("THROUGHPUT_4G", "mean"),
        # USERS_4G pasa por la misma ventana previa, no es un atributo del sitio.
        "USERS_4G": ("USERS_4G", "mean"),
    }

    def rodante(col: str, w: int, how: str) -> pd.Series:
        """Rolling agrupado por sitio. El `.rolling()` va DENTRO del grupo.

        El original hace `grp[col].shift(1).rolling(...)`, y ese `.rolling()`
        corre sobre una Serie ya desagrupada, asi que la ventana cruza de un
        sitio al siguiente y contamina los primeros 30 dias de cada sitio.
        """
        return grp[col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).agg(how))

    for new_col, (src_col, how) in rolling_specs.items():
        if src_col not in kpis.columns:
            continue
        kpis[new_col] = rodante(src_col, window, how)

    # Degradacion = ventana corta contra ventana larga, ambas previas.
    if "THROUGHPUT_4G" in kpis.columns:
        corta = rodante("THROUGHPUT_4G", WINDOW_CORTA, "mean")
        kpis["degradacion_THP_30d"] = corta / kpis["avg_THP_4G_30d"].replace(0, np.nan)

    cols = [c for c in rolling_specs if c in kpis.columns]
    if "degradacion_THP_30d" in kpis.columns:
        cols.append("degradacion_THP_30d")

    # Para cada sitio-mes, el valor del ultimo dia disponible ANTES del mes.
    # `merge_asof` hacia atras cubre los sitio-mes cuyo dato no arranca el dia 1
    # (0,7% del total): en ese caso toma el cierre del mes anterior, que sigue
    # siendo estrictamente previo.
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
