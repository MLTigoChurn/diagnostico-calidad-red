"""generate_synthetic_claims.py — TFM Grupo 1

Sanity-check del pipeline de modelado: genera un dataset SINTÉTICO en
`db_files_sint/` con una relación KPI->queja conocida e inyectada a propósito
(sitios con `avg_AVA_4G_7d` por debajo del umbral de negocio tienen mucha más
probabilidad de "queja" sintética). Si el pipeline (build_weekly + XGBoost +
SHAP) detecta esta señal con claridad, confirma que la metodología es sólida
y que el resultado débil sobre datos REALES es una limitación del dataset,
no un error de implementación.

No usa datos reales de `claims.csv`/`db_files/` para nada -- location y kpis
se copian sin cambios desde los `-v1` (no se usan en el pipeline real
actualmente), y las claims son 100% fabricadas y marcadas como tales
(`category_3='SINTETICO'`, descripción explícita).

Uso:
    python scripts/generate_synthetic_claims.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.build_modeling_dataset import normalize, build_weekly_kpis, WEEK_ANCHOR  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "db_files"
DST_DIR = BASE_DIR / "db_files_sint"

RANDOM_STATE = 42
AVA_THRESHOLD = 0.98

# Curva de probabilidad de queja sintética en función del déficit de AVA:
# sitios sanos (AVA >= umbral) -> BASELINE_RATE; sitios muy degradados
# (AVA <= umbral - RAMP_WIDTH) -> MAX_RATE. Rampa lineal entre ambos.
BASELINE_RATE = 0.01
MAX_RATE = 0.30
RAMP_WIDTH = 0.05


def _synthetic_probability(avg_ava: pd.Series) -> pd.Series:
    deficit = (AVA_THRESHOLD - avg_ava).clip(lower=0)
    ramp = (deficit / RAMP_WIDTH).clip(upper=1)
    return BASELINE_RATE + (MAX_RATE - BASELINE_RATE) * ramp


def main() -> None:
    DST_DIR.mkdir(exist_ok=True)

    shutil.copy(SRC_DIR / "location-v1.csv", DST_DIR / "location.csv")
    shutil.copy(SRC_DIR / "kpis_diario-v1.txt", DST_DIR / "kpis_diario.txt")
    print(f"Copiados location.csv y kpis_diario.txt (sin cambios) a {DST_DIR}")

    location = pd.read_csv(DST_DIR / "location.csv", sep=",", encoding="utf-8", low_memory=False)
    kpis = pd.read_csv(DST_DIR / "kpis_diario.txt", sep="\t", encoding="utf-8", low_memory=False)

    # Reusa normalize() con un claims dummy vacío -- solo nos interesan kpis/location aquí.
    dummy_claims = pd.DataFrame(columns=["user_id", "fecha_creacion"])
    dummy_claims, location, kpis = normalize(dummy_claims, location, kpis)

    kpis_weekly = build_weekly_kpis(kpis)
    kpis_weekly = kpis_weekly.dropna(subset=["avg_AVA_4G_7d"]).copy()
    kpis_weekly["p_queja_sint"] = _synthetic_probability(kpis_weekly["avg_AVA_4G_7d"])

    rng = np.random.default_rng(RANDOM_STATE)
    kpis_weekly["hit"] = rng.random(len(kpis_weekly)) < kpis_weekly["p_queja_sint"]
    hits = kpis_weekly[kpis_weekly["hit"]].copy()

    site_users = location.groupby("site_id")["user_id"].apply(list).to_dict()
    hits = hits[hits["site_id"].isin(site_users.keys())].copy()

    def _pick_user(site_id: str) -> str:
        users = site_users[site_id]
        return users[rng.integers(0, len(users))]

    hits["user_id"] = hits["site_id"].map(_pick_user)
    week_start = hits["semana"].dt.start_time
    day_offset = pd.to_timedelta(rng.integers(0, 7, size=len(hits)), unit="D")
    seconds_offset = pd.to_timedelta(rng.integers(0, 86400, size=len(hits)), unit="s")
    hits["fecha_creacion"] = week_start + day_offset + seconds_offset

    claims_sint = pd.DataFrame({
        "user_id": hits["user_id"].to_numpy(),
        "nro_ticket": [f"SINT{i:08d}" for i in range(len(hits))],
        "fecha_creacion": hits["fecha_creacion"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "motivo_atencion": "RECLAMO",
        "category_1": "RED",
        "category_2": "CALIDAD DE RED/COBERTURA DE RED",
        "category_3": "SINTETICO",
        "description": [
            f"SINTÉTICO - dato generado para validación de pipeline (avg_AVA_4G_7d={v:.3f})"
            for v in hits["avg_AVA_4G_7d"]
        ],
        "segmento": "SINTETICO",
    })

    claims_sint.to_csv(DST_DIR / "claims.csv", sep="|", index=False, encoding="utf-8-sig")

    print(f"\nClaims sintéticas generadas: {len(claims_sint)}")
    print(f"Tasa de hit global: {kpis_weekly['hit'].mean():.2%} "
          f"(sobre {len(kpis_weekly)} site-semanas con avg_AVA_4G_7d disponible)")
    print("\nTasa de hit por bucket AVA (chequeo de la señal inyectada):")
    bucket = np.where(kpis_weekly["avg_AVA_4G_7d"] < AVA_THRESHOLD, f"< {AVA_THRESHOLD}", f">= {AVA_THRESHOLD}")
    print(kpis_weekly.groupby(bucket)["hit"].mean())
    print(f"\nEscrito en: {DST_DIR / 'claims.csv'}")


if __name__ == "__main__":
    main()
