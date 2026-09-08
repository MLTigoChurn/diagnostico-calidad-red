"""
build_modeling_dataset.py

Unifica claims, location y kpis_diario en el dataset de modelado.

Uso:
    python src/build_modeling_dataset.py --grain weekly  --source files
    python src/build_modeling_dataset.py --grain monthly --source files
    python src/build_modeling_dataset.py --grain daily   --source files
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILES_DIR = (BASE_DIR / "db_files").resolve()
OUTPUT_PATH = (BASE_DIR / "output" / "modeling_dataset.parquet").resolve()
OUTPUT_PATH_WEEKLY = (BASE_DIR / "output" / "modeling_dataset_weekly.parquet").resolve()
OUTPUT_PATH_MONTHLY = (BASE_DIR / "output" / "modeling_dataset_monthly.parquet").resolve()

ROLLING_WINDOW_DAYS = 7
ROLLING_WINDOW_DAYS_MONTHLY = 30

WEEK_ANCHOR = "W-SUN"
MONTH_ANCHOR = "M"

ID_COLS = ("user_id", "site_id")


def _resolve_kpis_path(dir_path: Path) -> tuple[Path, str]:
    """Devuelve la ruta del archivo de KPIs y su separador."""
    csv_path = dir_path / "kpis_diario.csv"
    if csv_path.exists():
        return csv_path, ","
    return dir_path / "kpis_diario.txt", "\t"


def load_from_files(dir_path: Path = DB_FILES_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lee claims, location y kpis desde archivos locales."""
    if not dir_path.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {dir_path}")

    claims = pd.read_csv(
        dir_path / "claims.csv",
        sep="|",
        encoding="utf-8-sig",
        encoding_errors="replace",
        low_memory=False,
    )
    if "category_1" in claims.columns:
        claims = claims[claims["category_1"] == "RED"].copy()
    location = pd.read_csv(
        dir_path / "location.csv",
        sep=",",
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )
    kpis_path, kpis_sep = _resolve_kpis_path(dir_path)
    kpis = pd.read_csv(
        kpis_path,
        sep=kpis_sep,
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )
    return claims, location, kpis


def load_from_mysql() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lee claims, location y kpis desde MySQL. No implementado."""
    raise NotImplementedError(
        "Fuente MySQL pendiente. Requiere sqlalchemy + pymysql y credenciales "
        "en .env (MYSQL_HOST/PORT/USER/PASSWORD/DB)."
    )


def load_sources(source: str, data_dir: Path = DB_FILES_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Despacha la carga segun el origen elegido."""
    if source == "files":
        return load_from_files(data_dir)
    if source == "mysql":
        return load_from_mysql()
    raise ValueError(f"Fuente desconocida: {source!r} (usar 'files' o 'mysql')")


def normalize(
    claims: pd.DataFrame, location: pd.DataFrame, kpis: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pasa los IDs a string y las fechas a datetime truncado a dia."""
    for df in (claims, location, kpis):
        for col in ID_COLS:
            if col in df.columns:
                df[col] = df[col].astype("string").str.strip()

    claims["fecha_creacion"] = pd.to_datetime(claims["fecha_creacion"], errors="coerce")
    claims["dia"] = claims["fecha_creacion"].dt.normalize()

    # FECHA viene en DD/MM/YYYY.
    kpis["FECHA"] = pd.to_datetime(kpis["FECHA"], format="%d/%m/%Y", errors="coerce")
    kpis["dia"] = kpis["FECHA"].dt.normalize()

    return claims, location, kpis


def validate_integrity(claims: pd.DataFrame, location: pd.DataFrame, kpis: pd.DataFrame) -> None:
    """Imprime la cobertura de las claves de join y aborta si alguna queda vacia."""
    claims_users = set(claims["user_id"].dropna())
    loc_users = set(location["user_id"].dropna())
    loc_sites = set(location["site_id"].dropna())
    kpi_sites = set(kpis["site_id"].dropna())

    def pct(num: int, den: int) -> str:
        return f"{(100.0 * num / den):.1f}%" if den else "n/a"

    matched_users = claims_users & loc_users
    matched_sites = loc_sites & kpi_sites

    print("=== Validación de integridad referencial ===")
    print(f"claims.user_id con match en location : {pct(len(matched_users), len(claims_users))} "
          f"({len(matched_users)}/{len(claims_users)})")
    print(f"location.site_id con match en kpis    : {pct(len(matched_sites), len(loc_sites))} "
          f"({len(matched_sites)}/{len(loc_sites)})")

    if not matched_users:
        raise RuntimeError("Ningún user_id de claims cruza con location.")
    if not matched_sites:
        raise RuntimeError("Ningún site_id de location cruza con kpis.")


def build_target(claims: pd.DataFrame, location: pd.DataFrame) -> pd.DataFrame:
    """Agrega claims a site_id x dia, desglosado por motivo de atencion."""
    if "motivo_atencion" not in claims.columns:
        raise RuntimeError(
            "claims no tiene 'motivo_atencion': no se puede separar RECLAMO de CONSULTA."
        )

    user_site = location[["user_id", "site_id"]].drop_duplicates(subset=["user_id"])
    cl = claims.merge(user_site, on="user_id", how="inner")

    motivo = cl["motivo_atencion"].astype("string").str.upper().str.strip()
    cl = cl.assign(
        es_reclamo=(motivo == "RECLAMO").astype(int),
        es_consulta=(motivo == "CONSULTA").astype(int),
    )

    agg = (
        cl.groupby(["site_id", "dia"])
        .agg(
            n_reclamos=("es_reclamo", "sum"),
            n_consultas=("es_consulta", "sum"),
            n_claims_total=("nro_ticket", "count"),
        )
        .reset_index()
    )
    agg["es_reclamo_red"] = (agg["n_reclamos"] > 0).astype(int)
    return agg


def attach_kpis(target: pd.DataFrame, kpis: pd.DataFrame) -> pd.DataFrame:
    """Adjunta el KPI del dia mas agregados rolling de 7 dias previos."""
    kpis = kpis.sort_values(["site_id", "dia"]).copy()

    rolling_specs = {
        "avg_AVA_4G_7d": ("AVA_4G", "mean"),
        "avg_CSFR_V4G_7d": ("CSFR_V4G", "mean"),
        "max_DC_V4G_7d": ("DC_V4G", "max"),
        "avg_THROUGHPUT_4G_7d": ("THROUGHPUT_4G", "mean"),
    }

    grp = kpis.groupby("site_id", group_keys=False)
    for new_col, (src_col, how) in rolling_specs.items():
        if src_col not in kpis.columns:
            continue
        shifted = grp[src_col].shift(1)
        kpis[new_col] = shifted.rolling(ROLLING_WINDOW_DAYS, min_periods=1).agg(how)

    if "THROUGHPUT_4G" in kpis.columns and "avg_THROUGHPUT_4G_7d" in kpis.columns:
        kpis["degradacion_THROUGHPUT_7d"] = (
            kpis["THROUGHPUT_4G"] / kpis["avg_THROUGHPUT_4G_7d"]
        )

    return target.merge(kpis, on=["site_id", "dia"], how="left")


def build_weekly_target(
    claims: pd.DataFrame, location: pd.DataFrame, week_anchor: str = WEEK_ANCHOR
) -> pd.DataFrame:
    """Agrega claims a site_id x semana. Anade el target confirmado si hay pronopro."""
    user_site = location[["user_id", "site_id"]].drop_duplicates(subset=["user_id"])
    cl = claims.merge(user_site, on="user_id", how="inner")

    motivo = cl["motivo_atencion"].astype("string").str.upper().str.strip()
    cl = cl.assign(es_reclamo=(motivo == "RECLAMO").astype(int))
    cl["semana"] = cl["dia"].dt.to_period(week_anchor)

    agg_specs = {
        "n_reclamos_semana": ("es_reclamo", "sum"),
        "n_claims_semana": ("nro_ticket", "count"),
    }

    if "pronopro" in cl.columns:
        procede = (cl["pronopro"].astype("string").str.strip().str.casefold() == "procede").fillna(False)
        cl = cl.assign(es_reclamo_confirmado=(cl["es_reclamo"].astype(bool) & procede).astype(int))
        agg_specs["n_reclamos_confirmados_semana"] = ("es_reclamo_confirmado", "sum")
        cobertura = cl.loc[cl["es_reclamo"] == 1, "pronopro"].notna().mean()
        print(f"Cobertura de PRONOPRO en RECLAMO: {cobertura:.1%}")

    agg = cl.groupby(["site_id", "semana"]).agg(**agg_specs).reset_index()
    agg["es_reclamo_semanal"] = (agg["n_reclamos_semana"] > 0).astype(int)
    if "n_reclamos_confirmados_semana" in agg.columns:
        agg["es_reclamo_confirmado_semanal"] = (agg["n_reclamos_confirmados_semana"] > 0).astype(int)
    return agg


def build_weekly_kpis(
    kpis: pd.DataFrame,
    week_anchor: str = WEEK_ANCHOR,
    window: int = ROLLING_WINDOW_DAYS,
) -> pd.DataFrame:
    """Calcula agregados rolling diarios y los resume a site_id x semana."""
    kpis = kpis.sort_values(["site_id", "dia"]).copy()

    rolling_specs = {
        "avg_AVA_4G_7d": ("AVA_4G", "mean"),
        "avg_CSFR_V4G_7d": ("CSFR_V4G", "mean"),
        "max_DC_V4G_7d": ("DC_V4G", "max"),
        "avg_THP_4G_7d": ("THROUGHPUT_4G", "mean"),
    }
    grp = kpis.groupby("site_id", group_keys=False)
    for new_col, (src_col, how) in rolling_specs.items():
        if src_col not in kpis.columns:
            continue
        shifted = grp[src_col].shift(1)
        kpis[new_col] = shifted.rolling(window, min_periods=1).agg(how)

    if "THROUGHPUT_4G" in kpis.columns and "avg_THP_4G_7d" in kpis.columns:
        kpis["degradacion_THP_7d"] = kpis["THROUGHPUT_4G"] / kpis["avg_THP_4G_7d"].replace(0, np.nan)

    kpis["semana"] = kpis["dia"].dt.to_period(week_anchor)

    weekly_agg_specs = {
        "avg_AVA_4G_7d": "mean",
        "max_DC_V4G_7d": "max",
        "avg_THP_4G_7d": "mean",
        "degradacion_THP_7d": "mean",
        "avg_CSFR_V4G_7d": "mean",
        "USERS_4G": "mean",
    }
    present = {c: how for c, how in weekly_agg_specs.items() if c in kpis.columns}
    return kpis.groupby(["site_id", "semana"]).agg(present).reset_index()


def build_monthly_target(
    claims: pd.DataFrame, location: pd.DataFrame, month_anchor: str = MONTH_ANCHOR
) -> pd.DataFrame:
    """Agrega claims a site_id x mes. Anade el target confirmado si hay pronopro."""
    user_site = location[["user_id", "site_id"]].drop_duplicates(subset=["user_id"])
    cl = claims.merge(user_site, on="user_id", how="inner")

    motivo = cl["motivo_atencion"].astype("string").str.upper().str.strip()
    cl = cl.assign(es_reclamo=(motivo == "RECLAMO").astype(int))
    cl["mes"] = cl["dia"].dt.to_period(month_anchor)

    agg_specs = {
        "n_reclamos_mes": ("es_reclamo", "sum"),
        "n_claims_mes": ("nro_ticket", "count"),
    }

    if "pronopro" in cl.columns:
        procede = (cl["pronopro"].astype("string").str.strip().str.casefold() == "procede").fillna(False)
        cl = cl.assign(es_reclamo_confirmado=(cl["es_reclamo"].astype(bool) & procede).astype(int))
        agg_specs["n_reclamos_confirmados_mes"] = ("es_reclamo_confirmado", "sum")

    agg = cl.groupby(["site_id", "mes"]).agg(**agg_specs).reset_index()
    agg["es_reclamo_mensual"] = (agg["n_reclamos_mes"] > 0).astype(int)
    if "n_reclamos_confirmados_mes" in agg.columns:
        agg["es_reclamo_confirmado_mensual"] = (agg["n_reclamos_confirmados_mes"] > 0).astype(int)
    return agg


def build_monthly_kpis(
    kpis: pd.DataFrame,
    month_anchor: str = MONTH_ANCHOR,
    window: int = ROLLING_WINDOW_DAYS_MONTHLY,
) -> pd.DataFrame:
    """Calcula agregados rolling diarios de 30 dias y los resume a site_id x mes."""
    kpis = kpis.sort_values(["site_id", "dia"]).copy()

    rolling_specs = {
        "avg_AVA_4G_30d": ("AVA_4G", "mean"),
        "avg_CSFR_V4G_30d": ("CSFR_V4G", "mean"),
        "max_DC_V4G_30d": ("DC_V4G", "max"),
        "avg_THP_4G_30d": ("THROUGHPUT_4G", "mean"),
    }
    grp = kpis.groupby("site_id", group_keys=False)
    for new_col, (src_col, how) in rolling_specs.items():
        if src_col not in kpis.columns:
            continue
        shifted = grp[src_col].shift(1)
        kpis[new_col] = shifted.rolling(window, min_periods=1).agg(how)

    if "THROUGHPUT_4G" in kpis.columns and "avg_THP_4G_30d" in kpis.columns:
        kpis["degradacion_THP_30d"] = kpis["THROUGHPUT_4G"] / kpis["avg_THP_4G_30d"].replace(0, np.nan)

    kpis["mes"] = kpis["dia"].dt.to_period(month_anchor)

    monthly_agg_specs = {
        "avg_AVA_4G_30d": "mean",
        "max_DC_V4G_30d": "max",
        "avg_THP_4G_30d": "mean",
        "degradacion_THP_30d": "mean",
        "avg_CSFR_V4G_30d": "mean",
        "USERS_4G": "mean",
    }
    present = {c: how for c, how in monthly_agg_specs.items() if c in kpis.columns}
    return kpis.groupby(["site_id", "mes"]).agg(present).reset_index()


def build_monthly(
    source: str, output_path: Path = OUTPUT_PATH_MONTHLY, data_dir: Path = DB_FILES_DIR
) -> pd.DataFrame:
    """Construye y escribe el dataset a grano site_id x mes."""
    claims, location, kpis = load_sources(source, data_dir)
    claims, location, kpis = normalize(claims, location, kpis)
    validate_integrity(claims, location, kpis)

    kpis_monthly = build_monthly_kpis(kpis)
    target = build_monthly_target(claims, location)

    dataset = kpis_monthly.merge(target, on=["site_id", "mes"], how="left")
    count_cols = ["n_reclamos_mes", "n_claims_mes"]
    if "n_reclamos_confirmados_mes" in dataset.columns:
        count_cols.append("n_reclamos_confirmados_mes")
    for col in count_cols:
        dataset[col] = dataset[col].fillna(0).astype(int)
    dataset["es_reclamo_mensual"] = (dataset["n_reclamos_mes"] > 0).astype(int)
    if "n_reclamos_confirmados_mes" in dataset.columns:
        dataset["es_reclamo_confirmado_mensual"] = (dataset["n_reclamos_confirmados_mes"] > 0).astype(int)

    if "USERS_4G" in dataset.columns:
        dataset["reclamos_por_usuario"] = dataset["n_reclamos_mes"] / dataset["USERS_4G"].replace(0, pd.NA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    print(f"\nDataset mensual (site×mes): {dataset.shape[0]} filas, {dataset.shape[1]} columnas")
    print(f"Escrito en: {output_path}")
    return dataset


def build_weekly(
    source: str, output_path: Path = OUTPUT_PATH_WEEKLY, data_dir: Path = DB_FILES_DIR
) -> pd.DataFrame:
    """Construye y escribe el dataset a grano site_id x semana."""
    claims, location, kpis = load_sources(source, data_dir)
    claims, location, kpis = normalize(claims, location, kpis)
    validate_integrity(claims, location, kpis)

    kpis_weekly = build_weekly_kpis(kpis)
    target = build_weekly_target(claims, location)

    dataset = kpis_weekly.merge(target, on=["site_id", "semana"], how="left")
    count_cols = ["n_reclamos_semana", "n_claims_semana"]
    if "n_reclamos_confirmados_semana" in dataset.columns:
        count_cols.append("n_reclamos_confirmados_semana")
    for col in count_cols:
        dataset[col] = dataset[col].fillna(0).astype(int)
    dataset["es_reclamo_semanal"] = (dataset["n_reclamos_semana"] > 0).astype(int)
    if "n_reclamos_confirmados_semana" in dataset.columns:
        dataset["es_reclamo_confirmado_semanal"] = (dataset["n_reclamos_confirmados_semana"] > 0).astype(int)

    if "USERS_4G" in dataset.columns:
        dataset["reclamos_por_usuario"] = dataset["n_reclamos_semana"] / dataset["USERS_4G"].replace(0, pd.NA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    print(f"\nDataset semanal (site×semana): {dataset.shape[0]} filas, {dataset.shape[1]} columnas")
    print(f"Escrito en: {output_path}")
    return dataset


def build(source: str, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Construye y escribe el dataset a grano site_id x dia."""
    claims, location, kpis = load_sources(source)
    claims, location, kpis = normalize(claims, location, kpis)
    validate_integrity(claims, location, kpis)

    target = build_target(claims, location)
    dataset = attach_kpis(target, kpis)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    print(f"\nDataset de modelado: {dataset.shape[0]} filas, {dataset.shape[1]} columnas")
    print(f"Escrito en: {output_path}")
    return dataset


def parse_args() -> argparse.Namespace:
    """Define los argumentos de linea de comandos."""
    p = argparse.ArgumentParser(description="Unifica los datasets en el dataset de modelado.")
    p.add_argument("--source", choices=["files", "mysql"], default="files",
                   help="Origen de los datos (default: files).")
    p.add_argument("--grain", choices=["weekly", "monthly", "daily"], default="weekly",
                   help="Grano de salida (default: weekly).")
    p.add_argument("--output", type=Path, default=None,
                   help="Ruta del parquet de salida (default segun --grain).")
    p.add_argument("--data-dir", type=Path, default=DB_FILES_DIR,
                   help="Directorio de datasets crudos (default: db_files/).")
    return p.parse_args()


def main() -> None:
    """Punto de entrada: construye el dataset del grano pedido."""
    args = parse_args()
    if args.grain == "weekly":
        build_weekly(args.source, args.output or OUTPUT_PATH_WEEKLY, args.data_dir)
    elif args.grain == "monthly":
        build_monthly(args.source, args.output or OUTPUT_PATH_MONTHLY, args.data_dir)
    else:
        build(args.source, args.output or OUTPUT_PATH)


if __name__ == "__main__":
    main()
