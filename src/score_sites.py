"""
score_sites.py

Entrena XGBoost a grano mensual y genera el ranking de sitios en riesgo.

Prerrequisito:
    python src/build_modeling_dataset.py --grain monthly --source files

Uso:
    python src/score_sites.py
    python src/score_sites.py --n-trials 20 --top-k 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.metrics import (
    average_precision_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import optuna

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_MONTHLY_DEFAULT = BASE_DIR / "output" / "modeling_dataset_monthly.parquet"
OUTPUT_DIR_DEFAULT = BASE_DIR / "output" / "mensual-pronopro"

SITE_ID_COL = "site_id"
MONTH_COL = "mes"
TARGET_COL = "es_reclamo_confirmado_mensual"

FEATURE_COLS = [
    "avg_AVA_4G_30d",
    "max_DC_V4G_30d",
    "avg_THP_4G_30d",
    "degradacion_THP_30d",
    "avg_CSFR_V4G_30d",
    "USERS_4G",
    "thp_per_user",
]

AVA_THRESHOLD = 0.98
DC_THRESHOLD = 0.02
THP_THRESHOLD_MBP = 15.0

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Lee el parquet mensual y deriva thp_per_user."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Parquet no encontrado: {data_path}. "
            "Ejecutar `python src/build_modeling_dataset.py --grain monthly --source files` primero."
        )
    df_monthly = pd.read_parquet(data_path)
    df_monthly["thp_per_user"] = df_monthly["avg_THP_4G_30d"] / df_monthly["USERS_4G"].replace(0, np.nan)
    return df_monthly


def build_features(df_monthly: pd.DataFrame, test_cutoff: str) -> tuple[pd.DataFrame, ...]:
    """Descarta filas sin features y parte el dataset en train y test por fecha."""
    df_model = df_monthly.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS).copy()

    mask_train = df_model[MONTH_COL] < test_cutoff
    df_train = df_model[mask_train].copy()
    df_test = df_model[~mask_train].copy()

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL]

    return df_model, df_train, df_test, X_train, y_train, X_test, y_test


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_trials: int,
) -> XGBClassifier:
    """Busca hiperparametros con Optuna y devuelve el XGBoost entrenado."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="aucpr",
        )
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(y_val, prob))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params.update(scale_pos_weight=scale_pos_weight, random_state=random_state, eval_metric="aucpr")
    model_xgb = XGBClassifier(**best_params)
    model_xgb.fit(X_train, y_train)

    print(f"scale_pos_weight = {scale_pos_weight:.1f}")
    print(f"Optuna best AUC-PR (CV interno): {study.best_value:.4f}")
    print(f"Best params XGBoost: {best_params}")

    return model_xgb


def metrics_at_k(y_true: Sequence[int], y_prob: Sequence[float], k: int) -> dict:
    """Calcula TP, FP, precision y recall sobre las k alertas de mayor score."""
    order = np.argsort(-np.asarray(y_prob))
    top_k_true = np.asarray(y_true)[order][:k]
    tp = int(top_k_true.sum())
    fp = k - tp
    total_pos = int(np.asarray(y_true).sum())
    return {
        "k (alertas)": k,
        "TP": tp,
        "FP (falsas alarmas)": fp,
        "precision@k": round(tp / k, 4) if k else 0.0,
        "recall@k": round(tp / total_pos, 4) if total_pos else 0.0,
    }


def compute_shap_values(model_xgb: XGBClassifier, X_test: pd.DataFrame) -> np.ndarray:
    """Devuelve la matriz de valores SHAP del modelo sobre el conjunto de prueba."""
    explainer = shap.TreeExplainer(model_xgb)
    shap_values = explainer(X_test)
    return shap_values.values if hasattr(shap_values, "values") else shap_values


def clasifica_accion(shap_row: dict, signal_threshold: float) -> str:
    """Devuelve la accion recomendada segun el driver SHAP dominante del sitio."""
    driver_dominante = max(shap_row, key=shap_row.get)
    valor_maximo_shap = shap_row[driver_dominante]

    if valor_maximo_shap <= signal_threshold:
        return "Monitoreo"

    if any(kpi in driver_dominante.lower() for kpi in ["thp", "users", "traffic", "capacidad"]):
        return "Inversión tecnológica"
    elif any(kpi in driver_dominante.lower() for kpi in ["ava", "dc", "csfr", "drop", "disponibilidad"]):
        return "Mantenimiento programado"
    else:
        return "Monitoreo"


def cluster_sites(df_model: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Agrupa los sitios con K-Means y elige k por silhouette."""
    df_sites_agg = df_model.groupby(SITE_ID_COL)[FEATURE_COLS].mean().reset_index()

    scaler_cluster = StandardScaler()
    X_cluster = scaler_cluster.fit_transform(df_sites_agg[FEATURE_COLS])

    resultados_k = {}
    for k in range(3, 6):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_cluster)
        resultados_k[k] = silhouette_score(X_cluster, labels)

    best_k = max(resultados_k, key=resultados_k.get)
    print(f"Mejor k: {best_k} (silhouette={resultados_k[best_k]:.4f})")

    kmeans_final = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    df_sites_agg["cluster"] = kmeans_final.fit_predict(X_cluster)
    df_sites_agg["perfil"] = df_sites_agg.apply(_etiqueta_perfil, axis=1)

    return df_sites_agg[[SITE_ID_COL, "avg_AVA_4G_30d", "max_DC_V4G_30d", "avg_THP_4G_30d", "cluster", "perfil"]].rename(
        columns={"avg_AVA_4G_30d": "avg_AVA", "max_DC_V4G_30d": "avg_DC", "avg_THP_4G_30d": "avg_THP"}
    )


def _etiqueta_perfil(row: pd.Series) -> str:
    """Devuelve el perfil operativo del sitio segun sus KPI promedio."""
    if row["avg_AVA_4G_30d"] < 0.90:
        return "Crítico - disponibilidad severa"
    elif row["max_DC_V4G_30d"] > DC_THRESHOLD * 2:
        return "Crítico - caídas de llamada"
    elif row["avg_CSFR_V4G_30d"] > 0.05:
        return "Crítico - fallas de conexión (CSFR)"
    elif row["avg_THP_4G_30d"] < THP_THRESHOLD_MBP * 1.05:
        return "Congestión de capacidad"
    else:
        return "Estable"


def _umbrales_violados(row: pd.Series) -> str:
    """Devuelve los umbrales de negocio que incumple la fila."""
    violados = []
    if row["avg_AVA_4G_30d"] < AVA_THRESHOLD:
        violados.append("AVA")
    if row["max_DC_V4G_30d"] > DC_THRESHOLD:
        violados.append("DC")
    if row["avg_THP_4G_30d"] < THP_THRESHOLD_MBP:
        violados.append("THP")
    return ", ".join(violados) if violados else "ninguno"


def build_ranking(
    df_test: pd.DataFrame,
    X_test: pd.DataFrame,
    y_prob_xgb: np.ndarray,
    shap_array: np.ndarray,
    df_sites_clustered: pd.DataFrame,
    df_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Arma el ranking por sitio con score, drivers, umbrales, accion y perfil."""
    signal_threshold = 0.1 * np.abs(shap_array).mean()

    df_ranking = df_test[[SITE_ID_COL]].reset_index(drop=True).copy()
    df_ranking["score_riesgo"] = y_prob_xgb
    df_ranking["prioridad_sitio"] = pd.qcut(
        df_ranking["score_riesgo"], q=3, labels=["Baja", "Media", "Alta"]
    )

    top2_idx = np.argsort(-np.abs(shap_array), axis=1)[:, :2]
    df_ranking["kpis_driver"] = [
        ", ".join([FEATURE_COLS[i] for i in fila]) for fila in top2_idx
    ]

    df_ranking["umbral_incumplido"] = X_test.reset_index(drop=True).apply(_umbrales_violados, axis=1)
    df_ranking["accion_recomendada"] = [
        clasifica_accion(dict(zip(FEATURE_COLS, fila)), signal_threshold) for fila in shap_array
    ]

    df_ranking = df_ranking.merge(
        df_sites_clustered[[SITE_ID_COL, "cluster", "perfil"]], on=SITE_ID_COL, how="left"
    )

    meses_riesgo = df_monthly.groupby(SITE_ID_COL)[TARGET_COL].sum().rename("meses_en_riesgo")
    df_ranking = df_ranking.merge(meses_riesgo, on=SITE_ID_COL, how="left")

    df_ranking = (
        df_ranking
        .sort_values("score_riesgo", ascending=False)
        .drop_duplicates(subset=SITE_ID_COL, keep="first")
        .reset_index(drop=True)
    )
    return df_ranking.sort_values("score_riesgo", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Define los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_MONTHLY_DEFAULT, help="Parquet mensual de entrada")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT, help="Carpeta de salida")
    parser.add_argument("--test-cutoff", default="2026-02-01", help="Fecha de corte train/test (YYYY-MM-DD)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=50, help="Trials de Optuna para tuning de XGBoost")
    parser.add_argument("--top-k", type=int, default=77, help="N de alertas para precision@top-N")
    return parser.parse_args()


def main() -> None:
    """Punto de entrada: entrena, evalua y escribe el ranking de sitios."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_monthly = load_dataset(args.data)
    print(f"df_monthly cargado: {df_monthly.shape}")
    print(f"Positivos ({TARGET_COL}): {df_monthly[TARGET_COL].sum()} / {len(df_monthly)} = {df_monthly[TARGET_COL].mean():.2%}")

    df_model, df_train, df_test, X_train, y_train, X_test, y_test = build_features(df_monthly, args.test_cutoff)
    print(f"Train: {X_train.shape} | positivos={y_train.sum()} ({y_train.mean():.2%})")
    print(f"Test : {X_test.shape}  | positivos={y_test.sum()} ({y_test.mean():.2%})")

    model_xgb = train_xgboost(X_train, y_train, args.random_state, args.n_trials)
    y_prob_xgb = model_xgb.predict_proba(X_test)[:, 1]

    auc_pr = average_precision_score(y_test, y_prob_xgb)
    print(f"AUC-PR (test): {auc_pr:.4f}")

    df_topk = pd.DataFrame([metrics_at_k(y_test, y_prob_xgb, args.top_k)])
    print(f"=== Precisión operacional (top-{args.top_k} alertas) ===")
    print(df_topk.to_string(index=False))

    shap_array = compute_shap_values(model_xgb, X_test)
    df_sites_clustered = cluster_sites(df_model, args.random_state)

    df_ranking = build_ranking(df_test, X_test, y_prob_xgb, shap_array, df_sites_clustered, df_monthly)

    ranking_path = args.output_dir / "ranking_sitios.parquet"
    df_ranking.to_parquet(ranking_path, index=False)
    csv_path = args.output_dir / "ranking_sitios.csv"
    df_ranking.to_csv(csv_path, index=False)

    print(f"\nRanking guardado: {ranking_path}")
    print(f"Ranking guardado (CSV, lectura manual): {csv_path}")
    print(f"Total sitios en ranking: {len(df_ranking)}")
    print("\nTop 10 sitios de mayor riesgo:")
    print(df_ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
