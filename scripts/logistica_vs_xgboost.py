"""
logistica_vs_xgboost.py

Compara regresion logistica contra XGBoost sobre los dos pipelines, mide la
estabilidad de cada uno frente a la semilla y muestra los coeficientes de la
logistica como odds ratio.

Uso:
    uv run python scripts/logistica_vs_xgboost.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
import score_sites as ss  # noqa: E402

CUTOFF = "2026-02-01"
SEMILLAS = [42, 7, 123]
DATASETS = {
    "v1 (entregado)": BASE / "output" / "modeling_dataset_monthly_pronopro.parquet",
    "v2 (corregido)": BASE / "output_v2" / "modeling_dataset_monthly_v2.parquet",
}


def logistica(X, y, random_state: int):
    """Devuelve la regresion logistica balanceada ya entrenada."""
    modelo = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
    )
    return modelo.fit(X, y)


def main() -> None:
    """Imprime la comparacion, la estabilidad por semilla y los coeficientes."""
    for nombre, ruta in DATASETS.items():
        if not ruta.exists():
            print(f"\n{nombre}: parquet no encontrado, se omite ({ruta})")
            continue

        df = ss.load_dataset(ruta)
        _, _, _, X_train, y_train, X_test, y_test = ss.build_features(df, CUTOFF)
        prev = y_test.mean()
        print(f"\n=== {nombre}  (prevalencia {prev:.4%}) ===")

        lr = logistica(X_train, y_train, 42)
        xgb = ss.train_xgboost(X_train, y_train, 42, 50)
        for etiqueta, prob in [
            ("Regresion logistica", lr.predict_proba(X_test)[:, 1]),
            ("XGBoost + Optuna", xgb.predict_proba(X_test)[:, 1]),
        ]:
            ap = average_precision_score(y_test, prob)
            print(f"  {etiqueta:22} AUC-PR={ap:.4f}  lift={ap / prev:.2f}x")
        solo_users = average_precision_score(y_test, X_test["USERS_4G"])
        print(f"  {'Ordenar por USERS_4G':22} AUC-PR={solo_users:.4f}  lift={solo_users / prev:.2f}x")

        print("\n  Estabilidad frente a la semilla:")
        for semilla in SEMILLAS:
            l_ap = average_precision_score(y_test, logistica(X_train, y_train, semilla).predict_proba(X_test)[:, 1])
            x_ap = average_precision_score(y_test, ss.train_xgboost(X_train, y_train, semilla, 50).predict_proba(X_test)[:, 1])
            print(f"    semilla {semilla:>3}: xgboost={x_ap / prev:.3f}x   logistica={l_ap / prev:.3f}x")

        print("\n  Coeficientes de la logistica (odds ratio por desvio estandar):")
        coef = lr.named_steps["logisticregression"].coef_[0]
        for feature, beta in sorted(zip(ss.FEATURE_COLS, coef), key=lambda par: -abs(par[1])):
            print(f"    {feature:22} beta={beta:+.3f}   OR={np.exp(beta):.3f}")


if __name__ == "__main__":
    main()
