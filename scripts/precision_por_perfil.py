"""
precision_por_perfil.py

Cruza el ranking del modelo con el perfil operativo del clustering y reporta
el desempeno dentro de cada perfil y la composicion del top-N global.

Uso:
    uv run python scripts/precision_por_perfil.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

import score_sites as ss  # noqa: E402

TEST_CUTOFF = "2026-02-01"
RANDOM_STATE = 42
N_TRIALS = 50
CORTES = (30, 50, 100)


def build_scored_test() -> pd.DataFrame:
    """Entrena el modelo y devuelve el test con score, etiqueta real y perfil."""
    df_monthly = ss.load_dataset(ss.DATA_MONTHLY_DEFAULT)
    df_model, _, df_test, X_train, y_train, X_test, y_test = ss.build_features(df_monthly, TEST_CUTOFF)

    model = ss.train_xgboost(X_train, y_train, RANDOM_STATE, N_TRIALS)
    y_prob = model.predict_proba(X_test)[:, 1]

    clusters = ss.cluster_sites(df_model, RANDOM_STATE)

    scored = df_test[[ss.SITE_ID_COL]].reset_index(drop=True).copy()
    scored["y_real"] = y_test.reset_index(drop=True)
    scored["score"] = y_prob
    return scored.merge(clusters[[ss.SITE_ID_COL, "perfil"]], on=ss.SITE_ID_COL, how="left")


def vista_a(scored: pd.DataFrame) -> pd.DataFrame:
    """Desempeño dentro de cada perfil, con k proporcional al tamaño del grupo."""
    filas = []
    for perfil, grupo in scored.groupby("perfil"):
        positivos = int(grupo["y_real"].sum())
        fila = {
            "perfil": perfil,
            "filas test": len(grupo),
            "positivos": positivos,
            "tasa base %": round(100 * positivos / len(grupo), 2),
        }
        if positivos == 0:
            fila["AUC-PR"] = None
            fila["precision@k"] = None
        else:
            k = max(1, round(CORTES[0] * len(grupo) / len(scored)))
            top_k = grupo.sort_values("score", ascending=False).head(k)
            tp = int(top_k["y_real"].sum())
            fila["AUC-PR"] = round(average_precision_score(grupo["y_real"], grupo["score"]), 4)
            fila["precision@k"] = f"{tp}/{k} = {100 * tp / k:.1f}%"
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("filas test", ascending=False)


def vista_b(scored: pd.DataFrame, n: int) -> pd.DataFrame:
    """Dónde caen las alertas del top-N global."""
    top = scored.sort_values("score", ascending=False).head(n)
    comp = top.groupby("perfil").agg(alertas=("y_real", "size"), aciertos=("y_real", "sum"))
    comp["falsas alarmas"] = comp["alertas"] - comp["aciertos"]
    comp["% de las alertas"] = (100 * comp["alertas"] / n).round(1)
    comp["precision %"] = (100 * comp["aciertos"] / comp["alertas"]).round(1)
    return comp.sort_values("alertas", ascending=False)


def main() -> None:
    scored = build_scored_test()
    print(f"\nTest: {len(scored)} filas sitio-mes, {int(scored['y_real'].sum())} positivos\n")

    print("=" * 78)
    print("A. Desempeño dentro de cada perfil")
    print("=" * 78)
    print(vista_a(scored).to_string(index=False))

    print("\n" + "=" * 78)
    print("B. Composición del top-N global")
    print("=" * 78)
    for n in CORTES:
        top = scored.sort_values("score", ascending=False).head(n)
        tp = int(top["y_real"].sum())
        print(f"\ntop-{n}: TP={tp}, FP={n - tp}, precisión global={100 * tp / n:.1f}%")
        print(vista_b(scored, n).to_string())


if __name__ == "__main__":
    main()
