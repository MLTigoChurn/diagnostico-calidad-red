"""
precision_por_k.py — TFM Grupo 1

Curva precision@k y lift@k del modelo final, sobre el mismo split que
`src/score_sites.py` (grano mensual, target confirmado por PRONOPRO,
corte 2026-02-01). Sirve para responder en la defensa por qué se reporta
lift y no precision a secas: precision@k se mueve un factor 8 sin que el
modelo cambie, y lift@k converge a 1,00x cuando k es todo el universo.

Uso:
    uv run python scripts/precision_por_k.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
import score_sites as ss  # noqa: E402

DATA = BASE / "output" / "modeling_dataset_monthly_pronopro.parquet"
CUTOFF = "2026-02-01"
K_GRID = [10, 20, 30, 50, 77, 100, 150, 200, 300, 500, 1000]


def ic95(tp: int, k: int) -> tuple[float, float]:
    """Intervalo de Clopper-Pearson para precision@k."""
    lo = beta.ppf(0.025, tp, k - tp + 1) if tp > 0 else 0.0
    hi = beta.ppf(0.975, tp + 1, k - tp) if tp < k else 1.0
    return lo, hi


def main() -> None:
    df = ss.load_dataset(DATA)
    _, _, _, X_train, y_train, X_test, y_test = ss.build_features(df, CUTOFF)

    model = ss.train_xgboost(X_train, y_train, 42, 50)
    y_prob = model.predict_proba(X_test)[:, 1]

    prev = y_test.mean()
    ap = average_precision_score(y_test, y_prob)
    print(f"\nprevalencia = {prev:.4%}   (piso de precision y de AUC-PR)")
    print(f"AUC-PR = {ap:.4f}   lift = {ap / prev:.2f}x   AUC-ROC = {roc_auc_score(y_test, y_prob):.4f}")
    print(f"accuracy de un modelo que predice todo negativo = {1 - prev:.2%}\n")

    y_sorted = y_test.to_numpy()[np.argsort(-y_prob)]
    total_pos = int(y_sorted.sum())

    print(f"{'k':>6} {'TP':>4} {'FP':>5} {'precision@k':>12} {'lift@k':>8} {'recall@k':>9} {'IC95% lift':>18}")
    for k in K_GRID + [len(y_sorted)]:
        tp = int(y_sorted[:k].sum())
        lo, hi = ic95(tp, k)
        ic = f"[{lo / prev:.2f}x, {hi / prev:.2f}x]"
        print(f"{k:>6} {tp:>4} {k - tp:>5} {tp / k:>12.4f} {tp / k / prev:>7.2f}x "
              f"{tp / total_pos:>9.3f} {ic:>18}")

    print("\nlift@k converge a 1,00x en k = N por construccion: es precision@k / prevalencia.")


if __name__ == "__main__":
    main()
