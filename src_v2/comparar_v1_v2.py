"""
comparar_v1_v2.py — TFM Grupo 1

Compara el pipeline entregado (`src/`, `output/`) contra el corregido
(`src_v2/`, `output_v2/`). No modifica ninguno de los dos.

Uso:
    uv run python src_v2/comparar_v1_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
import score_sites as ss  # noqa: E402

CUT = "2026-02-01"
DATOS = {
    "v1 (entregado)": BASE / "output" / "modeling_dataset_monthly_pronopro.parquet",
    "v2 (corregido)": BASE / "output_v2" / "modeling_dataset_monthly_v2.parquet",
}


def evaluar(nombre: str, ruta: Path, desde: str | None = None) -> dict:
    df = ss.load_dataset(ruta)
    if desde is not None:
        df = df[df["mes"].astype(str) >= desde]
    _, _, te, Xtr, ytr, Xte, yte = ss.build_features(df, CUT)
    modelo = ss.train_xgboost(Xtr, ytr, 42, 50)
    p = modelo.predict_proba(Xte)[:, 1]
    prev = yte.mean()
    ap = average_precision_score(yte, p)
    solo_users = average_precision_score(yte, Xte["USERS_4G"])
    orden = np.argsort(-p)
    y = yte.to_numpy()[orden]
    return {
        "nombre": nombre, "train": len(Xtr), "test": len(Xte),
        "pos_test": int(yte.sum()), "prev": prev, "auc_pr": ap, "lift": ap / prev,
        "lift_users": solo_users / prev,
        "p30": y[:30].mean(), "p77": y[:77].mean(),
        "sitios": te.assign(p=p).sort_values("p", ascending=False),
    }


def main() -> None:
    res = [evaluar(n, r) for n, r in DATOS.items()]

    print("\n" + "=" * 78)
    print("COMPARACION v1 (entregado) vs v2 (sin informacion del propio periodo)")
    print("=" * 78)
    fmt = "{:<18} {:>8} {:>7} {:>8} {:>9} {:>8} {:>9} {:>8}"
    print(fmt.format("dataset", "train", "test", "pos", "AUC-PR", "lift", "solo USERS", "prec@30"))
    for r in res:
        print(fmt.format(r["nombre"], r["train"], r["test"], r["pos_test"],
                         f"{r['auc_pr']:.4f}", f"{r['lift']:.2f}x",
                         f"{r['lift_users']:.2f}x", f"{r['p30']:.4f}"))

    a, b = res
    print(f"\nCaida de lift: {a['lift']:.2f}x -> {b['lift']:.2f}x "
          f"({(b['lift'] / a['lift'] - 1) * 100:+.0f}%)")
    print(f"Aporte del modelo sobre ordenar por usuarios:"
          f"  v1 {a['lift'] / a['lift_users']:.2f}x   v2 {b['lift'] / b['lift_users']:.2f}x")

    # Control: v2 pierde octubre por construccion (no tiene mes previo), asi que
    # entrena con 26% menos filas. Sin este control, la caida de lift mezcla el
    # efecto de quitar la contaminacion con el de entrenar con menos datos.
    ctrl = evaluar("v1 sin octubre", DATOS["v1 (entregado)"], desde="2025-11")
    print("\n--- Control: mismo periodo de entrenamiento (nov a ene) en ambos ---")
    for r in [res[0], ctrl, res[1]]:
        print(f"  {r['nombre']:<20} train={r['train']:>5}  AUC-PR={r['auc_pr']:.4f}  lift={r['lift']:.2f}x")
    print(f"\n  Con el mismo periodo, contaminado y limpio dan {ctrl['lift']:.2f}x y {res[1]['lift']:.2f}x.")
    print("  La caida respecto de 1,61x es el mes de entrenamiento perdido, no la contaminacion.")

    print("\n--- Coincidencia del ranking (que tanto cambia la lista entregada) ---")
    for k in [10, 30, 77, 200]:
        s1 = set(a["sitios"].head(k).site_id)
        s2 = set(b["sitios"].head(k).site_id)
        print(f"  top-{k:<4}: {len(s1 & s2):>3} sitios en comun de {k}  ({len(s1 & s2) / k:.0%})")


if __name__ == "__main__":
    main()
