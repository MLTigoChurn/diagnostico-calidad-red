"""
verificacion_hallazgos.py

Comprueba sobre el split del modelo final si las variables usan informacion
del propio periodo, que aporta el modelo frente a baselines triviales, como
cambia el top-N calculado por mes y si el score puede leerse como probabilidad.

Uso:
    uv run python scripts/verificacion_hallazgos.py
"""

import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
import score_sites as ss

CUT = "2026-02-01"
df = ss.load_dataset(BASE / "output" / "modeling_dataset_monthly_pronopro.parquet")

def rep(nombre, y, s):
    """Imprime AUC-PR y lift de un scorer y devuelve el lift."""
    prev = y.mean()
    ap = average_precision_score(y, s)
    print(f"{nombre:<42} AUC-PR={ap:.4f}  lift={ap/prev:.2f}x")
    return ap/prev

_, tr, te, Xtr, ytr, Xte, yte = ss.build_features(df, CUT)
m = ss.train_xgboost(Xtr, ytr, 42, 50)
p_act = m.predict_proba(Xte)[:, 1]
print("\n=== A. REFERENCIA Y BASELINES TRIVIALES (mismo test) ===")
rep("XGBoost actual", yte, p_act)

rep("Ordenar por USERS_4G solo", yte, Xte["USERS_4G"].to_numpy())
rep("Ordenar por degradacion_THP_30d solo", yte, Xte["degradacion_THP_30d"].to_numpy())
rep("Ordenar por avg_THP_4G_30d (invertido)", yte, -Xte["avg_THP_4G_30d"].to_numpy())

LIMPIAS = ["avg_AVA_4G_30d", "max_DC_V4G_30d", "avg_THP_4G_30d", "avg_CSFR_V4G_30d"]
orig = ss.FEATURE_COLS[:]
ss.FEATURE_COLS = LIMPIAS
_, _, te2, Xtr2, ytr2, Xte2, yte2 = ss.build_features(df, CUT)
m2 = ss.train_xgboost(Xtr2, ytr2, 42, 50)
print("\n=== C. SIN LAS FEATURES CONTAMINADAS (4 vars) ===")
rep("XGBoost sin USERS/thp_per_user/degradacion", yte2, m2.predict_proba(Xte2)[:, 1])
ss.FEATURE_COLS = orig

d = df.sort_values(["site_id", "mes"]).copy()
lag = d.groupby("site_id")[orig].shift(1)
lag.columns = [c + "_lag1" for c in orig]
d = pd.concat([d[["site_id", "mes", ss.TARGET_COL]], lag], axis=1)
ss.FEATURE_COLS = list(lag.columns)
_, _, te3, Xtr3, ytr3, Xte3, yte3 = ss.build_features(d, CUT)
m3 = ss.train_xgboost(Xtr3, ytr3, 42, 50)
print("\n=== D. FEATURES DEL MES ANTERIOR (anti-fuga real) ===")
print(f"filas train={len(Xtr3)} test={len(Xte3)} positivos test={int(yte3.sum())}")
rep("XGBoost con features de t-1", yte3, m3.predict_proba(Xte3)[:, 1])
ss.FEATURE_COLS = orig

print("\n=== E. TOP-N POR MES vs TOP-N GLOBAL (hallazgo 6) ===")
r = te[["site_id", "mes"]].copy(); r["y"] = yte.to_numpy(); r["p"] = p_act
prev = yte.mean()
for k in [30, 38, 77]:
    tp_g = r.nlargest(k, "p").y.sum()
    print(f"  global top-{k:<3}: TP={tp_g:>3}  precision={tp_g/k:.4f}  lift={tp_g/k/prev:.2f}x")
for k in [15, 30]:
    g = r.groupby("mes", group_keys=False).apply(lambda x: x.nlargest(k, "p"), include_groups=False)
    print(f"  {k}/mes (total {len(g)}): TP={int(g.y.sum()):>3}  precision={g.y.mean():.4f}  lift={g.y.mean()/prev:.2f}x")
    for mes, sub in r.groupby("mes"):
        t = sub.nlargest(k, "p")
        print(f"      {mes}: TP={int(t.y.sum())}/{k}  precision={t.y.mean():.4f}")

print("\n=== F. CALIBRACION DEL SCORE (hallazgo 6) ===")
q = pd.qcut(p_act, 5, labels=False, duplicates="drop")
cal = pd.DataFrame({"q": q, "p": p_act, "y": yte.to_numpy()}).groupby("q").agg(
    score_medio=("p", "mean"), tasa_real=("y", "mean"), n=("y", "size"))
print(cal.to_string())
print(f"\nscore medio global={p_act.mean():.4f}  tasa real global={yte.mean():.4f}")
FE = ss.FEATURE_COLS[:]

d = df.sort_values(["site_id", "mes"]).copy()
lag = d.groupby("site_id")[FE].shift(1); lag.columns = [c + "_lag1" for c in FE]
d = pd.concat([d[["site_id", "mes", ss.TARGET_COL]], lag], axis=1)
d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=list(lag.columns))
te = d[d.mes.astype(str) >= "2026-02"]
y = te[ss.TARGET_COL]; prev = y.mean()
for c in ["USERS_4G_lag1", "avg_AVA_4G_30d_lag1"]:
    ap = average_precision_score(y, te[c] if c.startswith("USERS") else -te[c])
    print(f"{c:<28} (solo, t-1)  AUC-PR={ap:.4f}  lift={ap/prev:.2f}x")
print(f"referencia: modelo con TODAS las features t-1 = 1.35x   |  prevalencia={prev:.4%}\n")

print("=== Q3. Tasa de reclamo por umbral, estratificada por volumen de usuarios ===")
m = df.replace([np.inf,-np.inf], np.nan).dropna(subset=FE).copy()
m["bajo_umbral_AVA"] = m.avg_AVA_4G_30d < ss.AVA_THRESHOLD
m["tercil_users"] = pd.qcut(m.USERS_4G, 3, labels=["pocos", "medios", "muchos"])
t = m.groupby(["tercil_users", "bajo_umbral_AVA"], observed=True)[ss.TARGET_COL].agg(["mean", "size"])
print(t.to_string())
print()
glob = m.groupby("bajo_umbral_AVA")[ss.TARGET_COL].mean()
print(f"Sin estratificar: bajo umbral={glob[True]:.4f}  sobre umbral={glob[False]:.4f}  ratio={glob[True]/glob[False]:.2f}x")
for terc in ["pocos", "medios", "muchos"]:
    s = m[m.tercil_users == terc].groupby("bajo_umbral_AVA")[ss.TARGET_COL].mean()
    if len(s) == 2 and s[False] > 0:
        print(f"  tercil {terc:<7}: ratio dentro del estrato = {s[True]/s[False]:.2f}x")
print()
print("Tasa de reclamo por tercil de usuarios (sin mirar umbral):")
print(m.groupby("tercil_users", observed=True)[ss.TARGET_COL].agg(["mean","size"]).to_string())
