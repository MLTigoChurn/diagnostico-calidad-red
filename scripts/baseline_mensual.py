"""Linea base (reglas + regresion logistica) sobre el MISMO split que el modelo
final: mensual, target confirmado (PRONOPRO), features con thp_per_user.
Objetivo: dar un numero trazable para la slide 13, que hoy dice 1.7x sin fuente.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

BASE = Path("/Users/guillermoventura/Documents/Maestria/TFM/TFM-grupo1")
sys.path.insert(0, str(BASE / "src"))
import score_sites as ss

DATA = BASE / "output" / "modeling_dataset_monthly_pronopro.parquet"
CUTOFF = "2026-02-01"

df = ss.load_dataset(DATA)
_, df_tr, df_te, Xtr, ytr, Xte, yte = ss.build_features(df, CUTOFF)
prev = yte.mean()
print(f"test: {len(yte)} filas, {int(yte.sum())} positivos, prevalencia {prev:.4%}")

def report(name, score, pred):
    ap = average_precision_score(yte, score)
    print(f"{name:34s} AUC-PR={ap:.4f}  lift={ap/prev:.2f}x  "
          f"P={precision_score(yte,pred,zero_division=0):.4f} "
          f"R={recall_score(yte,pred,zero_division=0):.3f} "
          f"F1={f1_score(yte,pred,zero_division=0):.4f}")
    return ap

# 1. Reglas de negocio E2: score = numero de umbrales incumplidos (0-3)
def rules(X):
    return ((X["avg_AVA_4G_30d"] < ss.AVA_THRESHOLD).astype(int)
            + (X["max_DC_V4G_30d"] > ss.DC_THRESHOLD).astype(int)
            + (X["avg_THP_4G_30d"] < ss.THP_THRESHOLD_MBP).astype(int))
s_rules = rules(Xte)
ap_rules = report("Reglas (linea base E2)", s_rules, (s_rules >= 1).astype(int))

# 2. Regresion logistica balanceada
lr = make_pipeline(StandardScaler(),
                   LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
lr.fit(Xtr, ytr)
s_lr = lr.predict_proba(Xte)[:, 1]
ap_lr = report("Logistic Regression", s_lr, lr.predict(Xte))

AP_XGB = 0.0424
print(f"\nXGBoost + Optuna (reportado)      AUC-PR={AP_XGB:.4f}  lift={AP_XGB/prev:.2f}x")
print(f"\nXGBoost vs reglas   : {AP_XGB/ap_rules:.2f}x")
print(f"XGBoost vs log. reg.: {AP_XGB/ap_lr:.2f}x")
