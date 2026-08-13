"""EDA v2 — correct DD/MM/YYYY parsing + threshold analysis (key business question)."""
import pandas as pd
import numpy as np

DB = "db_files"
def line(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

claims = pd.read_csv(f"{DB}/claims.csv", sep="|", encoding="utf-8-sig", dtype=str)
claims["fecha_creacion"] = pd.to_datetime(claims["fecha_creacion"], errors="coerce")
loc = pd.read_csv(f"{DB}/location.csv", sep=",", dtype={"user_id": str, "site_id": str})
kpi = pd.read_csv(f"{DB}/kpis_diario.txt", sep="\t", dtype={"site_id": str})
kpi["FECHA"] = pd.to_datetime(kpi["FECHA"], format="%d/%m/%Y", errors="coerce")  # FIX

line("KPI date sanity after fix")
print("FECHA range:", kpi["FECHA"].min(), "->", kpi["FECHA"].max())
print("FECHA nulls:", kpi["FECHA"].isna().sum())
print("days covered:", kpi["FECHA"].nunique(), "| sites:", kpi["site_id"].nunique())
print("rows per site (describe):\n", kpi.groupby("site_id").size().describe())

line("517-claim user investigation")
top = claims["user_id"].value_counts().head(3)
print(top)
tu = top.index[0]
sub = claims[claims["user_id"] == tu]
print("segmento:", sub["segmento"].unique(), "| motivo:", sub["motivo_atencion"].value_counts().to_dict())
print("date span:", sub["fecha_creacion"].min(), "->", sub["fecha_creacion"].max())

line("TARGET panel (correct dates)")
user_site = loc[["user_id", "site_id"]].drop_duplicates("user_id")
cl = claims.merge(user_site, on="user_id", how="inner")
cl["dia"] = cl["fecha_creacion"].dt.normalize()
cl["es_reclamo_red"] = (cl["motivo_atencion"].str.upper().str.strip() == "RECLAMO").astype(int)
agg = cl.groupby(["site_id", "dia"]).agg(n_reclamos_red=("es_reclamo_red", "sum")).reset_index()
kpi["dia"] = kpi["FECHA"].dt.normalize()
panel = kpi.dropna(subset=["dia"])[["site_id", "dia"]].drop_duplicates()
panel = panel.merge(agg, on=["site_id", "dia"], how="left")
panel["y"] = (panel["n_reclamos_red"].fillna(0) > 0).astype(int)
print("panel rows:", len(panel), "| positives:", panel["y"].sum(),
      f"= {100*panel['y'].mean():.3f}%")
m = panel.merge(kpi, on=["site_id", "dia"], how="left")
for c in ["AVA_4G", "CSFR_V4G", "DC_V4G", "CSFR_D4G", "DC_D4G", "THROUGHPUT_4G", "USERS_4G"]:
    v = pd.to_numeric(m[c], errors="coerce")
    pos, neg = v[m.y == 1], v[m.y == 0]
    print(f"{c:14s} complaint p50={pos.median():.4f} p90={pos.quantile(.9):.4f} | "
          f"no-complaint p50={neg.median():.4f} p90={neg.quantile(.9):.4f}")

line("THRESHOLD ANALYSIS — complaint rate by KPI bucket (key business Q)")
for c, edges in [("AVA_4G", [0, .90, .95, .98, .99, .995, 1.01]),
                 ("DC_V4G", [-1, .001, .005, .01, .02, .05, 1.01]),
                 ("CSFR_V4G", [-1, .001, .005, .01, .02, .05, 1.01]),
                 ("THROUGHPUT_4G", [0, 5, 10, 15, 20, 30, 1000])]:
    v = pd.to_numeric(m[c], errors="coerce")
    b = pd.cut(v, edges)
    t = m.assign(b=b).groupby("b", observed=True)["y"].agg(["mean", "sum", "count"])
    t["rate_%"] = (t["mean"] * 100).round(3)
    print(f"\n{c}:\n", t[["rate_%", "sum", "count"]])
