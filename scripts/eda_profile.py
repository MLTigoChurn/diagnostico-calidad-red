"""EDA profiling pass — TFM Grupo 1. Throwaway exploration to source real findings."""
import pandas as pd
import numpy as np

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
DB = "db_files"

def line(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

# ---------- CLAIMS ----------
line("CLAIMS")
claims = pd.read_csv(f"{DB}/claims.csv", sep="|", encoding="utf-8-sig", dtype=str)
print("shape:", claims.shape)
print("cols:", list(claims.columns))
print("\nnulls:\n", claims.isna().sum())
print("\ndup nro_ticket:", claims["nro_ticket"].duplicated().sum())
claims["fecha_creacion"] = pd.to_datetime(claims["fecha_creacion"], errors="coerce")
print("\nfecha range:", claims["fecha_creacion"].min(), "->", claims["fecha_creacion"].max())
print("\nmotivo_atencion:\n", claims["motivo_atencion"].value_counts(dropna=False))
print("\ncategory_1 (top15):\n", claims["category_1"].value_counts(dropna=False).head(15))
print("\nsegmento:\n", claims["segmento"].value_counts(dropna=False).head(10))
# cross motivo x category_1
print("\nmotivo x category_1 (is RED a network claim?):\n",
      pd.crosstab(claims["motivo_atencion"], claims["category_1"]).iloc[:, :8])
# unique users complaining
print("\nunique users in claims:", claims["user_id"].nunique())
print("claims per user (describe):\n", claims["user_id"].value_counts().describe())

# ---------- LOCATION ----------
line("LOCATION")
loc = pd.read_csv(f"{DB}/location.csv", sep=",", dtype={"user_id": str, "site_id": str})
print("shape:", loc.shape)
print("cols:", list(loc.columns))
print("\nnulls:\n", loc.isna().sum())
print("\nunique users:", loc["user_id"].nunique(), "| unique sites:", loc["site_id"].nunique())
# user -> 1 site? (client says yes for whole period)
ups = loc.groupby("user_id")["site_id"].nunique()
print("users with >1 site:", (ups > 1).sum(), "/", len(ups))
# rows per user (user x month?)
print("rows per user (describe):\n", loc["user_id"].value_counts().describe())
print("\ntipo_negocio:\n", loc["tipo_negocio"].value_counts(dropna=False).head())
print("\ntecnologia_soporta:\n", loc["tecnologia_soporta"].value_counts(dropna=False).head())
for c in ["trafico_4g", "trafico_3g", "trafico_2g"]:
    v = pd.to_numeric(loc[c], errors="coerce")
    print(f"{c}: mean={v.mean():.2f} median={v.median():.2f} zeros={ (v==0).mean()*100:.1f}% nulls={v.isna().mean()*100:.1f}%")

# ---------- KPIS ----------
line("KPIS_DIARIO")
kpi = pd.read_csv(f"{DB}/kpis_diario.txt", sep="\t", dtype={"site_id": str})
print("shape:", kpi.shape)
print("cols:", list(kpi.columns))
kpi["FECHA"] = pd.to_datetime(kpi["FECHA"], errors="coerce")
print("FECHA range:", kpi["FECHA"].min(), "->", kpi["FECHA"].max())
print("unique sites:", kpi["site_id"].nunique())
print("days covered:", kpi["FECHA"].nunique())
print("\nnulls per col:\n", kpi.isna().sum())
for c in ["AVA_4G", "CSFR_V4G", "DC_V4G", "THROUGHPUT_4G", "USERS_4G", "AVA_3G"]:
    if c in kpi.columns:
        v = pd.to_numeric(kpi[c], errors="coerce")
        print(f"{c}: min={v.min():.3f} p25={v.quantile(.25):.3f} med={v.median():.3f} p75={v.quantile(.75):.3f} max={v.max():.3f} nulls={v.isna().mean()*100:.1f}%")

# ---------- JOIN COVERAGE ----------
line("JOIN COVERAGE")
claim_users = set(claims["user_id"].dropna())
loc_users = set(loc["user_id"].dropna())
loc_sites = set(loc["site_id"].dropna())
kpi_sites = set(kpi["site_id"].dropna())
print(f"claims.user_id in location: {len(claim_users & loc_users)}/{len(claim_users)} "
      f"= {100*len(claim_users & loc_users)/len(claim_users):.1f}%")
print(f"location.site_id in kpis  : {len(loc_sites & kpi_sites)}/{len(loc_sites)} "
      f"= {100*len(loc_sites & kpi_sites)/len(loc_sites):.1f}%")
print(f"kpi sites not in location : {len(kpi_sites - loc_sites)}")

# ---------- TARGET / CLASS BALANCE at site x day ----------
line("TARGET site x day")
user_site = loc[["user_id", "site_id"]].drop_duplicates("user_id")
cl = claims.merge(user_site, on="user_id", how="inner")
cl["dia"] = cl["fecha_creacion"].dt.normalize()
cl["es_reclamo"] = (cl["motivo_atencion"].str.upper().str.strip() == "RECLAMO").astype(int)
cl["es_reclamo_red"] = ((cl["motivo_atencion"].str.upper().str.strip() == "RECLAMO") &
                        (cl["category_1"].str.upper().str.strip() == "RED")).astype(int)
agg = cl.groupby(["site_id", "dia"]).agg(
    n_reclamos=("es_reclamo", "sum"),
    n_reclamos_red=("es_reclamo_red", "sum"),
    n_claims=("nro_ticket", "count"),
).reset_index()
print("claims matched to a site:", len(cl), "/", len(claims))
print("site x day rows with >=1 claim:", len(agg))
print("of those with >=1 RECLAMO:", (agg["n_reclamos"] > 0).sum())
print("of those with >=1 RECLAMO+RED:", (agg["n_reclamos_red"] > 0).sum())

# Full panel class balance: all site x day in kpi range that have a claim vs not
kpi["dia"] = kpi["FECHA"].dt.normalize()
panel = kpi[["site_id", "dia"]].drop_duplicates()
panel = panel.merge(agg, on=["site_id", "dia"], how="left")
panel["es_reclamo_red"] = (panel["n_reclamos_red"].fillna(0) > 0).astype(int)
print("\nFull panel (site x day with KPI):", len(panel))
print("positive (es_reclamo_red=1):", panel["es_reclamo_red"].sum(),
      f"= {100*panel['es_reclamo_red'].mean():.3f}% (CLASS IMBALANCE)")

# ---------- KPI vs COMPLAINT correlation (the key business question) ----------
line("KPI vs COMPLAINT (key finding)")
m = panel.merge(kpi, on=["site_id", "dia"], how="left")
for c in ["AVA_4G", "CSFR_V4G", "DC_V4G", "THROUGHPUT_4G", "USERS_4G"]:
    if c in m.columns:
        v = pd.to_numeric(m[c], errors="coerce")
        pos = v[m["es_reclamo_red"] == 1].median()
        neg = v[m["es_reclamo_red"] == 0].median()
        print(f"{c}: median(complaint)={pos:.3f} vs median(no-complaint)={neg:.3f}")
