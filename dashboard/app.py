"""
dashboard/app.py — TFM Grupo 1

Dashboard gerencial. Lee directamente el
Parquet que genera `src/score_sites.py` (mismo dato que el ranking en CSV,
no se recalcula nada acá) y lo muestra como tablero interactivo en vez de
gráficos estáticos.

Uso:
    uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
RANKING_PATH = BASE_DIR / "output" / "mensual-pronopro" / "ranking_sitios.parquet"
FIGURES_DIR = BASE_DIR / "output" / "mensual-pronopro" / "figures"

# El gráfico de barras se limita a este máximo aunque la tabla muestre más
# filas -- pasado este número las barras se vuelven ilegibles en pantalla.
MAX_SITIOS_EN_GRAFICO = 30

st.set_page_config(
    page_title="Diagnóstico de red — priorización de sitios",
    layout="wide",
)


@st.cache_data
def load_ranking(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # site_id llega como dtype "string" con contenido numérico (ej. "1485").
    # Plotly, si detecta que todas las etiquetas parecen números, arma un eje
    # continuo en vez de categórico y las barras se vuelven líneas finas sin
    # etiqueta. Se fuerza un label de texto explícito para evitar eso.
    df["site_label"] = "Sitio " + df["site_id"].astype(str)
    return df


def main() -> None:
    st.title("Diagnóstico de calidad de red: priorización de sitios")
    st.caption(
        "Demo del prototipo (TFM Grupo 1)."
    )

    if not RANKING_PATH.exists():
        st.error(
            f"No se encontró {RANKING_PATH}. "
            "Correr antes `python src/build_modeling_dataset.py --grain monthly --source files` "
            "y luego `python src/score_sites.py`."
        )
        return

    df = load_ranking(RANKING_PATH)

    # --- Filtros (sidebar) ---
    st.sidebar.header("Filtros")
    prioridades = st.sidebar.multiselect(
        "Prioridad", options=["Alta", "Media", "Baja"], default=["Alta"]
    )
    perfiles = sorted(df["perfil"].dropna().unique().tolist())
    perfiles_sel = st.sidebar.multiselect("Perfil de sitio", options=perfiles, default=perfiles)
    site_search = st.sidebar.text_input("Buscar site_id")

    df_filtrado = df.copy()
    if prioridades:
        df_filtrado = df_filtrado[df_filtrado["prioridad_sitio"].isin(prioridades)]
    if perfiles_sel:
        df_filtrado = df_filtrado[df_filtrado["perfil"].isin(perfiles_sel)]
    if site_search:
        df_filtrado = df_filtrado[df_filtrado["site_id"].str.contains(site_search, case=False, na=False)]
    df_filtrado = df_filtrado.sort_values("score_riesgo", ascending=False).reset_index(drop=True)

    # --- KPI tiles ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sitios evaluados (total)", f"{len(df):,}")
    col2.metric("Sitios prioridad Alta", f"{(df['prioridad_sitio'] == 'Alta').sum():,}")
    col3.metric("Score promedio (filtro actual)", f"{df_filtrado['score_riesgo'].mean():.3f}" if len(df_filtrado) else "N/D")
    col4.metric("Sitios en vista actual", f"{len(df_filtrado):,}")

    st.divider()

    # --- Tabla: top de sitios con sus recomendaciones (primero, arriba) ---
    st.subheader("Top de sitios en riesgo")
    top_n = st.slider(
        "Cantidad de sitios a mostrar", min_value=10, max_value=min(500, max(10, len(df_filtrado))),
        value=min(50, max(10, len(df_filtrado))), step=10,
    )
    df_top = df_filtrado.head(top_n)

    st.dataframe(
        df_top[
            ["site_id", "score_riesgo", "prioridad_sitio", "accion_recomendada",
             "kpis_driver", "umbral_incumplido", "perfil", "meses_en_riesgo"]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "site_id": "Sitio",
            "score_riesgo": st.column_config.ProgressColumn(
                "Score de riesgo", min_value=0.0, max_value=float(df["score_riesgo"].max()), format="%.3f",
            ),
            "prioridad_sitio": "Prioridad",
            "accion_recomendada": "Acción recomendada",
            "kpis_driver": "Indicadores driver",
            "umbral_incumplido": "Umbral incumplido",
            "perfil": "Perfil",
            "meses_en_riesgo": "Meses en riesgo (histórico)",
        },
    )

    st.divider()

    # --- Gráfico de barras: score de riesgo de esos mismos sitios (debajo) ---
    df_grafico = df_top.head(MAX_SITIOS_EN_GRAFICO)
    st.subheader(f"Score de riesgo (top {len(df_grafico)} de la tabla de arriba)")
    if len(df_top) > MAX_SITIOS_EN_GRAFICO:
        st.caption(
            f"La tabla muestra {len(df_top)} sitios, el gráfico se limita a los primeros "
            f"{MAX_SITIOS_EN_GRAFICO} para que las barras se puedan leer."
        )

    fig_bar = px.bar(
        df_grafico.sort_values("score_riesgo", ascending=True),
        x="score_riesgo",
        y="site_label",
        color="accion_recomendada",
        orientation="h",
        height=max(400, 26 * len(df_grafico)),
        labels={"score_riesgo": "Score de riesgo", "site_label": "Sitio", "accion_recomendada": "Acción recomendada"},
    )
    fig_bar.update_yaxes(type="category", categoryorder="total ascending")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # --- Distribución por perfil (dona) ---
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Distribución por perfil (vista actual)")
        fig_pie = px.pie(df_filtrado, names="perfil", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with g2:
        st.subheader("Distribución por acción recomendada (vista actual)")
        fig_pie_accion = px.pie(df_filtrado, names="accion_recomendada", hole=0.4)
        st.plotly_chart(fig_pie_accion, use_container_width=True)

    st.divider()

    # --- Gráficos de apoyo (SHAP / clustering, generados por el notebook) ---
    with st.expander("Gráficos de apoyo (explicabilidad del modelo)"):
        st.caption("Generados por el notebook de análisis, no se recalculan acá.")
        cols = st.columns(3)
        figuras = ["shap_global_bar.png", "shap_beeswarm.png", "cluster_sitios.png"]
        for col, nombre in zip(cols, figuras):
            fp = FIGURES_DIR / nombre
            if fp.exists():
                col.image(str(fp), caption=nombre, use_container_width=True)
            else:
                col.info(f"{nombre} no encontrado en {FIGURES_DIR}")


if __name__ == "__main__":
    main()
