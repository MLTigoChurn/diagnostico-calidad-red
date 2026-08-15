# Diagnóstico de calidad de red

Pipeline en Python que procesa reclamos e indicadores de red de Tigo, entrena un modelo de riesgo por sitio y genera un ranking accionable de sitios, con la acción recomendada para cada uno (mantenimiento programado o inversión en capacidad).

## Manual de usuario

El manual completo (`manual/manual.pdf`) explica, en lenguaje no técnico, cómo usar el sistema: prerrequisitos, anonimización de los datos crudos, generación del ranking mensual, cómo leer el resultado y los gráficos de apoyo, e incidencias habituales. Incluye capturas reales de una corrida del sistema y un glosario de términos.

En resumen, el flujo de uso tiene tres pasos:

1. Anonimizar los datos crudos del cliente: `python scripts/cleanup-data.py`
2. Construir el dataset del mes: `python src/build_modeling_dataset.py --grain monthly --source files`
3. Generar el ranking de sitios en riesgo: `python src/score_sites.py`

El resultado queda en `output/mensual-pronopro/ranking_sitios.csv`, un registro por sitio con su score de riesgo, prioridad, acción recomendada y el indicador de red que la sustenta.

## Estructura del repositorio

- `src/`: código del pipeline (construcción de datos, scoring)
- `dashboard/`: tablero interactivo (Streamlit), lee el ranking generado
- `scripts/`: utilidades, incluye la anonimización de datos
- `notebooks/`: notebooks de análisis y exploración
- `manual/`: manual de usuario (PDF) y capturas de referencia

## Requisitos

Python 3.11 o superior y `uv`. Instalar dependencias con `uv sync`.
