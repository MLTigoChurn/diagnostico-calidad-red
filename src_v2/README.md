# Pipeline corregido (v2)

Este directorio **no es el pipeline de la entrega**. El entregable es `src/`, con salida en `output/`, y es el que produjo todas las cifras de la memoria y de la presentación.

`src_v2/` existe para responder a la pregunta, planteada en la revisión del entregable final, de si las variables usan información del propio periodo. En vez de argumentar si el sesgo era grande o pequeño, se reconstruyó el pipeline sin los defectos y se midió la diferencia.

## Qué corrige

| # | Defecto en `src/build_modeling_dataset.py` | Corrección |
|---|---|---|
| 1 | `degradacion_THP_30d` usa `THROUGHPUT_4G` del día en curso como numerador, sin desplazar (línea 433) | Cociente entre una ventana de 7 días y una de 30, ambas cerradas antes del mes |
| 2 | `USERS_4G` entra como media del propio mes, sin `shift` ni `rolling` (línea 443) | Pasa por la misma ventana previa que el resto de las variables |
| 3 | Las columnas rolling se promedian *dentro* del mes, así que la ventana del día 20 contiene del 1 al 19 del mismo mes | Se toma el valor del último día disponible anterior al mes, vía `merge_asof` |
| 4 | El `.rolling()` se aplica sobre el resultado de `groupby().shift(1)`, que ya es una Serie sin agrupar, así que la ventana cruza de un sitio al siguiente (líneas 268, 352 y 429) | `rolling` agrupado por sitio |

`thp_per_user` no se corrige acá: `score_sites.py` lo deriva de `avg_THP_4G_30d / USERS_4G`, y al quedar limpios sus dos insumos queda limpio.

## Resultado

| Pipeline | Entrenamiento | AUC-PR | Lift |
|---|---|---|---|
| v1, lo entregado | oct a ene, 8.931 filas | 0,0424 | 1,61x |
| v1 sin octubre, variables contaminadas | nov a ene, 6.769 filas | 0,0325 | 1,23x |
| v2, variables limpias | nov a ene, 6.573 filas | 0,0322 | 1,23x |

Con el mismo periodo de entrenamiento, contaminado y limpio dan lo mismo. Los defectos no estaban inflando las métricas. Lo que sostiene el 1,61x publicado es un mes más de entrenamiento que un pipeline correcto no puede usar, porque octubre no tiene mes previo del cual sacar la ventana.

Lo que sí cambia es la lista concreta de sitios: el top 30 de v1 y el de v2 comparten 5 sitios.

## Cómo correrlo

```bash
uv run python src_v2/build_modeling_dataset_v2.py
uv run python src/score_sites.py \
    --data output_v2/modeling_dataset_monthly_v2.parquet \
    --output-dir output_v2/mensual-pronopro
uv run python src_v2/comparar_v1_v2.py
```

Las salidas van a `output_v2/`, que no se versiona por la misma razón que `output/`: son datos derivados de datos reales del cliente.

## Por qué no se rehízo la entrega con esto

Los defectos se encontraron cuatro días antes de la defensa. Rehacer el pipeline habría cambiado el ranking, las figuras SHAP, los perfiles del clustering y los números financieros de toda la memoria. Se optó por documentar, medir y reportar la cifra corregida, dejando la corrección escrita y ejecutable.
