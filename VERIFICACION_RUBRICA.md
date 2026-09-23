# Verificación de la entrega frente a la rúbrica

## Entregables obligatorios

- **Informe del proceso de desarrollo y justificación de decisiones:** `INFORME_FINAL_JFMZ_COMPLETO.pdf` y `INFORME_FINAL_JFMZ_COMPLETO.docx`.
- **Carpeta con los archivos requeridos para visualizar el dashboard:** `app.py`, `model_utils.py`, `assets/style.css`, `requirements.txt`, `Dashboard_Auto_MPG.ipynb`, scripts de inicio y documentación.
- **Archivo con los dos enlaces solicitados:** `LINKS.txt`. Apunta al repositorio `Martinezzafrajose21/JFMZ1` y al notebook en Binder.

## Diseño visual

Evidencia principal: `assets/style.css` y la estructura visual definida en `app.py`. El dashboard utiliza una composición responsive, tarjetas KPI, pestañas, panel de filtros, jerarquía tipográfica, espacios consistentes y una paleta coherente.

## Claridad de la información

Evidencia principal: pestaña **Resumen**, tarjetas de métricas, textos narrativos y sección de trazabilidad. El dashboard relaciona las visualizaciones con la pregunta de investigación y conserva las métricas del informe.

## Interactividad

Evidencia principal: callbacks de `app.py`. Incluye filtros por año, origen y cilindros; selección de modelos de regresión; ajuste de `alpha` para Ridge; ajuste de `C`, `solver` y umbral para regresión logística; tabla filtrable y visualizaciones que se actualizan con las decisiones del usuario.

## Relevancia del contenido

El contenido se mantiene centrado en la problemática definida durante el curso: relación entre `weight` y `mpg` y evaluación del aporte de otras características técnicas a la predicción y clasificación de la eficiencia de combustible.

## Originalidad y aportación al problema

El producto no se limita a mostrar gráficos estáticos. Permite contrastar modelos, observar residuos, explorar hiperparámetros, modificar el umbral de clasificación, examinar la matriz de confusión y la curva ROC, y vincular cada resultado con una interpretación del problema.

## Reproducibilidad y control de calidad

- `random_state = 42`.
- Dependencias declaradas en `requirements.txt`.
- Scripts de ejecución para Windows y macOS/Linux.
- Notebook preparado para Binder/Jupyter.
- Prueba estructural en `tests/smoke_test.py`.
- Datos originales de UCI incluidos en `data/auto-mpg.data`, con atribución y licencia.
- Prueba `tests/integration_test.py` de carga, HTTP, pestañas, filtros y resultados de los modelos.

## Publicación

El proyecto está publicado en el repositorio GitHub del estudiante. `LINKS.txt` contiene los dos enlaces para la entrega.
