# Proyecto final - Auto MPG - Etapa de transferencia

**Estudiante:** José Felipe Martínez Zafra  
**Institución:** UCompensar  
**Fecha:** septiembre de 2026

## Propósito

Este repositorio contiene el dashboard interactivo y la documentación del proyecto final. La problemática estudia la relación entre el peso de un vehículo y su eficiencia de combustible (MPG), así como el aporte de otras variables técnicas mediante contraste de hipótesis, regresión lineal, Ridge y regresión logística.

## Resultados de referencia

- 398 observaciones en Auto MPG.
- Pearson `weight`-`mpg`: `r = -0.831741`, `p unilateral = 1.4864e-103`.
- Regresión múltiple: `R² prueba = 0.7138`, `RMSE = 4.0527`, `MAE = 3.2892`.
- Regresión simple con `weight`: `R² prueba = 0.7243`, `RMSE = 3.9777`, `MAE = 3.2240`.
- Ridge con `alpha = 1`: `R² prueba ≈ 0.7137`, `RMSE ≈ 4.0537`.
- Regresión logística final: `accuracy = 86.67%`, `sensibilidad = 86.21%`, `especificidad = 87.10%`, `AUC = 0.9349`.

## Estructura

```text
ENTREGA_FINAL_AUTO_MPG/
├── app.py
├── model_utils.py
├── assets/
│   └── style.css
├── data/
│   ├── README.txt
│   ├── auto-mpg.data
│   └── LICENSE-ATTRIBUTION.md
├── figuras/
├── tests/
│   ├── smoke_test.py
│   └── integration_test.py
├── Dashboard_Auto_MPG.ipynb
├── requirements.txt
├── run_local.sh
├── run_local.bat
├── LINKS.txt
├── CONSIGNA_PROYECTO_FINAL.md
├── INFORME_FINAL_JFMZ_COMPLETO.docx
└── INFORME_FINAL_JFMZ_COMPLETO.pdf
```

## Ejecución local

### Windows

1. Abrir una terminal en esta carpeta.
2. Ejecutar `run_local.bat`.
3. Abrir `http://127.0.0.1:8050`.

### macOS / Linux

```bash
chmod +x run_local.sh
./run_local.sh
```

### Ejecución manual

```bash
python -m pip install -r requirements.txt
python app.py
```

## Datos

El archivo original `data/auto-mpg.data` se incluye en el proyecto. La aplicación puede iniciar sin descargar los datos; al primer arranque crea una caché local `data/auto_mpg.csv`. El origen y las condiciones de atribución están documentados en `data/LICENSE-ATTRIBUTION.md`.

## Binder

El notebook `Dashboard_Auto_MPG.ipynb` importa la misma aplicación y utiliza el modo Jupyter integrado de Dash. El repositorio previsto es `Martinezzafrajose21/JFMZ1`; tras subir todos los archivos, abre el enlace de `LINKS.txt` y verifica la ejecución en Binder antes de entregar.

## Controles del dashboard

- Filtros por año, origen y cilindros.
- Comparación entre regresión simple, múltiple y Ridge.
- Ajuste interactivo de `alpha` para Ridge.
- Ajuste de `C`, `solver` y umbral para regresión logística.
- Matriz de confusión, curva ROC, probabilidades, correlaciones y tabla filtrable.

## Reproducibilidad

Los modelos usan `random_state = 42`. La regresión utiliza una partición 70/30 y la clasificación una partición estratificada 70/30. `StandardScaler` está dentro de `Pipeline` para evitar fuga de información.

Para comprobar los datos, el arranque HTTP, las pestañas y los controles principales desde la carpeta raíz, ejecuta `python tests/integration_test.py`. La comparación interactiva de hiperparámetros se presenta como exploración de una sola partición; para estimar el rendimiento tras seleccionarlos se necesitaría validación cruzada sobre entrenamiento y otra evaluación final.
