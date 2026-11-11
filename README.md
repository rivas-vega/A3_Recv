# A3 RecVis — ResNet-18 en Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/main_notebook.ipynb)

Comparación de dos CNN para la competición [MVA RecVis 2026](https://www.kaggle.com/competitions/mva-recvis-2026):

| Modelo | Entrenamiento | Submission |
|---|---|---|
| ResNet-18 baseline | Crop + flip y entropía cruzada | `submission_baseline.csv` |
| ResNet-18 AugMix | Mismos crop + flip, dos vistas AugMix y consistencia JSD (peso 12) | `submission_augmix.csv` |

Ambos parten del mismo checkpoint ImageNet (`ResNet18_Weights.IMAGENET1K_V1`) y ajustan todas sus capas. Las imágenes se amplían de 32 × 32 a **128 × 128 con bilinear** en train, validación y test. Los aumentos aleatorios se aplican solo durante entrenamiento, antes del escalado.

## Ejecutar

1. Abre `main_notebook.ipynb` en Colab y selecciona una GPU A100.
2. Ejecuta las celdas en orden. El notebook es independiente de los scripts `.py` y no necesita un clon local.
3. Autoriza el montaje de Drive para conservar checkpoints. Por defecto, se guardan en `MyDrive/A3_Recv/runs/resnet18_augmix_128_v1`.
4. Autentica Kaggle con `KAGGLE_API_TOKEN` en Secretos de Colab, o con el login interactivo. Obtén el token en [Kaggle Settings → API](https://www.kaggle.com/settings/api). El login del navegador no se comparte con Python. Debes haber aceptado las reglas de la competición.
5. KaggleHub descarga los cuatro archivos privados: `train.npz`, `validation.npz`, `test.npz` y `sample_submission.csv`.
6. Entrena baseline y AugMix. El primer CSV se guarda antes de empezar el segundo modelo.
7. Descarga el ZIP de resultados, descomprímelo y sube **cada `submission_*.csv` por separado** a Kaggle. Copia las puntuaciones en `KAGGLE_SCORES` para completar la comparación.

La celda opcional de actualización enlaza a la última revisión de GitHub. Una copia anterior guardada en Drive no incorpora cambios del repositorio automáticamente.

## Entrenamiento y reanudación

- Hasta 30 épocas por modelo, con parada temprana tras 8 épocas sin mejora, a partir de la época 15.
- AdamW, LR del backbone `3e-4` y de la cabeza `1e-3`, calentamiento de 2 épocas y descenso coseno. Batch de 128 imágenes originales; AugMix procesa 384 vistas.
- Presupuesto aproximado de 2,5 horas por modelo **y por ejecución**, comprobado entre épocas. Preparación e inferencia quedan fuera. La duración real depende del runtime de Colab y se estima tras la primera época.
- `last.pt` guarda el modelo actual, mejor modelo, optimizador, estado AMP, historial y RNG al acabar cada época. `best.pt` contiene el modelo seleccionado para inferencia.
- Al reanudar se repite una época interrumpida y se reconstruye el LR por número de época. Los experimentos completados no se vuelven a entrenar; los pausados por tiempo continúan al volver a ejecutar su celda.
- El mejor checkpoint se selecciona exclusivamente por precisión de validación limpia, con desempate por pérdida. Los dos experimentos pueden terminar en épocas distintas; la tabla indica cuáles se ejecutaron y cuál se eligió.
- Cambia `RUN_NAME` para iniciar un experimento independiente o modificar hiperparámetros. El notebook detecta configuraciones y archivos de datos incompatibles con checkpoints previos.

Las curvas, `history.csv`, `comparison.csv` y la tasa de desacuerdo permiten inspeccionar los resultados. Train y validación son CIFAR-10; el test es sintético. La precisión de validación y el desacuerdo entre modelos **no son la precisión del test**: esta solo se obtiene en Kaggle. No hay resultados de entrenamiento incluidos en el repositorio.

## Código y referencias

- `main_notebook.ipynb`: comparación completa y autocontenida para Colab; sustituye las exploraciones de histogramas, OT y escalados.
- `main.py`, `model.py`, `data.py`, `requirements.txt`: baseline original de línea de comandos; no implementan el nuevo experimento del notebook.
- [Starter RecVis](https://github.com/fcendra-enpc/recvis26_a3), del que parte este repositorio; se conserva su licencia.
- [AugMix original](https://github.com/google-research/augmix) y [transformación de torchvision](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.AugMix.html). Se usa la variante `all_ops=True` con severidad 3 y tres ramas; la pérdida compara una vista básica y dos aumentadas.

## Datos y entrega

Este repositorio es público. No subas datos privados, tokens, pesos ni salidas de notebooks que contengan imágenes del dataset. `.gitignore` no borra salidas incrustadas en un `.ipynb`: límpialas antes de hacer commit.

El informe debe declarar el preentrenamiento ImageNet, las transformaciones, los hiperparámetros y los resultados medidos. La entrega del curso es un PDF CVPR de una página (referencias aparte) y un ZIP de **código** de menos de 8 MB, sin datos ni pesos. El ZIP de resultados que descarga el notebook es distinto del ZIP de código.
