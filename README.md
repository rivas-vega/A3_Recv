# A3 RecVis — DINOv3 ConvNeXt en Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/main_notebook.ipynb)

El notebook principal compara dos adaptaciones de **DINOv3 ConvNeXt Base** para [MVA RecVis 2026](https://www.kaggle.com/competitions/mva-recvis-2026):

| Fase | Entrenamiento | CSV |
|---|---|---|
| Linear probe | CNN congelada, nueva cabeza lineal de 10 clases | `submission_dinov3_linear_256.csv` |
| Ajuste parcial | Desde la cabeza anterior; última etapa CNN + LayerNorm final + cabeza, con AugMix suave/JSD | `submission_dinov3_partial_ft_256.csv` |

Checkpoint: [facebook/dinov3-convnext-base-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m). La arquitectura ejecutada es una CNN; los pesos publicados proceden de destilación de un ViT-7B sobre imágenes web LVD-1689M. No ejecutamos ese profesor ni repetimos el preentrenamiento. Se registra una revisión inmutable del checkpoint y se verifica que todos los pesos se carguen.

## Resolución y procesamiento

Entrada **256 × 256**, ampliando las imágenes originales de 32 × 32 con bilinear y antialias. Orden de operaciones: tensor uint8 → resize → float [0,1] → normalización ImageNet, con media (0.485, 0.456, 0.406) y desviación (0.229, 0.224, 0.225). Es el mismo procesamiento para las características, el ajuste y la inferencia.

La elección sigue el [README oficial](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/README.md#image-transform), que recomienda 256 para los pesos web. El paper evalúa ConvNeXt a 256 y 512. La receta pública de ejemplo de destilación ConvNeXt utiliza crops globales de 512 para el profesor y reducción por 2 para el alumno; no debe confundirse esa receta de ejemplo con toda la historia del checkpoint publicado.

No se hereda el tamaño de ResNet ni se aplica un segundo procesador que vuelva a redimensionar a 224. La configuración del procesador Hugging Face se registra como referencia. No se hace recorte central adicional al evaluar estas imágenes cuadradas. La interpolación no recupera información perdida.

## Ejecutar

1. Abre [el checkpoint](https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m) con tu cuenta Hugging Face y completa la solicitud/aceptación de acceso. Debe estar concedido antes de descargar.
2. Crea un [token de tipo Read](https://huggingface.co/settings/tokens), o uno fine-grained con permiso de lectura para ese modelo gated. Pégalo entre las comillas de **`HF_TOKEN = ""`** en la celda dedicada anterior a la carga del checkpoint y ejecuta esa celda. No se solicita en la salida ni se sustituyen tus credenciales por tokens guardados en caché. Hazlo solo en tu copia privada de Colab y borra el token antes de compartirla o guardarla en GitHub. Un error **403 / not in the authorized list** requiere obtener acceso al modelo con la misma cuenta del token; el campo de entrada no concede ese permiso.
3. Abre el notebook, selecciona GPU A100 y ejecuta las celdas en orden. No necesitas clonar ni ejecutar los scripts `.py`.
4. Autoriza Drive para conservar cachés y checkpoints en `MyDrive/A3_Recv/runs/dinov3_convnext_base_256_v1`. La carpeta ResNet anterior queda separada.
5. Autentica Kaggle con `KAGGLE_API_TOKEN` o el login interactivo. Debes haber aceptado las reglas con esa cuenta. KaggleHub descarga train, validation, test y sample_submission.
6. La fase lineal genera su CSV antes del ajuste parcial. Si Colab se interrumpe, conserva ese primer resultado.
7. Descarga `dinov3_two_submissions.zip`, descomprímelo y sube **cada CSV por separado** a Kaggle. Introduce después las puntuaciones en `KAGGLE_SCORES`.

La carga del checkpoint comprueba primero la identidad del token (`whoami`) y muestra el nombre de la cuenta, sin imprimir el token. Después comprueba el permiso efectivo sobre el modelo (`auth_check`). **«Acceso de lectura a DINOv3: OK»** confirma ambas comprobaciones. Si falla la identidad, revisa o sustituye el token; si reconoce la cuenta pero falla el acceso, revisa la aprobación del modelo y los permisos del token. Un 401 durante la descarga por sí solo no distingue ambas causas. Tras cambiar el token, vuelve a ejecutar su celda y la carga del checkpoint.

El notebook fija `transformers==4.57.3`, cuya implementación ConvNeXt se comprobó al preparar el código. Si Colab tenía importada otra versión, reinicia la sesión después de instalar. No instala ni sustituye PyTorch/torchvision.

## Qué se entrena y cómo se reanuda

- **Linear probe:** características congeladas de train original y volteado; hasta 100 épocas solo de la cabeza, AdamW LR 1e-3 y paciencia 10 desde época 10. Val/test usan solo imagen completa. Se cachean las características por conjunto/vista; una extracción incompleta se repite.
- **Ajuste parcial:** hasta 10 épocas, primeras tres etapas CNN congeladas; LR 1e-5 en última etapa y normalización final, 1e-4 en cabeza, una época de warmup y descenso coseno. Crop con padding 2 a 32px, flip, AugMix severidad 2 y JSD de peso 6. Paciencia 3 desde época 4.
- **Presupuesto:** aproximadamente 3 h de ajuste por ejecución, comprobadas entre épocas; preparación, extracción, entrenamiento de cabeza e inferencia son adicionales. La duración se estima con la primera época real.
- El modelo previo al ajuste es un candidato **época 0**. Si ninguna época mejora la validación, se conserva ese modelo. Esto no prueba que la validación elija el mejor resultado sobre el dominio sintético.
- Se guarda al terminar cada época: parámetros modificables, optimizador, AMP, historial y RNG. Al reanudar se repite una época interrumpida. Las capas congeladas se reconstruyen desde la misma revisión original.
- Los checkpoints del ajuste comprueban también la identidad de la cabeza lineal de partida. Un cambio de configuración/datos requiere un `RUN_NAME` nuevo.
- `training_hours` de la tabla mide épocas y excluye extracción de características e inferencia. La extracción muestra su propio tiempo durante la ejecución.
- La comparación lineal vs ajuste parcial incorpora también AugMix: compara protocolos completos de adaptación, no una ablación aislada de descongelación.

## Resultados anteriores

Comunicados por el usuario a partir de su ejecución y los scores de Kaggle; no recalculados por el asistente:

| Experimento | Entrada | Validación | Kaggle público | Épocas / mejor |
|---|---|---:|---:|---|
| ResNet-18 baseline | Bilinear 128 × 128 | 96.04% | 61.093% | 30 / 29 |
| ResNet-18 AugMix + JSD | Bilinear 128 × 128 | 96.56% | 75.796% | 30 / 29 |

Se mantiene el notebook reproducible anterior: [resnet18_augmix.ipynb](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/resnet18_augmix.ipynb). **Todavía no se incluyen resultados reales de DINOv3.**

Train/validation son CIFAR-10 y el test es sintético. Los scores de validación, confianza y desacuerdo no sustituyen la accuracy de Kaggle. Comparar DINOv3 con ResNet cambia arquitectura, preentrenamiento, resolución y adaptación; no demuestra una ventaja atribuible solo a la arquitectura.

## Fuentes y código

- [DINOv3: repositorio oficial](https://github.com/facebookresearch/dinov3) y [paper, sección 7.2](https://arxiv.org/html/2508.10104v1#S7.SS2).
- [Configuración de destilación ConvNeXt de ejemplo](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/configs/train/distillation_convnext/multi_distillation_convnext_test.yaml).
- [Implementación del descriptor global ConvNeXt en Transformers 4.57.3](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/dinov3_convnext/modeling_dinov3_convnext.py).
- [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/abs/2202.10054), motivación para evaluar primero la cabeza con extractor congelado.
- [AugMix](https://github.com/google-research/augmix).
- `main.py`, `model.py`, `data.py` y `requirements.txt` conservan el [starter original](https://github.com/fcendra-enpc/recvis26_a3); no ejecutan el nuevo experimento. Se conserva su licencia.

## Datos y entrega

Repositorio público: no subas tokens, datos privados, características, pesos ni salidas de notebooks que contengan imágenes. `.gitignore` no borra salidas incrustadas en un notebook.

Declara en el informe el checkpoint y revisión, el preentrenamiento/destilación, las transformaciones, capas entrenadas, hiperparámetros y scores medidos. La entrega del curso es PDF CVPR de una página (referencias aparte) y ZIP de código menor de 8 MB, sin datos ni pesos. El ZIP de resultados que descarga el notebook es diferente del ZIP de código.
