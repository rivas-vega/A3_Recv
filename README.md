# A3 RecVis — SigLIP 2 Giant + DINOv3 ViT en Colab

## Nuevo experimento: PE-Core-G y vistas locales

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/pe_semantic.ipynb)

[`pe_semantic.ipynb`](pe_semantic.ipynb) prueba **PE-Core-G14-448**, otro modelo de imagen y texto, congelado y zero-shot. La torre visual tiene aproximadamente 1.88B parámetros y la textual 0.47B. La entrada sigue la transformación oficial: imagen completa a 448×448, bilinear PIL y normalización 0.5/0.5. Se usan las mismas 80 plantillas públicas que en SigLIP. **Todavía no hay una puntuación Kaggle de PE.**

El experimento genera tres candidatos, sin enviar nada automáticamente a Kaggle:

| CSV | Comparación |
|---|---|
| `submission_pe_zero_shot.csv` | PE, imagen completa + flip |
| `submission_pe_local_views.csv` | PE con recortes suaves en las imágenes de menor margen |
| `submission_siglip_pe_zero_shot.csv` | 50/50 entre SigLIP zero-shot anterior y PE completo/flip |

Las vistas locales son cinco crops de 28×28 (cuatro esquinas y centro) sobre el original de 32×32. Se calculan solo en el 25% de imágenes con menor margen entre las dos primeras clases, sin usar etiquetas. Para esas imágenes, se conserva el 75% del logit de imagen completa/flip y se añade el 25% del promedio de crops. El resto no cambia. Los porcentajes son una hipótesis fijada antes de las puntuaciones de PE; no se presentan como óptimos. Un crop puede recortar parte del objeto y no crea información nueva.

Usa una sesión Colab separada y el nuevo `RUN_NAME='pe_g14_448_v1'`. `PREVIOUS_RUN_NAME` debe apuntar a la carpeta anterior con `siglip_zero_shot_probabilities.npz`, no solo al CSV. Los IDs, hashes de datos, orden de clases y particiones de validación se comprueban antes de combinar. Si no se encuentra ese archivo, los CSV PE independientes siguen disponibles. El experimento anterior es un input de solo lectura.

El código [`pe_semantic.py`](pe_semantic.py) descarga solo los archivos necesarios del encoder oficial desde [esta revisión](https://github.com/facebookresearch/perception_models/tree/3e352cca660658d4b5c90f42a7808b11469e4c66/core/vision_encoder), verificando cada blob y conservando `LICENSE.PE` (Apache 2.0). Las revisiones de los pesos también quedan fijadas. La carga es estricta; los pesos permanecen FP32 y las operaciones matriciales usan autocast BF16. El modelo es público y no requiere el token gated DINO.

El benchmark mide el batch, velocidad y pico de memoria en la GPU real. El presupuesto inicial es de 7 horas con cachés por bloques y reanudación; no es una duración garantizada. Primero se exportan PE completo/flip y SigLIP+PE, después las vistas locales. Los CSV completos se pueden descargar si las fases posteriores se interrumpen. Las cachés incluyen cada crop, subconjunto, transformación y versión.

Se mantienen la calibración y la parte reservada de validación del experimento anterior. Además se evalúan hasta 1,000 imágenes reservadas con gris, posterización de cuatro bits y blur suave. Es un diagnóstico de sensibilidad, no una validación representativa de juguetes, cómics o tatuajes; no selecciona automáticamente un candidato ni busca pesos en el leaderboard.

Verificación CPU: `python3 -m unittest -q test_semantic_vit test_pe_semantic` ejecuta 29 tests sintéticos. Incluyen formatos reales de serialización con un modelo pequeño, carga estricta y congelación, recortes, selección sin etiquetas, separación de cachés y lectura alineada de predicciones previas. No sustituyen la ejecución del checkpoint completo en A100.

Fuentes: [modelo PE-Core-G14-448](https://huggingface.co/facebook/PE-Core-G14-448), [transformación oficial](https://github.com/facebookresearch/perception_models/blob/3e352cca660658d4b5c90f42a7808b11469e4c66/core/vision_encoder/transforms.py).

## Experimento SigLIP + DINOv3

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/vit_semantic.ipynb)

Experimento para el cambio de dominio entre CIFAR-10 y las representaciones sintéticas de objetos de [MVA RecVis 2026](https://www.kaggle.com/competitions/mva-recvis-2026). **Resultados comunicados por el participante: SigLIP zero-shot 94.015%; ensemble 94.844%.** El notebook CNN anterior se conserva en [`main_notebook.ipynb`](main_notebook.ipynb), y ResNet en [`resnet18_augmix.ipynb`](resnet18_augmix.ipynb).

## Modelos y etiquetas

| Rama | Procesamiento | Entrenamiento con nuestras etiquetas | Resultado |
|---|---|---|---|
| SigLIP 2 Giant semántico | 384×384 bilinear; normalización oficial 0.5/0.5; imagen completa y flip | Ninguno para clasificar; temperatura en validación | `submission_siglip_zero_shot.csv` |
| SigLIP 2 Giant + cabeza | Igual; train original + una vista AugMix suave | Cabeza lineal de 10 clases | `submission_siglip_probe.csv` |
| DINOv3 ViT-7B + cabeza | 256×256 bilinear; normalización ImageNet; CLS + media de patches | Cabeza lineal de 10 clases | `submission_dino_7b_probe.csv` |
| Combinación | Probabilidades calibradas; pesos fijos 50% / 25% / 25% | Sin optimizar pesos contra Kaggle | `submission_vit_ensemble.csv` |

SigLIP aporta la alineación entre conceptos y representaciones visuales adquirida en preentrenamiento imagen-texto. DINO aporta características visuales autosupervisadas. Ambos extractores permanecen congelados en este experimento: SSL describe su preentrenamiento; las etiquetas de train entrenan ahora las cabezas supervisadas. La complementariedad y la mejora sobre la CNN son hipótesis a comprobar.

Los prompts proceden íntegramente de las **80 plantillas públicas de OpenCLIP**, sin añadir ni seleccionar descripciones propias. Se fijan por [commit](https://github.com/mlfoundations/open_clip/blob/4cc208d3a44a2fb6d6435006a46f520d2ccc29b1/src/open_clip/zero_shot_metadata.py) y SHA256; se usa un parser que no ejecuta el código descargado. Se conserva su [licencia MIT](OPENCLIP_LICENSE.txt). Las reglas prohíben usar modelos cerrados incluso para generar prompts: no se utiliza esa vía, ni APIs de inferencia. Los nombres y el orden de las clases son los oficiales de CIFAR-10.

## Ejecutar en A100

1. Abre **[`vit_semantic.ipynb`](vit_semantic.ipynb)** en Colab y selecciona A100. La configuración inicial usa DINO-7B y un presupuesto de sesión de 7 h.
2. Ejecuta las celdas en orden. El notebook descarga [`semantic_vit.py`](semantic_vit.py) desde una revisión fija y comprueba su hash. Instala Transformers 4.57.3; conserva PyTorch/torchvision de Colab.
3. Monta Drive. Los resultados van a `MyDrive/A3_Recv/runs/vit_giant_7b_v1`; las características reutilizables, a `MyDrive/A3_Recv/cache_semantic_vit`.
4. Autentica Kaggle mediante `KAGGLE_API_TOKEN` en secretos o `kagglehub.login()`. Los datos privados se descargan a Colab, no al repositorio.
5. Ejecuta SigLIP. Guarda primero un CSV semántico, después la cabeza supervisada y `submission_siglip_blend.csv` (mezcla 50/50 de las dos ramas SigLIP).
6. En tu copia privada, escribe el token Hugging Face en **`HF_TOKEN = ""`**. DINO requiere acceso concedido al modelo exacto; se comprueban identidad y permisos sin imprimir ni guardar el token. SigLIP es público y no requiere ese permiso.
7. Ejecuta DINO y el ensemble. Descarga el ZIP de resultados y sube **CSV individuales** a Kaggle. El ZIP de resultados no es el ZIP de código de la entrega.

Para una primera comparación, sube `submission_siglip_zero_shot.csv` y `submission_vit_ensemble.csv`. Si DINO no termina, puedes comparar las ramas SigLIP. El ensemble no está garantizado como ganador; conservar las ramas permite comprobarlo.

## GPU, presupuesto y reanudación

- Los modelos grandes se cargan **secuencialmente**, en BF16 con SDPA. Se prueban varios tamaños de batch y se registra rendimiento y pico de VRAM. No hay una estimación de horas inventada ni una garantía de completar todo en una sesión.
- DINO-7B tiene aproximadamente 6.7B parámetros. Por defecto se ejecuta completo y congelado; no se almacenan gradientes u optimizadores del backbone. `dino_variant='hplus'` (aprox. 840M) o `'large'` (aprox. 300M) son alternativas explícitas. Cambia también `RUN_NAME`; no hay sustitución automática.
- Train/val/test se conservan como uint8 a 32×32. El resize y normalización se prueban por batch en GPU, contrastados contra la referencia CPU en píxeles y embeddings. Si superan una diferencia de 0.025 o el coseno baja de 0.999, se usa automáticamente el procesador de referencia CPU para todas las imágenes de esa rama; el modelo permanece en GPU. Los valores, criterios incumplidos y backend elegido quedan en `siglip_preprocessing_check.json` o `dino_preprocessing_check.json`. Las características y cabezas de la vía de referencia se guardan por separado para evitar mezclar preprocesados. Escalar no reconstruye información perdida.
- Cachés FP16 por bloques de 2,048 imágenes con identidad del modelo, revisión, datos, transformación y versiones; verificación de forma y hash antes de reutilizarlas. Los bloques completos sobreviven a una interrupción.
- Las cabezas guardan parámetros, estado del optimizador e historial después de cada época. Mantén `RUN_NAME` para reanudar la misma configuración. Se mide el presupuesto entre bloques/épocas; no interrumpe una descarga ni garantiza la latencia de Drive.
- Reserva varios GB en Drive para características y decenas de GB de disco temporal en Colab para checkpoints. Los checkpoints completos no se copian a Drive ni se incluyen en GitHub.

## Validación y controles

Se divide la validación original por clase: 20% para selección de cabeza, 20% para temperatura y 60% para el informe. La cabeza se elige por la media de accuracy limpia y con una vista AugMix determinista. Máximo 80 épocas, paciencia 10 desde época 10, AdamW, LR inicial 1e-3 con descenso coseno. El conjunto original ya se utilizó en experimentos anteriores; no es una evaluación externa nueva.

AugMix aporta un diagnóstico de robustez, pero no simula necesariamente dibujos, juguetes o tatuajes. La validación limpia tampoco estima por sí sola el rendimiento del test sintético. No se usan etiquetas ocultas, pseudoetiquetas de test, histogram matching, recortes agresivos ni generación de imágenes.

Los pesos del ensemble se fijan de antemano. Las temperaturas se ajustan únicamente en el subconjunto de calibración. Se guardan probabilidades e IDs, además de CSV, para reproducir la combinación. Cada CSV se alinea con `sample_submission.csv` y valida las diez clases.

Pruebas CPU: `python3 -m unittest -v test_semantic_vit` (requiere NumPy y PyTorch). Los 18 tests cubren selección del procesador de referencia ante diferencias de píxeles o embeddings, aislamiento de sus cachés y cabezas, exclusión de registros DINO, integridad, reanudación exacta de cabeza y optimizador, límites de uso de etiquetas y orden de IDs. **No equivalen a ejecutar los checkpoints completos en A100.** El notebook valida el preprocesamiento y mide el rendimiento real al ejecutarlo.

Referencias: [SigLIP 2 Giant](https://huggingface.co/google/siglip2-giant-opt-patch16-384), [su procesador](https://huggingface.co/google/siglip2-giant-opt-patch16-384/blob/main/preprocessor_config.json), [DINOv3 ViT-7B](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m), [transformación oficial DINOv3](https://github.com/facebookresearch/dinov3#image-transform).

## Resultados anteriores comunicados por el participante

| Experimento | Kaggle público |
|---|---:|
| ResNet18 bilinear | 61.093% |
| ResNet18 AugMix | 75.796% |
| DINOv3 ConvNeXt Base: cabeza lineal | 82.269% |
| DINOv3 ConvNeXt Base: ajuste parcial | 86.217% |
| SigLIP 2 Giant: cabeza lineal | 92.019% |
| DINOv3 ViT-7B: cabeza lineal | 92.978% |
| SigLIP 2 Giant: zero-shot | 94.015% |
| Ensemble SigLIP/DINO 50/25/25 | 94.844% |
| PE-Core-G: nuevo experimento | Pendiente |

<details>
<summary>Documentación del experimento CNN anterior</summary>

## A3 RecVis — DINOv3 ConvNeXt en Colab

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

</details>
