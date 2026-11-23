# A3 RecVis experiments

Pedro Rivas Vega

My experiments on synthetic CIFAR-10, from the first CNN baseline to PE zero-shot and a smaller MobileCLIP model. The notebooks are numbered in the order I worked through the project.

| Notebook | Experiment | Public Kaggle score |
|---|---|---|
| [00 — Image exploration](00_image_exploration.ipynb) | RGB histograms, color OT, interpolation and quantization | Exploration only |
| [01 — ResNet-18](01_resnet18_augmix.ipynb) | Baseline / AugMix | 0.61093 / 0.75796 |
| [02 — DINOv3 ConvNeXt](02_dinov3_convnext.ipynb) | Linear probe / partial fine-tuning | 0.82269 / 0.86217 |
| [03 — Large ViTs](03_vit_semantic.ipynb) | SigLIP probe / DINO probe / SigLIP zero-shot / ensemble | 0.92019 / 0.92978 / 0.94015 / 0.94844 |
| [04 — PE zero-shot](04_pe_zero_shot.ipynb) | Full image + horizontal flip | **0.96776** |
| [05 — MobileCLIP2-S0](05_mobileclip_s0_cinic.ipynb) | Supervised fine-tuning on CINIC-10 | 0.84364 |

PE was my best-scoring submission. I then tried MobileCLIP to compare accuracy with a much smaller model. The OT and histogram analysis stayed in the exploration notebook; none of its test statistics or transforms were used in the scored pipelines. DINOv3 ConvNeXt uses a published distilled checkpoint; I did not rerun teacher-student distillation.

## Open in Colab

- [00_image_exploration](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/finished/colabs/00_image_exploration.ipynb)
- [01_resnet18_augmix](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/finished/colabs/01_resnet18_augmix.ipynb)
- [02_dinov3_convnext](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/finished/colabs/02_dinov3_convnext.ipynb)
- [03_vit_semantic](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/finished/colabs/03_vit_semantic.ipynb)
- [04_pe_zero_shot](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/finished/colabs/04_pe_zero_shot.ipynb)
- [05_mobileclip_s0_cinic](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/finished/colabs/05_mobileclip_s0_cinic.ipynb)

## Run setup

- Run each model notebook in its own Colab session, following the cells in order. The large-model notebooks target an A100; my MobileCLIP run used an L4.
- Use the Kaggle account that joined `mva-recvis-2026`. Authentication is handled through Colab Secrets or the login cell. DINOv3 also requires Hugging Face access and a read token; token fields are empty in this copy.
- Mount Drive where requested. Keep `RUN_NAME` and the configuration to resume a run; use a new name for a separate experiment.
- For PE, set **`RUN_INFERENCE=True`** to generate predictions. With `False`, the notebook reads the previous run from Drive. The inference path uses PE alone, with full-image/flip averaging.
- For MobileCLIP, complete training and checkpoint selection before exporting with **`EXPORT_FINAL=True`**.
- Download each submission CSV and upload it separately on Kaggle. The table above records my earlier public scores; a new run produces its own local metrics and CSV files.

## Files and dependencies

The Python modules beside the notebooks contain the experiment functions. The notebooks download the same modules from fixed GitHub revisions and check their SHA256 hashes. Model checkpoints and datasets are downloaded separately into Colab or Drive.

| Module | SHA256 |
|---|---|
| `semantic_vit.py` | `8d3baa53881282e34cbdb21c3527c7164213f3e78c1d921f393193c011958afc` |
| `pe_semantic.py` | `00f6ede2beb99c5abe82e3a6813e06f1802d2dedca3b82576fb2477ce951c214` |
| `mobileclip_s0_cinic.py` | `5b524949368f0f21890d6fcfab864c24ac7da8e4b142cf7078c4b53a700ad7d7` |

The code ZIP contains these notebooks, helper modules and licenses, without weights, data or saved notebook outputs. Experiment settings and model references are documented in each notebook.
