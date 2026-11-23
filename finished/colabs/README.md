# A3 RecVis - Colab notebooks

Pedro Rivas Vega

The archive contains the five Google Colab notebooks covering all ten Kaggle submissions in the report, plus a notebook documenting the initial image exploration. Each notebook is supplied as an `.ipynb` file. Auxiliary Python modules and licenses are included. No pretrained weights, datasets, feature caches, credentials or saved notebook outputs are included.

## Open directly in Google Colab

| Notebook | Google Colab |
|---|---|
| [00_image_exploration](00_image_exploration.ipynb) | [Open in Colab](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/finished/colabs/00_image_exploration.ipynb) |
| [01_resnet18_augmix](01_resnet18_augmix.ipynb) | [Open in Colab](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/finished/colabs/01_resnet18_augmix.ipynb) |
| [02_dinov3_convnext](02_dinov3_convnext.ipynb) | [Open in Colab](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/finished/colabs/02_dinov3_convnext.ipynb) |
| [03_vit_semantic](03_vit_semantic.ipynb) | [Open in Colab](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/finished/colabs/03_vit_semantic.ipynb) |
| [04_pe_zero_shot](04_pe_zero_shot.ipynb) | [Open in Colab](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/finished/colabs/04_pe_zero_shot.ipynb) |
| [05_mobileclip_s0_cinic](05_mobileclip_s0_cinic.ipynb) | [Open in Colab](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/finished/colabs/05_mobileclip_s0_cinic.ipynb) |

The notebooks and helper modules in this folder are identical to those in the [code archive](../A3_RIVAS_VEGA_Pedro.zip). The links above open the copies stored in `finished/colabs`; the original notebooks also retain their historical links.

## Experimental progression

| Notebook | Experiment | Submitted CSV and public score |
|---|---|---|
| `00_image_exploration.ipynb` | RGB quantization/contrast, nearest/bilinear/bicubic/Lanczos upsampling, RGB histograms and marginal color OT | Exploration only; did not produce a scored submission |
| `01_resnet18_augmix.ipynb` | ImageNet ResNet-18: basic augmentation and AugMix/JSD | `submission_baseline.csv`: 0.61093; `submission_augmix.csv`: 0.75796 |
| `02_dinov3_convnext.ipynb` | Distilled DINOv3 ConvNeXt-Base checkpoint: linear probe and partial fine-tuning | `submission_dinov3_linear_256.csv`: 0.82269; `submission_dinov3_partial_ft_256.csv`: 0.86217 |
| `03_vit_semantic.ipynb` | Frozen SigLIP 2 Giant and DINOv3 ViT-7B; supervised probes, zero-shot classification and fixed ensemble | `submission_siglip_probe.csv`: 0.92019; `submission_dino_7b_probe.csv`: 0.92978; `submission_siglip_zero_shot.csv`: 0.94015; `submission_vit_ensemble.csv`: 0.94844 |
| `04_pe_zero_shot.ipynb` | PE-Core-G14-448 with full-image and horizontal-flip scores | `submission_pe_zero_shot.csv`: 0.96776 |
| `05_mobileclip_s0_cinic.ipynb` | 11.4M-parameter MobileCLIP2-S0 classifier fine-tuned on CINIC-10 | `submission_mobileclip2_s0_cinic.csv`: 0.84364 |

PE has the highest measured public score. MobileCLIP is the subsequent compact-model comparison. The RGB/OT exploration used aggregate unlabeled test statistics, but its transformations/statistics were not used by any scored pipeline. The DINOv3 CNN uses published distilled weights; no teacher-student distillation was rerun for the assignment.

## Run in Colab

1. Open the desired `.ipynb` through Colab's **File > Upload notebook**. Run each model notebook in a separate session, following its cells in order. Setup/install cells declare their dependencies.
2. Use a GPU runtime for model experiments. The large-encoder notebooks were designed for an A100; the recorded MobileCLIP run used an NVIDIA L4. The exploration notebook does not require GPU inference.
3. Use an account authorized for `mva-recvis-2026` and supply Kaggle credentials through the notebook's supported login or Colab Secrets. The scored notebooks download the competition's `train.npz`, `validation.npz`, `test.npz` and `sample_submission.csv`. DINOv3 also requires access to its gated Hugging Face checkpoints; follow the token/access cells without saving a token into a public notebook.
4. Mount Drive where requested. Existing checkpoints are used for resumption, so choose a fresh run directory when intentionally starting a new independent run.
5. `04_pe_zero_shot.ipynb` defaults to reading previous results. Set **`RUN_INFERENCE=True`** in its configuration cell for reproduction. The existing `PEZeroShotOnly` path disables the PE/SigLIP blend and local-view branch; it runs exactly the reported full-image/flip method. Previous SigLIP predictions are not required.
6. For MobileCLIP, complete training and then run the final export cell with **`EXPORT_FINAL=True`**. It exports only after checkpoint selection. Its setup downloads the public CINIC-10 train/validation mirror, in addition to pretrained weights.
7. Download each required CSV from the corresponding notebook. The notebooks do not submit automatically to Kaggle. Scores in this README are the previously recorded public scores, not newly executed results.

The notebooks obtain their helper modules and official model code from pinned repository revisions at runtime, verifying the recorded hashes. Copies of the same helper modules are included in the archive for inspection. Pretrained checkpoints are downloaded separately; the package is not intended for offline execution.

## Provenance and packaging

Source repository: https://github.com/rivas-vega/A3_Recv

Model notebooks were obtained from repository snapshot `0c7aa9f4f047df746527a9a2861a4c5ea3de17d0` and renamed with numerical prefixes. Their executable code was preserved; saved cell outputs, execution counts and widget metadata were removed.

`00_image_exploration.ipynb` extracts the data setup and exploration cells (original cells 4-15) from historical `main_notebook.ipynb` at commit `6f30b0d304`. It adds an explanatory heading and omits the later, unrelated training cells and the button that would open the current repository notebook. Its original exploration code is unchanged.

Helper identities verified in the archive:

| Module | SHA256 |
|---|---|
| `semantic_vit.py` | `8d3baa53881282e34cbdb21c3527c7164213f3e78c1d921f393193c011958afc` |
| `pe_semantic.py` | `00f6ede2beb99c5abe82e3a6813e06f1802d2dedca3b82576fb2477ce951c214` |
| `mobileclip_s0_cinic.py` | `5b524949368f0f21890d6fcfab864c24ac7da8e4b142cf7078c4b53a700ad7d7` |

The PE notebook reuses the previous weight-revision manifest when available; otherwise it resolves and records the public checkpoint revision on first use. The original Drive manifests were not included during packaging, and no exact cross-environment identity of predictions is promised.

Verification covers notebook JSON, Python cell syntax, absence of saved outputs, module hashes and ZIP integrity. Full GPU training/inference was not rerun for this hand-in package. The retained model notebooks include their original Spanish explanations.
