# A3 RecVis — Colab

[Open notebook in Colab](https://colab.research.google.com/github/rivas-vega/A3_Recv/blob/main/main_notebook.ipynb)

Starter adapted from https://github.com/fcendra-enpc/recvis26_a3 (original license retained).

## Workflow

1. Open the notebook in Colab and save a working copy if needed.
2. Run the KaggleHub setup cell. Configure KAGGLE_API_TOKEN in Colab Secrets or run kagglehub.login() separately if authentication is needed. Browser login is not automatically shared with Colab Python.
3. Obtain train.npz, validation.npz, test.npz and sample_submission.csv from the authenticated Kaggle competition: https://www.kaggle.com/competitions/mva-recvis-2026/data
4. To view images, run only the KaggleHub setup and preview cells. It displays 24 random test images, not the entire dataset. Only test.npz is needed for this step.
5. Training needs all four files. Select a GPU before training. The inherited baseline notebook does not yet support checkpoint/resume; do not rely on it for long runs.
6. Upload the generated submission.csv to Kaggle manually.

## Storage

This repository is public. Do not commit private competition data, credentials, model weights, or notebook outputs containing test images. Data/checkpoints are ignored by Git. Keep notebook outputs cleared before committing. The .gitignore does not remove outputs embedded in notebooks.

Google Drive files are not automatically accessible to the assistant. Dataset access must be established separately through Colab or an uploaded file; GitHub provides shared access to the code only.

## Code

- main_notebook.ipynb: self-contained baseline with Colab paths and random preview.
- main.py, model.py, data.py: original command-line baseline.
- requirements.txt: upstream dependencies; Colab already includes most baseline packages.

Training and validation are ordinary CIFAR-10; the synthetic test domain differs. Record validation and Kaggle scores separately. No training or scoring has been performed in this repository yet.

## Submission

One-page CVPR report (references excluded) and code ZIP under 8 MB; exclude datasets, weights, credentials and generated outputs.
