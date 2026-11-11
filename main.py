"""Train a baseline model and create a Kaggle submission."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import CLASS_NAMES, make_loaders, predict
from model import build_model, freeze_backbone, train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("RECVIS_INPUT_DIR", "/kaggle/input/recvis-mva-a3-2026")),
        help="Directory containing train.npz, validation.npz, test.npz, and sample_submission.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("RECVIS_OUTPUT_DIR", ".")),
        help="Directory for the checkpoint, history, and submission.",
    )
    parser.add_argument("--model", choices=("resnet18", "resnet50", "vit_b_16"), default="resnet18")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained weights.")
    parser.add_argument("--linear-probe", action="store_true", help="Train only the classifier head.")
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to 32, or 224 for pretrained models and ViT.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_inputs(data_dir: Path) -> None:
    required = ("train.npz", "validation.npz", "test.npz", "sample_submission.csv")
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {data_dir}: {missing}")
    with np.load(data_dir / "test.npz") as archive:
        if "labels" in archive.files:
            raise ValueError("test.npz must not contain labels.")


def main() -> None:
    args = parse_args()
    if args.linear_probe and not args.pretrained:
        raise ValueError("--linear-probe requires --pretrained.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    validate_inputs(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    image_size = args.image_size or (224 if args.pretrained or args.model == "vit_b_16" else 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model} image_size={image_size} data={args.data}")

    train_loader, validation_loader = make_loaders(
        args.data,
        image_size=image_size,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    model = build_model(args.model, pretrained=args.pretrained, image_size=image_size)
    if args.linear_probe:
        freeze_backbone(model)

    history = train(
        model,
        train_loader,
        validation_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )
    history_path = args.output / "history.csv"
    history.to_csv(history_path, index=False)

    checkpoint_path = args.output / "model.pth"
    torch.save(
        {
            "model_name": args.model,
            "pretrained": args.pretrained,
            "image_size": image_size,
            "class_names": CLASS_NAMES,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    submission = predict(
        model,
        args.data / "test.npz",
        image_size=image_size,
        pretrained=args.pretrained,
        batch_size=max(args.batch_size, 256),
        workers=args.workers,
        device=device,
    )
    sample = pd.read_csv(args.data / "sample_submission.csv")
    if sample.columns.tolist() != ["id", "label"]:
        raise ValueError("sample_submission.csv must contain exactly the columns id,label.")
    if not np.array_equal(sample["id"].to_numpy(), submission["id"].to_numpy()):
        raise ValueError("Submission IDs are not aligned with sample_submission.csv.")
    if not submission["label"].between(0, 9).all():
        raise ValueError("Predicted labels must be integers from 0 to 9.")

    submission_path = args.output / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"saved {checkpoint_path}")
    print(f"saved {history_path}")
    print(f"saved {submission_path}")


if __name__ == "__main__":
    main()
