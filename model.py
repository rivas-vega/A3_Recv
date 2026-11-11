"""Model definitions and training utilities."""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.amp import GradScaler, autocast

from data import CLASS_NAMES


def build_model(name="resnet18", pretrained=False, image_size=32):
    """Build one of the supported supervised baselines."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        if image_size == 32 and not pretrained:
            model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        return model
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        return model
    if name == "vit_b_16":
        if image_size != 224:
            raise ValueError("The ViT-B/16 baseline requires 224px inputs.")
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASS_NAMES))
        return model
    raise ValueError(f"Unknown model: {name}")


def freeze_backbone(model) -> None:
    """Freeze all parameters except the classification head."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = model.fc if hasattr(model, "fc") else model.heads
    for parameter in head.parameters():
        parameter.requires_grad = True


def run_epoch(model, loader, optimizer, device, scaler=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = total_correct = total = 0
    amp = device.type == "cuda"
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def train(model, train_loader, validation_loader, epochs=10, lr=3e-4, weight_decay=1e-4, device=None):
    """Train a model, restore its best validation weights, and return metrics."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    best_accuracy, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, scaler)
        validation_metrics = run_epoch(model, validation_loader, None, device)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}: train_acc={row['train_accuracy']:.4f}, "
            f"val_acc={row['val_accuracy']:.4f}"
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    return pd.DataFrame(history)
