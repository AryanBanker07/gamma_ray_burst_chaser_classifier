"""
Training module for NASA Zooniverse Burst Chaser classification.
Features:
- Fine-tuning pretrained ResNet-18 / ConvNeXt-Tiny for 3 classes (pulse, noise, unclear).
- Inverse class frequency weighting in Cross-Entropy Loss to counter class imbalance.
- Validation tracking for loss, accuracy, and macro F1-score.
- Automatic best checkpoint saving (best_model.pth) and holdout test set persistence.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report

from data_loader import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    MockBurstChaserLoader,
    ZooniverseBurstChaserLoader,
)
from dataset import (
    BurstChaserDataset,
    build_dataloaders,
    create_stratified_splits,
)
from model import BurstChaserClassifier


def compute_inverse_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """
    Computes inverse class frequency weights:
    w_c = N / (num_classes * count_c)
    """
    total_samples = len(labels)
    class_counts = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        cnt = np.sum(labels == c)
        class_counts[c] = max(cnt, 1)  # avoid division by zero

    weights = total_samples / (num_classes * class_counts)
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    print(f"Class counts in training set: {class_counts.astype(int)}")
    print(f"Computed inverse class weights: {np.round(weights, 3)}")
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Runs one training epoch. Returns (avg_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)

        if targets.size(0) <= 1 and len(dataloader) > 1:
            continue

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluates model on validation or test DataLoader.
    Returns (loss, accuracy, macro_f1, all_targets, all_preds).
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    total = len(all_targets)
    val_loss = running_loss / max(total, 1)
    val_acc = accuracy_score(all_targets, all_preds) if total > 0 else 0.0
    val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) if total > 0 else 0.0

    return val_loss, val_acc, val_f1, all_targets, all_preds


def run_training(
    data_csv: Optional[str] = None,
    use_mock: bool = False,
    fetch_zooniverse: bool = False,
    max_subjects: Optional[int] = 100,
    num_mock_samples: int = 120,
    backbone: str = "resnet18",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    crop_roi: bool = True,
    checkpoint_dir: str = "checkpoints",
    device_name: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full training pipeline orchestration.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # 1. Dataset loading / generation
    df: Optional[pd.DataFrame] = None
    if fetch_zooniverse:
        print("Fetching live Burst Chaser subjects from Zooniverse...")
        fetcher = ZooniverseBurstChaserLoader()
        df = fetcher.fetch_subjects(max_subjects=max_subjects)
        # Check how many have valid labels
        valid_count = len(df[df["label_id"].isin([0, 1, 2])])
        if valid_count < 10:
            print(f"Warning: Only {valid_count} Zooniverse subjects have ground truth labels. Augmenting with mock samples for robust training.")
            mock_loader = MockBurstChaserLoader(output_dir="data/mock_burst_chaser", seed=seed)
            df_mock = mock_loader.generate_dataset(num_samples=num_mock_samples)
            df = pd.concat([df[df["label_id"].isin([0, 1, 2])], df_mock], ignore_index=True)
    elif data_csv and os.path.exists(data_csv):
        print(f"Loading dataset from manifest: {data_csv}")
        df = pd.read_csv(data_csv)
    else:
        print("No existing manifest specified; generating realistic synthetic mock dataset...")
        mock_loader = MockBurstChaserLoader(output_dir="data/mock_burst_chaser", seed=seed)
        df = mock_loader.generate_dataset(num_samples=num_mock_samples)

    # 2. Stratified splits: 80% train, 10% val, 10% test
    train_df, val_df, test_df = create_stratified_splits(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=seed)

    # Save splits for reproducibility and evaluation
    train_df.to_csv(ckpt_path / "train_split.csv", index=False)
    val_df.to_csv(ckpt_path / "val_split.csv", index=False)
    test_df.to_csv(ckpt_path / "holdout_test.csv", index=False)
    print(f"Splits saved to {ckpt_path}. Holdout test set has {len(test_df)} samples.")

    # 3. DataLoaders
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        batch_size=batch_size,
        crop_roi=crop_roi,
    )

    # 4. Model & Loss function with inverse class weights
    model = BurstChaserClassifier(
        backbone_name=backbone,
        num_classes=3,
        pretrained=True,
        dropout=0.3,
    ).to(device)

    train_labels = train_df["label_id"].values
    class_weights = compute_inverse_class_weights(train_labels, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. Training loop
    best_val_f1 = -1.0
    best_epoch = -1
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}

    print("\n" + "=" * 70)
    print(f"Starting Training: {epochs} epochs | Backbone: {backbone} | ROI Cropping: {crop_roi}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_f1, 4))

        is_best = val_f1 > best_val_f1
        marker = " (*BEST*)" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.1f}% || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.1f}% | "
            f"Val Macro-F1: {val_f1:.4f}{marker}"
        )

        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "backbone": backbone,
                "crop_roi": crop_roi,
                "class_names": CLASS_NAMES,
                "class_to_idx": CLASS_TO_IDX,
            }
            torch.save(checkpoint, ckpt_path / "best_model.pth")

    # Save latest model and history
    torch.save(model.state_dict(), ckpt_path / "latest_model.pth")
    with open(ckpt_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Training Complete! Best model from Epoch {best_epoch} with Val Macro-F1: {best_val_f1:.4f}")
    print(f"Best checkpoint saved to: {ckpt_path / 'best_model.pth'}")
    print("=" * 70)

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "checkpoint_path": str(ckpt_path / "best_model.pth"),
        "holdout_test_path": str(ckpt_path / "holdout_test.csv"),
        "history": history,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Burst Chaser 3-Class Classifier")
    parser.add_argument("--data_csv", type=str, default=None, help="Path to manifest CSV")
    parser.add_argument("--use_mock", action="store_true", help="Generate and train on synthetic mock data")
    parser.add_argument("--fetch_zooniverse", action="store_true", help="Fetch subjects from live Zooniverse API")
    parser.add_argument("--max_subjects", type=int, default=50, help="Max subjects to fetch from Zooniverse")
    parser.add_argument("--num_mock_samples", type=int, default=120, help="Number of synthetic mock samples to generate")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "convnext_tiny"])
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--no_crop", action="store_true", help="Disable red elliptical ROI cropping")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        data_csv=args.data_csv,
        use_mock=args.use_mock,
        fetch_zooniverse=args.fetch_zooniverse,
        max_subjects=args.max_subjects,
        num_mock_samples=args.num_mock_samples,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        crop_roi=not args.no_crop,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
    )
