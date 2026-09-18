"""
Training and Evaluation script for the Dual-Stream (Global + Local) Vision Architecture.
Fuses whole-plot baseline noise context with high-resolution red-marker ROI curvatures.
Generates confusion matrices on both the holdout test set and the 19 real NASA Swift-BAT subjects.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from data_loader import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, MockBurstChaserLoader
from dataset import DualStreamBurstChaserDataset, create_stratified_splits
from model import DualStreamBurstChaserClassifier
from train import compute_inverse_class_weights
from experiments import FocalLoss


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """
    Renders and saves a formatted confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(7, 6), dpi=150)
    sns.set_theme(style="white")

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            annot[i, j] = f"{count}\n({pct:.1f}%)"

    ax = sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=1.5,
        linecolor="white",
        annot_kws={"size": 11, "weight": "normal"},
    )

    plt.title(title, fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, weight="bold", labelpad=8)
    plt.ylabel("True Class (Ground Truth)", fontsize=11, weight="bold", labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrix: {output_path}")


def train_dual_stream(
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 2e-4,
    device: str = "cpu",
    output_dir: str = "checkpoints",
    eval_only: bool = False,
    data_csv: Optional[str] = None,
) -> dict:
    device = torch.device(device)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Prepare data
    if data_csv and os.path.exists(data_csv):
        print(f"Loading Dual-Stream dataset from: {data_csv}")
        df = pd.read_csv(data_csv)
    else:
        mock_dir = "data/mock_burst_chaser"
        loader = MockBurstChaserLoader(output_dir=mock_dir)
        df = loader.generate_dataset(num_samples=80)

    train_df, val_df, test_df = create_stratified_splits(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42)

    train_dataset = DualStreamBurstChaserDataset(train_df, is_train=True)
    val_dataset = DualStreamBurstChaserDataset(val_df, is_train=False)
    test_dataset = DualStreamBurstChaserDataset(test_df, is_train=False)

    eff_batch_size = max(2, min(batch_size, len(train_df))) if len(train_df) >= 2 else 1
    drop_last_train = (len(train_df) > eff_batch_size and len(train_df) % eff_batch_size == 1)
    train_loader = DataLoader(train_dataset, batch_size=eff_batch_size, shuffle=True, drop_last=drop_last_train)
    val_loader = DataLoader(val_dataset, batch_size=eff_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eff_batch_size, shuffle=False)

    # 2. Build model
    model = DualStreamBurstChaserClassifier(num_classes=3, pretrained=True, dropout=0.3).to(device)

    # 3. Class weights and Focal Loss
    train_labels = train_df["label_id"].values
    class_weights = compute_inverse_class_weights(train_labels, num_classes=3).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_weights_path = out_path / "dual_stream_best.pth"
    if not best_weights_path.exists():
        torch.save(model.state_dict(), best_weights_path)

    # 4. Training loop (skip if eval_only and weights exist)
    if not (eval_only and best_weights_path.exists()):
        print("Beginning Dual-Stream Model Training...")
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                x_g = batch["image_global"].to(device)
                x_l = batch["image_local"].to(device)
                y = batch["label"].to(device)

                if y.size(0) <= 1 and len(train_loader) > 1:
                    continue

                optimizer.zero_grad()
                logits = model(x_g, x_l)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * y.size(0)

            scheduler.step()
            train_loss = running_loss / max(len(train_dataset), 1)

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x_g = batch["image_global"].to(device)
                    x_l = batch["image_local"].to(device)
                    y = batch["label"].to(device)
                    logits = model(x_g, x_l)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(y.cpu().numpy())

            val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
            val_acc = accuracy_score(val_targets, val_preds)
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_weights_path)
    else:
        print(f"Using pre-trained checkpoint from: {best_weights_path}")

    # Load best checkpoint
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    # 5. Evaluate on Holdout Test Set
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_g = batch["image_global"].to(device)
            x_l = batch["image_local"].to(device)
            logits = model(x_g, x_l)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(batch["label"].cpu().numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    print(f"\nHoldout Test Accuracy: {test_acc:.4f} | Macro F1: {test_f1:.4f}")

    holdout_cm_path = str(out_path / "dual_stream_holdout_cm.png")
    plot_confusion_matrix(
        test_targets,
        test_preds,
        class_names=CLASS_NAMES,
        output_path=holdout_cm_path,
        title="Dual-Stream Model (Global + Local)\nHoldout Test Confusion Matrix",
    )

    # 6. Evaluate on Real NASA Swift-BAT Subjects
    real_csv_path = "data/zooniverse/subjects_manifest.csv"
    if os.path.exists(real_csv_path):
        real_df = pd.read_csv(real_csv_path)
        if "label_id" in real_df.columns:
            real_df = real_df[real_df["label_id"].isin([0, 1, 2])].copy()
        elif "label" in real_df.columns:
            real_df = real_df[real_df["label"].isin(CLASS_NAMES)].copy()

        if "image_path" in real_df.columns:
            real_df = real_df[real_df["image_path"].apply(lambda p: os.path.exists(str(p)) if pd.notna(p) else False)].copy()

        if len(real_df) > 0:
            real_dataset = DualStreamBurstChaserDataset(real_df, is_train=False)
            real_loader = DataLoader(real_dataset, batch_size=1, shuffle=False)

            real_preds, real_targets = [], []
            with torch.no_grad():
                for batch in real_loader:
                    x_g = batch["image_global"].to(device)
                    x_l = batch["image_local"].to(device)
                    logits = model(x_g, x_l)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    real_preds.extend(preds)
                    real_targets.extend(batch["label"].cpu().numpy())

            real_preds = np.array(real_preds)
            real_targets = np.array(real_targets)
            real_acc = accuracy_score(real_targets, real_preds)
            real_f1 = f1_score(real_targets, real_preds, average="macro", zero_division=0)
            print(f"Real Swift-BAT Evaluation: Acc = {real_acc:.4f} | Macro F1 = {real_f1:.4f}")
            print("\nReal Swift-BAT Classification Report:")
            print(classification_report(real_targets, real_preds, target_names=CLASS_NAMES, zero_division=0))

            real_cm_path = str(out_path / "dual_stream_real_cm.png")
            plot_confusion_matrix(
                real_targets,
                real_preds,
                class_names=CLASS_NAMES,
                output_path=real_cm_path,
                title=f"Dual-Stream Model (Global + Local)\nReal NASA Swift-BAT Subjects (N={len(real_df)})",
            )
        else:
            real_acc, real_f1, real_cm_path = None, None, None
    else:
        real_acc, real_f1, real_cm_path = None, None, None

    results = {
        "holdout_test_accuracy": float(test_acc),
        "holdout_test_macro_f1": float(test_f1),
        "holdout_cm_path": holdout_cm_path,
        "real_accuracy": float(real_acc) if real_acc is not None else None,
        "real_macro_f1": float(real_f1) if real_f1 is not None else None,
        "real_cm_path": real_cm_path,
    }

    with open(out_path / "dual_stream_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or Evaluate Dual-Stream Burst Chaser Model")
    parser.add_argument("--data_csv", type=str, default=None, help="Path to training CSV manifest (e.g. data/annotated_training_data.csv)")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda)")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation without retraining")
    args = parser.parse_args()

    train_dual_stream(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        eval_only=args.eval_only,
        data_csv=args.data_csv,
    )
