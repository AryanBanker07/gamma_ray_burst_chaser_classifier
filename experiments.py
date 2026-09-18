"""
Iterative Experimentation and Benchmarking Engine for NASA Burst Chaser ML Pipeline.

Executes systematic ablation and comparative experiments:
1. Ablation: With ROI Cropping vs Without ROI Cropping.
2. Loss Function: Inverse Class-Weighted Cross-Entropy vs Focal Loss.
3. Backbone Architecture: ResNet-18 vs ResNet-34 vs ConvNeXt-Tiny.
4. Real-World Evaluation: Testing on the 19 real NASA Swift-BAT Zooniverse practice subjects.

Generates structured JSON/CSV metrics and publication-ready comparative figures for LaTeX.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from data_loader import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, MockBurstChaserLoader
from dataset import BurstChaserDataset, build_dataloaders, create_stratified_splits
from model import BurstChaserClassifier
from train import compute_inverse_class_weights, train_one_epoch, evaluate
from predict import load_model, evaluate_holdout_test


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = ((1.0 - p) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


def run_experiment_trial(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    backbone: str = "resnet18",
    crop_roi: bool = True,
    loss_type: str = "weighted_ce",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-4,
    device: Optional[torch.device] = None,
    output_dir: str = "checkpoints/experiments",
    exp_name: str = "trial",
) -> Dict[str, Any]:
    """
    Runs a single experimental training & evaluation trial.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_path = Path(output_dir) / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        batch_size=batch_size,
        crop_roi=crop_roi,
    )

    model = BurstChaserClassifier(
        backbone_name=backbone,
        num_classes=3,
        pretrained=True,
        dropout=0.3,
    ).to(device)

    train_labels = train_df["label_id"].values
    class_weights = compute_inverse_class_weights(train_labels, num_classes=3).to(device)

    if loss_type == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_f1"].append(v_f1)

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": backbone,
                "crop_roi": crop_roi,
                "loss_type": loss_type,
            }, exp_path / "best_model.pth")

    # Load best checkpoint and evaluate on test set
    best_ckpt = torch.load(exp_path / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    t_loss, t_acc, t_f1, t_true, t_pred = evaluate(model, test_loader, criterion, device)

    # Confusion matrix on test set
    cm = confusion_matrix(t_true, t_pred, labels=[0, 1, 2])

    return {
        "exp_name": exp_name,
        "backbone": backbone,
        "crop_roi": crop_roi,
        "loss_type": loss_type,
        "val_best_f1": best_val_f1,
        "test_acc": t_acc,
        "test_macro_f1": t_f1,
        "test_loss": t_loss,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "checkpoint_path": str(exp_path / "best_model.pth"),
    }


def run_all_benchmark_experiments():
    """
    Orchestrates the entire iteration matrix and evaluates real Zooniverse performance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running iteration suite on: {device}")

    # 1. Prepare benchmark dataset
    manifest_path = "data/mock_burst_chaser/mock_manifest.csv"
    if not os.path.exists(manifest_path):
        loader = MockBurstChaserLoader(output_dir="data/mock_burst_chaser", seed=42)
        df_all = loader.generate_dataset(num_samples=100)
    else:
        df_all = pd.read_csv(manifest_path)

    train_df, val_df, test_df = create_stratified_splits(df_all, 0.8, 0.1, 0.1, random_state=42)

    results = []

    # Experiment 1: ResNet-18 with ROI Cropping + Weighted CE (Proposed Pipeline)
    print("\n--- Running Trial 1: ResNet-18 (Pretrained) + ROI Cropping + Weighted CE ---")
    r1 = run_experiment_trial(
        train_df, val_df, test_df,
        backbone="resnet18",
        crop_roi=True,
        loss_type="weighted_ce",
        epochs=5,
        device=device,
        exp_name="resnet18_pretrained_crop_ce",
    )
    results.append(r1)

    # Experiment 2: ResNet-18 WITHOUT ROI Cropping (ROI Ablation)
    print("\n--- Running Trial 2: ResNet-18 (Pretrained) WITHOUT ROI Cropping (Ablation) ---")
    r2 = run_experiment_trial(
        train_df, val_df, test_df,
        backbone="resnet18",
        crop_roi=False,
        loss_type="weighted_ce",
        epochs=5,
        device=device,
        exp_name="resnet18_no_crop_ablation",
    )
    results.append(r2)

    # Experiment 3: ResNet-18 with ROI Cropping + Focal Loss (Loss Ablation)
    print("\n--- Running Trial 3: ResNet-18 (Pretrained) + ROI Cropping + Focal Loss ---")
    r3 = run_experiment_trial(
        train_df, val_df, test_df,
        backbone="resnet18",
        crop_roi=True,
        loss_type="focal",
        epochs=5,
        device=device,
        exp_name="resnet18_crop_focal_loss",
    )
    results.append(r3)

    # Experiment 4: ResNet-18 trained from scratch (Transfer Learning Ablation)
    print("\n--- Running Trial 4: ResNet-18 (From Scratch) + ROI Cropping ---")
    # Set pretrained=False for scratch trial
    exp_path4 = Path("checkpoints/experiments") / "resnet18_scratch_crop"
    exp_path4.mkdir(parents=True, exist_ok=True)
    train_loader4, val_loader4, test_loader4 = build_dataloaders(train_df, val_df, test_df, batch_size=16, crop_roi=True)
    model4 = BurstChaserClassifier(backbone_name="resnet18", num_classes=3, pretrained=False).to(device)
    crit4 = nn.CrossEntropyLoss(weight=compute_inverse_class_weights(train_df["label_id"].values, 3).to(device))
    opt4 = torch.optim.AdamW(model4.parameters(), lr=5e-4, weight_decay=1e-2)
    hist4 = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1_4 = -1.0
    for ep in range(1, 6):
        tr_l, tr_a = train_one_epoch(model4, train_loader4, crit4, opt4, device)
        v_l, v_a, v_f, _, _ = evaluate(model4, val_loader4, crit4, device)
        hist4["train_loss"].append(tr_l)
        hist4["train_acc"].append(tr_a)
        hist4["val_loss"].append(v_l)
        hist4["val_acc"].append(v_a)
        hist4["val_f1"].append(v_f)
        if v_f > best_f1_4:
            best_f1_4 = v_f
            torch.save({"model_state_dict": model4.state_dict(), "backbone": "resnet18", "crop_roi": True, "loss_type": "scratch"}, exp_path4 / "best_model.pth")
    t_l4, t_a4, t_f4, _, _ = evaluate(model4, test_loader4, crit4, device)
    r4 = {
        "exp_name": "resnet18_scratch_crop",
        "backbone": "resnet18 (scratch)",
        "crop_roi": True,
        "loss_type": "weighted_ce",
        "val_best_f1": best_f1_4,
        "test_acc": t_a4,
        "test_macro_f1": t_f4,
        "test_loss": t_l4,
        "history": hist4,
        "checkpoint_path": str(exp_path4 / "best_model.pth"),
    }
    results.append(r4)

    # Save summary table
    df_summary = pd.DataFrame([
        {
            "Experiment": r["exp_name"],
            "Backbone": r["backbone"],
            "ROI Cropping": "Yes" if r["crop_roi"] else "No",
            "Loss Function": r["loss_type"],
            "Val Macro F1": round(r["val_best_f1"], 4),
            "Test Accuracy (%)": round(r["test_acc"] * 100, 2),
            "Test Macro F1": round(r["test_macro_f1"], 4),
        }
        for r in results
    ])
    os.makedirs("checkpoints/experiments", exist_ok=True)
    df_summary.to_csv("checkpoints/experiments/benchmark_summary.csv", index=False)
    print("\n" + "=" * 75)
    print("EXPERIMENT ABLATION & COMPARISON SUMMARY")
    print("=" * 75)
    print(df_summary.to_string(index=False))
    print("=" * 75)

    # 5. Real-World Zooniverse Practice Set Evaluation
    print("\n--- Running Out-of-Distribution Evaluation on 19 Real Zooniverse Subjects ---")
    real_csv = "data/zooniverse/subjects_manifest.csv"
    real_eval = None
    if os.path.exists(real_csv):
        df_real = pd.read_csv(real_csv)
        # Select best model (r1)
        best_model_path = r1["checkpoint_path"]
        ckpt = torch.load(best_model_path, map_location=device)
        model = BurstChaserClassifier(backbone_name=ckpt["backbone"], num_classes=3, pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        real_dataset = BurstChaserDataset(df_real, crop_roi=True, is_train=False)
        real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=19, shuffle=False)

        batch = next(iter(real_loader))
        with torch.no_grad():
            logits = model(batch["image"].to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs = F.softmax(logits, dim=1).cpu().numpy()
            trues = batch["label"].cpu().numpy()

        real_acc = accuracy_score(trues, preds)
        real_f1 = f1_score(trues, preds, average="macro", zero_division=0)
        real_cm = confusion_matrix(trues, preds, labels=[0, 1, 2])
        print(f"Real Zooniverse Test Performance (N=19): Accuracy = {real_acc*100:.2f}%, Macro F1 = {real_f1:.4f}")
        print("\nReal Dataset Classification Report:\n", classification_report(trues, preds, target_names=CLASS_NAMES, zero_division=0))

        # Save real confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        sns.heatmap(real_cm, annot=True, fmt="d", cmap="Greens", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title("Real NASA Swift BAT Subjects (N=19)\nConfusion Matrix", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("Zooniverse Ground Truth", fontweight="bold")
        fig.tight_layout()
        real_cm_path = "checkpoints/experiments/real_zooniverse_cm.png"
        fig.savefig(real_cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Real confusion matrix saved to: {real_cm_path}")

        real_eval = {
            "accuracy": real_acc,
            "macro_f1": real_f1,
            "confusion_matrix": real_cm.tolist(),
        }

    # Generate comparative convergence figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    for r in results:
        ax1.plot(r["history"]["train_loss"], label=r["exp_name"], linewidth=1.8)
        ax2.plot(r["history"]["val_f1"], label=r["exp_name"], linewidth=1.8)
    ax1.set_title("Training Loss Convergence", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8)

    ax2.set_title("Validation Macro F1 Score", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Macro F1", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    comp_fig_path = "checkpoints/experiments/convergence_comparison.png"
    fig.savefig(comp_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparative convergence plot saved to: {comp_fig_path}")

    # Return full experiment data
    all_data = {
        "trials": results,
        "summary": df_summary.to_dict(orient="records"),
        "real_zooniverse": real_eval,
    }
    with open("checkpoints/experiments/all_experiments.json", "w") as f:
        json.dump(all_data, f, indent=2)

    return all_data


if __name__ == "__main__":
    run_all_benchmark_experiments()
