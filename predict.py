"""
Inference, Evaluation, and Live Zooniverse Application CLI Tool
for NASA Burst Chaser Light-Curve Classification (Workflow ID: 25777).

Capabilities:
1. Single Subject Inference (Local file or live Zooniverse image URL):
   python predict.py --image "https://panoptes-uploads.zooniverse.org/..." --dual_stream --strict_unclear

2. Batch Workflow Manifest Scoring (CSV from ZooniverseBurstChaserLoader):
   python predict.py --manifest data/zooniverse/subjects_manifest.csv --dual_stream --strict_unclear --output_csv predictions.csv

3. Holdout Test Set Evaluation & Confusion Matrix Export:
   python predict.py --evaluate --test_csv checkpoints/holdout_test.csv --dual_stream
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from data_loader import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS
from dataset import crop_roi_or_full, get_transforms, BurstChaserDataset, DualStreamBurstChaserDataset
from model import BurstChaserClassifier, DualStreamBurstChaserClassifier


def load_model(
    checkpoint_path: Optional[str] = None,
    use_dual_stream: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, str, Dict[str, Any]]:
    """
    Loads trained Burst Chaser model (single-stream or dual-stream) and metadata.
    Automatically detects architecture if not explicitly specified.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = "checkpoints/dual_stream_best.pth" if use_dual_stream else "checkpoints/best_model.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file '{checkpoint_path}' not found. "
            "Please verify training completion or specify --checkpoint."
        )

    raw_checkpoint = torch.load(checkpoint_path, map_location=device)

    # Detect architecture
    is_dual = use_dual_stream or "dual_stream" in checkpoint_path or (
        isinstance(raw_checkpoint, dict) and any(k.startswith("global_stream") for k in raw_checkpoint.keys())
    )

    if is_dual:
        model = DualStreamBurstChaserClassifier(num_classes=3, pretrained=False)
        state_dict = raw_checkpoint if isinstance(raw_checkpoint, dict) and "model_state_dict" not in raw_checkpoint else raw_checkpoint.get("model_state_dict", raw_checkpoint)
        model.load_state_dict(state_dict)
        model_type = "dual_stream"
        metadata = {"backbone": "resnet18_dual_stream", "architecture": "DualStream (Global + Local)"}
    else:
        state_dict = raw_checkpoint.get("model_state_dict", raw_checkpoint)
        backbone = raw_checkpoint.get("backbone", "resnet18") if isinstance(raw_checkpoint, dict) else "resnet18"
        model = BurstChaserClassifier(backbone_name=backbone, num_classes=3, pretrained=False)
        model.load_state_dict(state_dict)
        model_type = "single_stream"
        metadata = raw_checkpoint if isinstance(raw_checkpoint, dict) else {}

    model.to(device)
    model.eval()
    metadata["model_type"] = model_type
    return model, metadata


def apply_decision_threshold(
    probs: np.ndarray,
    strict_unclear: bool = False,
    unclear_threshold: float = 0.50,
) -> Tuple[int, str, float]:
    """
    Applies calibrated decision logic across class probabilities.
    When strict_unclear is enabled:
    - If P(unclear) >= unclear_threshold -> classified as 'unclear'
    - Otherwise -> classified as argmax between 'pulse' and 'noise'
    """
    if strict_unclear:
        prob_unclear = float(probs[2])
        if prob_unclear >= unclear_threshold:
            pred_idx = 2
        else:
            pred_idx = 0 if probs[0] >= probs[1] else 1
    else:
        pred_idx = int(np.argmax(probs))

    pred_label = IDX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_idx, pred_label, confidence


def predict_single_image(
    image_path_or_url: str,
    model: torch.nn.Module,
    model_type: Optional[str] = None,
    device: Optional[torch.device] = None,
    crop_roi: bool = True,
    padding_ratio: float = 0.25,
    strict_unclear: bool = False,
    unclear_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Runs prediction on a single light-curve image (local path or live web URL).
    Returns class prediction, confidence, probabilities, and ROI detection coordinates.
    """
    if model_type is None:
        model_type = "dual_stream" if isinstance(model, DualStreamBurstChaserClassifier) else "single_stream"

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    # 1. Fetch image from URL if necessary
    temp_downloaded = None
    if image_path_or_url.startswith(("http://", "https://")):
        temp_downloaded = "temp_predict_image.png"
        urllib.request.urlretrieve(image_path_or_url, temp_downloaded)
        local_path = temp_downloaded
    else:
        local_path = image_path_or_url

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Image not found at path: {local_path}")

    # 2. Preprocess image
    orig_img = Image.open(local_path).convert("RGB")
    transform = get_transforms(is_train=False)

    with torch.no_grad():
        if model_type == "dual_stream":
            tensor_global = transform(orig_img).unsqueeze(0).to(device)
            cropped_img, roi_bbox = crop_roi_or_full(orig_img, crop_roi=True, padding_ratio=padding_ratio)
            tensor_local = transform(cropped_img).unsqueeze(0).to(device)
            logits = model(tensor_global, tensor_local)
        else:
            processed_img, roi_bbox = crop_roi_or_full(orig_img, crop_roi=crop_roi, padding_ratio=padding_ratio)
            tensor_img = transform(processed_img).unsqueeze(0).to(device)
            logits = model(tensor_img)

        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # 3. Decision rule
    pred_idx, pred_label, confidence = apply_decision_threshold(
        probs, strict_unclear=strict_unclear, unclear_threshold=unclear_threshold
    )

    # Clean up temporary download
    if temp_downloaded and os.path.exists(temp_downloaded):
        try:
            os.remove(temp_downloaded)
        except Exception:
            pass

    return {
        "image_source": image_path_or_url,
        "predicted_label": pred_label,
        "predicted_idx": pred_idx,
        "confidence": confidence,
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "roi_box": roi_bbox,
        "cropped": roi_bbox is not None,
        "strict_unclear_applied": strict_unclear,
    }


def predict_manifest(
    manifest_csv: str,
    model: torch.nn.Module,
    model_type: str,
    device: torch.device,
    output_csv: Optional[str] = None,
    strict_unclear: bool = False,
    unclear_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Processes a batch CSV manifest containing subjects from Workflow 25777.
    Outputs predictions, posterior probabilities, and saves enriched CSV.
    """
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    print(f"\nProcessing batch manifest ({len(df)} subjects) with {model_type} architecture...")

    results = []
    for idx, row in df.iterrows():
        img_target = None
        for col in ["image_path", "local_image_path", "image_url"]:
            if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                val = str(row[col]).strip()
                if os.path.exists(val) or val.startswith(("http://", "https://")):
                    img_target = val
                    break

        if not img_target:
            print(f"Skipping row {idx}: No valid local image path or URL found.")
            continue

        try:
            res = predict_single_image(
                image_path_or_url=img_target,
                model=model,
                model_type=model_type,
                device=device,
                strict_unclear=strict_unclear,
                unclear_threshold=unclear_threshold,
            )
            subj_id = row.get("subject_id", idx)
            grb_id = row.get("grb_id", "UNKNOWN")
            ground_truth = row.get("label", "unknown")

            results.append({
                "subject_id": subj_id,
                "grb_id": grb_id,
                "ground_truth": ground_truth,
                "predicted_label": res["predicted_label"],
                "confidence": res["confidence"],
                "prob_pulse": res["probabilities"]["pulse"],
                "prob_noise": res["probabilities"]["noise"],
                "prob_unclear": res["probabilities"]["unclear"],
                "roi_detected": res["cropped"],
                "image_source": img_target,
            })
        except Exception as e:
            print(f"Error processing subject {row.get('subject_id', idx)}: {e}")

    df_out = pd.DataFrame(results)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df_out.to_csv(output_csv, index=False)
        print(f"\nBatch predictions successfully exported to: {output_csv}")

    # Summary report
    print("\n" + "=" * 65)
    print(f"WORKFLOW 25777 BATCH TRIAGE SUMMARY (N = {len(df_out)})")
    print("=" * 65)
    counts = df_out["predicted_label"].value_counts()
    for cname in CLASS_NAMES:
        cnt = counts.get(cname, 0)
        pct = (cnt / len(df_out) * 100) if len(df_out) > 0 else 0
        print(f"  {cname.upper().ljust(10)}: {cnt:3d} subjects ({pct:5.1f}%)")
    print("=" * 65)

    if "ground_truth" in df_out and (df_out["ground_truth"] != "unknown").any():
        valid_mask = df_out["ground_truth"].isin(CLASS_NAMES)
        if valid_mask.sum() > 0:
            y_t = df_out.loc[valid_mask, "ground_truth"].values
            y_p = df_out.loc[valid_mask, "predicted_label"].values
            acc = accuracy_score(y_t, y_p)
            f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
            print(f"Known Label Validation (N = {valid_mask.sum()}): Accuracy = {acc*100:.2f}%, Macro F1 = {f1:.4f}")

    return df_out


def evaluate_holdout_test(
    test_csv: str,
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    crop_roi: bool = True,
    output_cm_path: str = "checkpoints/confusion_matrix.png",
    model_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs full evaluation on the holdout test set and plots a confusion matrix.
    """
    if model_type is None:
        model_type = "dual_stream" if isinstance(model, DualStreamBurstChaserClassifier) else "single_stream"

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test dataset CSV not found: {test_csv}")

    df_test = pd.read_csv(test_csv)
    print(f"\nEvaluating on holdout test set ({len(df_test)} samples) from: {test_csv}")

    if model_type == "dual_stream":
        dataset = DualStreamBurstChaserDataset(df_test, is_train=False)
    else:
        dataset = BurstChaserDataset(df_test, crop_roi=crop_roi, is_train=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    all_targets = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if model_type == "dual_stream":
                xg = batch["image_global"].to(device)
                xl = batch["image_local"].to(device)
                logits = model(xg, xl)
            else:
                images = batch["image"].to(device)
                logits = model(images)

            targets = batch["label"].to(device)
            preds = torch.argmax(logits, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    report_str = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)

    print("\n" + "=" * 65)
    print("HOLDOUT TEST SET CLASSIFICATION REPORT")
    print("=" * 65)
    print(report_str)
    print(f"Overall Accuracy: {acc*100:.2f}% | Macro F1-Score: {macro_f1:.4f}")
    print("=" * 65)

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-9)
    annot = np.empty_like(cm).astype(str)
    for i in range(3):
        for j in range(3):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=True, ax=ax, linewidths=1.5, linecolor="#f1f5f9")
    ax.set_title(f"Burst Chaser Light-Curve Classification\nHoldout Test Confusion Matrix ({model_type.upper()})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold", labelpad=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_cm_path) or ".", exist_ok=True)
    fig.savefig(output_cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nConfusion matrix plot saved to: {output_cm_path}")

    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm.tolist()}


def parse_args():
    parser = argparse.ArgumentParser(description="Burst Chaser Prediction & Evaluation CLI")
    parser.add_argument("--image", type=str, default=None, help="Local image path or live web URL for single prediction")
    parser.add_argument("--manifest", type=str, default=None, help="Path to Zooniverse subjects manifest CSV for batch triage")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on holdout test set")
    parser.add_argument("--test_csv", type=str, default="checkpoints/holdout_test.csv", help="Path to holdout test CSV")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path (default: auto-detected)")
    parser.add_argument("--dual_stream", action="store_true", help="Use dual-stream (global + local) vision architecture")
    parser.add_argument("--strict_unclear", action="store_true", help="Apply calibrated strict threshold (tau_unclear >= 0.50)")
    parser.add_argument("--unclear_threshold", type=float, default=0.50, help="Probability threshold for unclear class (default: 0.50)")
    parser.add_argument("--output_csv", type=str, default="data/zooniverse/predictions.csv", help="Path to save batch predictions")
    parser.add_argument("--output_cm", type=str, default="checkpoints/confusion_matrix.png", help="Path to save confusion matrix")
    parser.add_argument("--no_crop", action="store_true", help="Disable red elliptical ROI cropping (single-stream only)")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image and not args.manifest and not args.evaluate:
        print("NASA Burst Chaser Prediction CLI (Workflow ID: 25777)")
        print("-" * 60)
        print("Usage Examples:")
        print("1. Single image prediction (URL or local file):")
        print("   python predict.py --image 'https://panoptes-uploads.zooniverse.org/...' --dual_stream --strict_unclear")
        print("2. Batch manifest triage:")
        print("   python predict.py --manifest data/zooniverse/subjects_manifest.csv --dual_stream --strict_unclear")
        print("3. Holdout evaluation:")
        print("   python predict.py --evaluate --dual_stream")
        sys.exit(1)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, meta = load_model(
        checkpoint_path=args.checkpoint,
        use_dual_stream=args.dual_stream,
        device=device,
    )
    model_type = meta.get("model_type", "single_stream")
    print(f"Loaded {model_type.upper()} model onto {device}.")

    # 1. Single Image Inference
    if args.image:
        res = predict_single_image(
            image_path_or_url=args.image,
            model=model,
            model_type=model_type,
            device=device,
            crop_roi=not args.no_crop,
            strict_unclear=args.strict_unclear,
            unclear_threshold=args.unclear_threshold,
        )

        print("\n" + "=" * 58)
        print("BURST CHASER PREDICTION RESULT (WORKFLOW 25777)")
        print("=" * 58)
        print(f"Image Source:     {res['image_source']}")
        print(f"Architecture:     {model_type.upper()}")
        print(f"Decision Rule:    {'Strict (tau_unclear >= ' + str(args.unclear_threshold) + ')' if args.strict_unclear else 'Standard Argmax'}")
        print(f"Predicted Class:  [{res['predicted_label'].upper()}]")
        print(f"Confidence:       {res['confidence']*100:.2f}%")
        print(f"ROI Detected:     {'Yes, Bounding Box: ' + str(res['roi_box']) if res['cropped'] else 'No red marker found (Used full image)'}")
        print("\nClass Probabilities:")
        for cls_name in CLASS_NAMES:
            p_val = res['probabilities'][cls_name]
            bar_len = int(p_val * 30)
            bar = "#" * bar_len + "-" * (30 - bar_len)
            print(f"  {cls_name.ljust(8)} : {p_val:.4f} ({p_val*100:5.1f}%) |{bar}|")
        print("=" * 58 + "\n")

    # 2. Batch Manifest Prediction
    if args.manifest:
        predict_manifest(
            manifest_csv=args.manifest,
            model=model,
            model_type=model_type,
            device=device,
            output_csv=args.output_csv,
            strict_unclear=args.strict_unclear,
            unclear_threshold=args.unclear_threshold,
        )

    # 3. Holdout Test Set Evaluation
    if args.evaluate:
        evaluate_holdout_test(
            test_csv=args.test_csv,
            model=model,
            device=device,
            crop_roi=not args.no_crop,
            output_cm_path=args.output_cm,
            model_type=model_type,
        )


if __name__ == "__main__":
    main()
