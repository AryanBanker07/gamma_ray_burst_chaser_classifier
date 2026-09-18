"""
PyTorch Dataset and preprocessing module for NASA Zooniverse Burst Chaser light-curve plots.

Features:
- OpenCV-based red elliptical marker detection and ROI cropping.
- Augmentations: 224x224 resize, horizontal flip, subtle brightness/contrast jitter, ImageNet normalization.
- PyTorch Dataset supporting image loading, caching, and batching.
- Stratified dataset splitting into train (80%), val (10%), and test (10%).
"""

import os
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from data_loader import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS


def detect_red_marker_roi(
    image_input: Union[Image.Image, np.ndarray, str],
    padding_ratio: float = 0.25,
    min_area: float = 40.0,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detects the red elliptical region-of-interest marker on a light-curve plot.
    Returns the padded bounding box (x_min, y_min, x_max, y_max) or None if no marker found.
    """
    # Load or convert to numpy BGR image for OpenCV
    if isinstance(image_input, str):
        bgr = cv2.imread(image_input)
        if bgr is None:
            return None
        height, width = bgr.shape[:2]
    elif isinstance(image_input, Image.Image):
        rgb = np.array(image_input)
        height, width = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            bgr = image_input.copy()
            height, width = bgr.shape[:2]
        else:
            return None
    else:
        return None

    # Convert to HSV color space for robust color isolation
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 1. Primary Marker: Red elliptical marker (wraps across 0/180 boundary)
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([165, 70, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_red = [c for c in contours_red if cv2.contourArea(c) >= min_area]

    if valid_red:
        best_contour = max(valid_red, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(best_contour)
    else:
        # 2. Secondary Marker: Blue shaded candidate interval (used in Pulse_shape & Combined_Fermi workflows)
        # Typically H in [90, 130], S in [20, 140], V in [160, 255]
        mask_blue = cv2.inRange(hsv, np.array([90, 20, 160]), np.array([135, 140, 255]))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_blue = [c for c in contours_blue if cv2.contourArea(c) >= 150.0]
        if not valid_blue:
            return None
        best_contour = max(valid_blue, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(best_contour)

    # Apply padding to capture context around the marked feature
    pad_w = int(bw * padding_ratio)
    pad_h = int(bh * padding_ratio)

    x_min = max(0, bx - pad_w)
    y_min = max(0, by - pad_h)
    x_max = min(width, bx + bw + pad_w)
    y_max = min(height, by + bh + pad_h)

    # Ensure box has non-zero dimension
    if (x_max - x_min < 10) or (y_max - y_min < 10):
        return None

    return (x_min, y_min, x_max, y_max)


def crop_roi_or_full(
    image: Image.Image,
    crop_roi: bool = True,
    padding_ratio: float = 0.25,
) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
    """
    Crops image to the detected red marker ROI if crop_roi is True and marker is found.
    Returns (processed_image, roi_bbox).
    """
    if not crop_roi:
        return image, None

    roi_box = detect_red_marker_roi(image, padding_ratio=padding_ratio)
    if roi_box is not None:
        x_min, y_min, x_max, y_max = roi_box
        cropped_img = image.crop((x_min, y_min, x_max, y_max))
        return cropped_img, roi_box

    return image, None


def get_transforms(is_train: bool = True, target_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Returns torchvision transforms for training or evaluation.
    Augmentations: 224x224 resize, random horizontal flip, subtle brightness/contrast shift, ImageNet normalization.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class BurstChaserDataset(Dataset):
    """
    PyTorch Dataset for Burst Chaser light-curve images.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        crop_roi: bool = True,
        is_train: bool = True,
        transform: Optional[transforms.Compose] = None,
        padding_ratio: float = 0.25,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.crop_roi = crop_roi
        self.is_train = is_train
        self.padding_ratio = padding_ratio
        self.transform = transform if transform is not None else get_transforms(is_train=is_train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = row.get("image_path") if "image_path" in row and pd.notna(row["image_path"]) else (row.get("local_image_path") or row.get("image_source"))

        # Load image
        if not img_path or not os.path.exists(str(img_path)):
            raise FileNotFoundError(f"Image path does not exist: {img_path}")

        image = Image.open(str(img_path)).convert("RGB")

        # Optional red marker ROI cropping
        cropped_image, roi_box = crop_roi_or_full(
            image,
            crop_roi=self.crop_roi,
            padding_ratio=self.padding_ratio,
        )

        # Apply augmentations and normalization
        tensor_img = self.transform(cropped_image)

        # Target label
        label_str = str(row.get("label") or row.get("verified_label") or "unknown")
        if "label_id" in row and pd.notna(row["label_id"]) and int(row["label_id"]) >= 0:
            label_id = int(row["label_id"])
        elif "verified_label_id" in row and pd.notna(row["verified_label_id"]) and int(row["verified_label_id"]) >= 0:
            label_id = int(row["verified_label_id"])
        elif label_str in CLASS_TO_IDX:
            label_id = CLASS_TO_IDX[label_str]
        else:
            label_id = -1

        return {
            "image": tensor_img,
            "label": torch.tensor(label_id, dtype=torch.long),
            "label_str": label_str,
            "subject_id": str(row.get("subject_id", "")),
            "grb_id": str(row.get("grb_id", "")),
            "roi_box": roi_box if roi_box is not None else (-1, -1, -1, -1),
            "image_path": str(img_path),
        }


class DualStreamBurstChaserDataset(Dataset):
    """
    Dataset yielding both the global full light-curve image and the local cropped ROI.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        is_train: bool = True,
        transform: Optional[transforms.Compose] = None,
        padding_ratio: float = 0.25,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.is_train = is_train
        self.padding_ratio = padding_ratio
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transforms(is_train=is_train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = row.get("image_path") if "image_path" in row and pd.notna(row["image_path"]) else (row.get("local_image_path") or row.get("image_source"))
        if not img_path or not os.path.exists(str(img_path)):
            raise FileNotFoundError(f"Image path does not exist: {img_path}")

        image = Image.open(str(img_path)).convert("RGB")

        # Global stream: full light curve
        tensor_global = self.transform(image)

        # Local stream: cropped red marker ROI
        cropped_image, roi_box = crop_roi_or_full(
            image,
            crop_roi=True,
            padding_ratio=self.padding_ratio,
        )
        tensor_local = self.transform(cropped_image)

        label_str = str(row.get("label") or row.get("verified_label") or "unknown")
        if "label_id" in row and pd.notna(row["label_id"]) and int(row["label_id"]) >= 0:
            label_id = int(row["label_id"])
        elif "verified_label_id" in row and pd.notna(row["verified_label_id"]) and int(row["verified_label_id"]) >= 0:
            label_id = int(row["verified_label_id"])
        elif label_str in CLASS_TO_IDX:
            label_id = CLASS_TO_IDX[label_str]
        else:
            label_id = -1

        return {
            "image_global": tensor_global,
            "image_local": tensor_local,
            "label": torch.tensor(label_id, dtype=torch.long),
            "label_str": label_str,
            "subject_id": str(row.get("subject_id", "")),
            "grb_id": str(row.get("grb_id", "")),
            "roi_box": roi_box if roi_box is not None else (-1, -1, -1, -1),
            "image_path": str(img_path),
        }


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratifies dataset into train (80%), validation (10%), and test (10%) splits
    preserving the class distribution. Robust to small sample sizes per class.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    df_copy = df.copy()

    # Normalize label_id column if coming from personal checking CSV
    if "label_id" not in df_copy.columns or df_copy["label_id"].isnull().any():
        if "verified_label_id" in df_copy.columns:
            df_copy["label_id"] = df_copy["verified_label_id"]
        elif "label" in df_copy.columns:
            df_copy["label_id"] = df_copy["label"].map(CLASS_TO_IDX)
        elif "verified_label" in df_copy.columns:
            df_copy["label_id"] = df_copy["verified_label"].map(CLASS_TO_IDX)

    if "image_path" not in df_copy.columns:
        if "image_source" in df_copy.columns:
            df_copy["image_path"] = df_copy["image_source"]
        elif "local_image_path" in df_copy.columns:
            df_copy["image_path"] = df_copy["local_image_path"]

    # Filter out rows with invalid label_id
    valid_df = df_copy[df_copy["label_id"].isin([0, 1, 2])].copy()
    if len(valid_df) < len(df):
        print(f"Filtered out {len(df) - len(valid_df)} records without valid 3-class labels.")

    rng = np.random.default_rng(random_state)
    train_indices = []
    val_indices = []
    test_indices = []

    # Allocate per class to strictly preserve proportions and guarantee no crashes
    for label_id in sorted(valid_df["label_id"].unique()):
        cls_indices = valid_df.index[valid_df["label_id"] == label_id].tolist()
        rng.shuffle(cls_indices)
        n = len(cls_indices)

        if n >= 10:
            n_train = int(np.round(n * train_ratio))
            n_val = int(np.round(n * val_ratio))
            # Ensure at least 1 in val and test
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            # Re-adjust train if needed
            n_train = n - n_val - n_test
        elif n >= 3:
            # For small counts, ensure at least 1 in train, 1 in val, 1 in test
            n_train = max(1, n - 2)
            n_val = 1
            n_test = n - n_train - n_val
        elif n == 2:
            n_train = 1
            n_val = 1
            n_test = 0
        else:
            n_train = 1
            n_val = 0
            n_test = 0

        train_indices.extend(cls_indices[:n_train])
        val_indices.extend(cls_indices[n_train:n_train + n_val])
        test_indices.extend(cls_indices[n_train + n_val:])

    train_df = valid_df.loc[train_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = valid_df.loc[val_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = valid_df.loc[test_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"Dataset split summary (Total: {len(valid_df)}):")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(valid_df)*100:.1f}%) | Classes: {dict(train_df['label'].value_counts())}")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(valid_df)*100:.1f}%) | Classes: {dict(val_df['label'].value_counts())}")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(valid_df)*100:.1f}%) | Classes: {dict(test_df['label'].value_counts())}")

    return train_df, val_df, test_df


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 16,
    crop_roi: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to construct PyTorch DataLoaders for train, val, and test splits.
    """
    train_dataset = BurstChaserDataset(train_df, crop_roi=crop_roi, is_train=True)
    val_dataset = BurstChaserDataset(val_df, crop_roi=crop_roi, is_train=False)
    test_dataset = BurstChaserDataset(test_df, crop_roi=crop_roi, is_train=False)

    eff_batch_size = max(2, min(batch_size, len(train_df))) if len(train_df) >= 2 else 1
    drop_last_train = (len(train_df) > eff_batch_size and len(train_df) % eff_batch_size == 1)

    train_loader = DataLoader(train_dataset, batch_size=eff_batch_size, shuffle=True, drop_last=drop_last_train, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=eff_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=eff_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test with mock dataset
    manifest_path = "data/mock_burst_chaser/mock_manifest.csv"
    if not os.path.exists(manifest_path):
        from data_loader import MockBurstChaserLoader
        loader = MockBurstChaserLoader(output_dir="data/mock_burst_chaser")
        df_mock = loader.generate_dataset(num_samples=40)
    else:
        df_mock = pd.read_csv(manifest_path)

    train_df, val_df, test_df = create_stratified_splits(df_mock, 0.8, 0.1, 0.1)
    train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df, batch_size=4)

    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}, labels: {batch['label']}")
