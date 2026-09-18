"""
Unit and integration test suite for the Burst Chaser ML pipeline.
"""

import os
import shutil
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

from data_loader import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    parse_grb_trigger_id,
    parse_time_window_coordinates,
    MockBurstChaserLoader,
)
from dataset import (
    detect_red_marker_roi,
    crop_roi_or_full,
    get_transforms,
    BurstChaserDataset,
    create_stratified_splits,
    build_dataloaders,
)
from model import BurstChaserClassifier
from train import compute_inverse_class_weights, run_training
from predict import load_model, predict_single_image, evaluate_holdout_test


class TestBurstChaserPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("data/test_scratch")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        # Generate a small set of mock light curves for tests
        cls.mock_loader = MockBurstChaserLoader(output_dir=str(cls.test_dir / "mock"), seed=123)
        cls.df_mock = cls.mock_loader.generate_dataset(num_samples=24)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_metadata_parsing(self):
        """Verify GRB trigger ID and time window coordinate extraction."""
        meta_url = {
            "BAT light curve": "https://swift.gsfc.nasa.gov/results/batgrbcat//GRB111228A/web/GRB111228A.html#lc",
            "#feedback_1_answer": "0",
        }
        trigger_id = parse_grb_trigger_id(meta_url)
        self.assertEqual(trigger_id, "GRB111228A")

        meta_explicit = {"trigger_id": "GRB200415A", "t_start": "10.5", "t_stop": "42.0"}
        self.assertEqual(parse_grb_trigger_id(meta_explicit), "GRB200415A")

        coords = parse_time_window_coordinates(meta_explicit)
        self.assertAlmostEqual(coords["t_start"], 10.5)
        self.assertAlmostEqual(coords["t_stop"], 42.0)
        self.assertAlmostEqual(coords["duration"], 31.5)

    def test_02_mock_generator(self):
        """Verify mock data generation and file persistence."""
        self.assertEqual(len(self.df_mock), 24)
        sample = self.df_mock.iloc[0]
        self.assertTrue(os.path.exists(sample["image_path"]))
        self.assertIn(sample["label"], CLASS_NAMES)
        self.assertEqual(CLASS_TO_IDX[sample["label"]], sample["label_id"])

    def test_03_red_marker_detection(self):
        """Verify OpenCV red elliptical ROI detection."""
        sample = self.df_mock.iloc[0]
        roi_box = detect_red_marker_roi(sample["image_path"], padding_ratio=0.25)
        self.assertIsNotNone(roi_box, "Red marker ellipse should be detected on mock light curve")
        x_min, y_min, x_max, y_max = roi_box
        self.assertTrue(x_max > x_min)
        self.assertTrue(y_max > y_min)

        # Test cropping
        img = Image.open(sample["image_path"])
        cropped, bbox = crop_roi_or_full(img, crop_roi=True)
        self.assertIsNotNone(bbox)
        self.assertEqual(cropped.size, (x_max - x_min, y_max - y_min))

    def test_04_dataset_and_dataloaders(self):
        """Verify PyTorch Dataset and DataLoader tensor outputs."""
        dataset = BurstChaserDataset(self.df_mock, crop_roi=True, is_train=True)
        self.assertEqual(len(dataset), len(self.df_mock))

        item = dataset[0]
        self.assertEqual(item["image"].shape, torch.Size([3, 224, 224]))
        self.assertIsInstance(item["label"], torch.Tensor)
        self.assertIn(item["label"].item(), [0, 1, 2])

    def test_05_stratified_splits(self):
        """Verify stratified splitting into train, val, and test partitions."""
        train_df, val_df, test_df = create_stratified_splits(
            self.df_mock, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42
        )
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(self.df_mock))
        # Ensure each class has representation in train
        train_classes = set(train_df["label_id"].unique())
        self.assertEqual(train_classes, {0, 1, 2})

    def test_06_inverse_class_weights(self):
        """Verify inverse class weights computation."""
        # Simulated imbalanced labels: 70 pulse, 20 noise, 10 unclear
        mock_labels = np.array([0] * 70 + [1] * 20 + [2] * 10)
        weights = compute_inverse_class_weights(mock_labels, num_classes=3)
        self.assertEqual(weights.shape, (3,))
        # Least frequent class (2: unclear) should receive highest weight
        self.assertTrue(weights[2] > weights[1] > weights[0])

    def test_07_model_architecture_and_forward(self):
        """Verify ResNet-18 classifier forward pass and backward gradients."""
        model = BurstChaserClassifier(backbone_name="resnet18", num_classes=3, pretrained=False)
        dummy_input = torch.randn(4, 3, 224, 224)
        logits = model(dummy_input)
        self.assertEqual(logits.shape, (4, 3))

        targets = torch.tensor([0, 1, 2, 0])
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, targets)
        loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for parameter {name}")
                break

    def test_08_end_to_end_train_and_predict(self):
        """Run a mini 1-epoch training loop and verify checkpointing and CLI inference."""
        ckpt_dir = self.test_dir / "checkpoints"
        result = run_training(
            data_csv=str(self.test_dir / "mock" / "mock_manifest.csv"),
            backbone="resnet18",
            epochs=1,
            batch_size=8,
            lr=1e-3,
            checkpoint_dir=str(ckpt_dir),
            device_name="cpu",
        )

        best_ckpt = Path(result["checkpoint_path"])
        self.assertTrue(best_ckpt.exists(), "best_model.pth should be created")
        holdout_csv = Path(result["holdout_test_path"])
        self.assertTrue(holdout_csv.exists(), "holdout_test.csv should be created")

        # Test single-image prediction
        model, _ = load_model(str(best_ckpt), device=torch.device("cpu"))
        sample_img = self.df_mock.iloc[0]["image_path"]
        pred_res = predict_single_image(sample_img, model, device=torch.device("cpu"), crop_roi=True)

        self.assertIn(pred_res["predicted_label"], CLASS_NAMES)
        self.assertTrue(0.0 <= pred_res["confidence"] <= 1.0)
        prob_sum = sum(pred_res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

        # Test holdout test evaluation and confusion matrix generation
        cm_path = str(self.test_dir / "test_cm.png")
        eval_res = evaluate_holdout_test(
            test_csv=str(holdout_csv),
            model=model,
            device=torch.device("cpu"),
            crop_roi=True,
            output_cm_path=cm_path,
        )
        self.assertTrue(os.path.exists(cm_path), "confusion_matrix.png should be generated")
        self.assertIn("accuracy", eval_res)
        self.assertIn("macro_f1", eval_res)


if __name__ == "__main__":
    unittest.main()
