# NASA Zooniverse "Burst Chaser" ML Classification Pipeline

An end-to-end deep learning pipeline to detect and classify physical features in gamma-ray burst (GRB) light-curve plots from the NASA Zooniverse **Burst Chaser** citizen science project (`amylien/burst-chaser`, Workflow ID: `25777`).

The model classifies light-curve features marked by citizen scientists into three distinct classes:
1. **`pulse`**: Real astrophysical emission from the GRB characterized by sudden count rate increases above the baseline.
2. **`noise`**: Statistical background fluctuations and instrumental variations.
3. **`unclear`**: Marginal or ambiguous features near the detection threshold ("It's hard to tell").

---

## Pipeline Architecture

```
gamma-ray-burst-detection/
├── data_loader.py          # Panoptes SDK fetcher, metadata parser, classification ingestion & mock generator
├── dataset.py              # PyTorch Dataset, red-marker ROI detection/cropping, augmentations, stratified splitting
├── model.py                # Transfer learning vision backbones (ResNet-18, ConvNeXt-Tiny) + MLP head
├── train.py                # Training loop with inverse class weighting, validation tracking, and checkpointing
├── predict.py              # CLI inference tool (single image & holdout test evaluation with confusion matrix)
├── requirements.txt        # Core dependencies
├── tests/
│   └── test_pipeline.py    # Complete test suite
├── data/
│   └── mock_burst_chaser/  # Generated/cached light curves and manifests
└── checkpoints/
    ├── best_model.pth      # Best checkpoint based on validation macro F1-score
    ├── holdout_test.csv    # Holdout test set partition
    ├── training_history.json
    └── confusion_matrix.png
```

---

## Key Features

1. **Zooniverse Panoptes SDK Integration**:
   - Programmatically connects to project `amylien/burst-chaser` (Workflow ID `25777`).
   - Downloads light-curve plot images and extracts GRB trigger IDs (e.g. `GRB111228A`), time-window coordinates, and ground-truth feedback answers (`#feedback_1_answer`).
   - Ingestion module parses volunteer classification exports (CSV or JSON lines) and aggregates volunteer votes into consensus labels.

2. **OpenCV Red-Marker ROI Cropping**:
   - Detects the red elliptical marker placed on the light curves using HSV dual-mask thresholding.
   - Extracts the bounding box with 25% contextual padding to isolate the feature under review.
   - Gracefully falls back to the full plot if no marker is present.

3. **Stratified Splitting**:
   - Splits data into 80% Train, 10% Validation, and 10% Test partitions.
   - Guarantees class representation across all splits even with severe class imbalance.

4. **Class Imbalance Mitigation**:
   - Cross-Entropy Loss with dynamic inverse class frequency weights:
     $$w_c = \frac{N}{K \cdot N_c}$$
   - Prevents the dominant pulse/noise classes from drowning out the rarer `unclear` class.

5. **Transfer Learning Vision Backbone**:
   - Pretrained ResNet-18 / ConvNeXt-Tiny with custom MLP classification head (`Linear` -> `BatchNorm1d` -> `ReLU` -> `Dropout` -> `Linear(3)`).
   - AdamW optimizer with cosine annealing learning rate scheduling.

---

## Quickstart Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3. Model Training
Train on synthetic mock light curves:
```bash
python train.py --use_mock --num_mock_samples 120 --epochs 10 --batch_size 16 --lr 2e-4
```
Or train with live Zooniverse subjects:
```bash
python train.py --fetch_zooniverse --max_subjects 50 --epochs 10
```

### 4. Single Image Prediction
Run inference on a local file:
```bash
python predict.py --image data/mock_burst_chaser/images/mock_100000.png
```
Or directly on a web image URL:
```bash
python predict.py --image "https://panoptes-uploads.zooniverse.org/subject_location/89f658d8-8007-4499-80f9-c016d94cb514.png"
```

### 5. Holdout Evaluation & Confusion Matrix
Evaluate performance on the holdout test set and generate `confusion_matrix.png`:
```bash
python predict.py --evaluate --test_csv checkpoints/holdout_test.csv
```
