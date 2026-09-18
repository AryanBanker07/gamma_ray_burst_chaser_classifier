"""
Data loader and ingestion module for NASA Zooniverse 'Burst Chaser' project.
Target Project: amylien/burst-chaser (Workflow ID: 25777)

Handles:
- Panoptes SDK integration for fetching subjects and downloading light-curve plots.
- Metadata parsing (GRB trigger IDs, time-window coordinates).
- Ingestion and vote aggregation from exported volunteer classifications.
- Synthetic mock light-curve generator for offline local training and testing.
"""

import os
import re
import json
import urllib.request
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Canonical class mappings
CLASS_NAMES = ["pulse", "noise", "unclear"]
CLASS_TO_IDX = {"pulse": 0, "noise": 1, "unclear": 2}
IDX_TO_CLASS = {0: "pulse", 1: "noise", 2: "unclear"}

# Mapping from Zooniverse answer strings / feedback codes to canonical labels
LABEL_MAPPING = {
    "0": "pulse",
    0: "pulse",
    "This is a pulse.": "pulse",
    "pulse": "pulse",
    "1": "noise",
    1: "noise",
    "This is noise.": "noise",
    "noise": "noise",
    "2": "unclear",
    2: "unclear",
    "It's hard to tell.": "unclear",
    "unclear": "unclear",
}


def parse_grb_trigger_id(metadata: Dict[str, Any]) -> str:
    """
    Extract GRB Trigger ID from metadata fields or URLs.
    Example: 'https://swift.gsfc.nasa.gov/results/batgrbcat//GRB111228A/web/...' -> 'GRB111228A'
    """
    # 1. Check explicit fields
    for field in ["trigger_id", "grb_id", "grb_name", "#trigger_id", "trigger"]:
        if field in metadata and metadata[field]:
            return str(metadata[field]).strip()

    # 2. Check BAT light curve URL or similar URLs
    for key, val in metadata.items():
        if isinstance(val, str):
            match = re.search(r"(GRB\s*\d{6}[A-Z]?)", val, re.IGNORECASE)
            if match:
                return match.group(1).upper().replace(" ", "")

    # 3. Check filename
    filename = metadata.get("Filename", "") or metadata.get("filename", "")
    match = re.search(r"(GRB\s*\d{6}[A-Z]?)", str(filename), re.IGNORECASE)
    if match:
        return match.group(1).upper().replace(" ", "")

    return "GRB_UNKNOWN"


def parse_time_window_coordinates(metadata: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Parse time-window coordinates from subject metadata.
    Returns dict with keys: t_start, t_stop, duration.
    """
    coords = {"t_start": None, "t_stop": None, "duration": None}

    for key, val in metadata.items():
        k_lower = key.lower()
        if "t_start" in k_lower or "tmin" in k_lower or "start_time" in k_lower:
            try:
                coords["t_start"] = float(val)
            except (ValueError, TypeError):
                pass
        elif "t_stop" in k_lower or "tmax" in k_lower or "stop_time" in k_lower:
            try:
                coords["t_stop"] = float(val)
            except (ValueError, TypeError):
                pass

    if coords["t_start"] is not None and coords["t_stop"] is not None:
        coords["duration"] = coords["t_stop"] - coords["t_start"]

    return coords


def extract_label_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    """
    Extract ground-truth label from feedback metadata if available.
    In Burst Chaser practice workflow: '#feedback_1_answer': '0' -> pulse.
    """
    for key in ["#feedback_1_answer", "feedback_1_answer", "answer", "label", "consensus_label"]:
        if key in metadata:
            val = str(metadata[key]).strip()
            if val in LABEL_MAPPING:
                return LABEL_MAPPING[val]
    return None


class ZooniverseBurstChaserLoader:
    """
    Fetcher for Burst Chaser subjects and images using panoptes-client.
    """

    def __init__(
        self,
        project_slug: str = "amylien/burst-chaser",
        workflow_id: int = 25777,
        subject_set_ids: Optional[List[Union[int, str]]] = None,
        cache_dir: str = "data/zooniverse",
    ):
        self.project_slug = project_slug
        self.workflow_id = workflow_id
        # Subject sets in Burst Chaser:
        # 118003: 'Pulse_shape' (1,649 Swift-BAT light-curve subjects)
        # 137715: 'Combined_Fermi' (2,632 Fermi GBM light-curve subjects)
        # 117958: 'Pulse_vs_noise' (19 Practice subjects)
        self.subject_set_ids = [str(s) for s in subject_set_ids] if subject_set_ids else ["118003", "137715", "117958"]
        self.cache_dir = Path(cache_dir)
        self.images_dir = self.cache_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def fetch_subjects(
        self,
        max_subjects: Optional[int] = 100,
        subject_set_id: Optional[Union[int, str]] = None,
        max_workers: int = 8,
    ) -> pd.DataFrame:
        """
        Fetches authentic NASA Burst Chaser subjects directly from Zooniverse public APIs.
        Supports parallel image downloading and metadata parsing across thousands of available candidates.
        """
        from concurrent.futures import ThreadPoolExecutor

        target_sets = [str(subject_set_id)] if subject_set_id else self.subject_set_ids
        manifest_path = self.cache_dir / "subjects_manifest.csv"

        existing_df = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        existing_ids = set(existing_df["subject_id"].astype(str)) if not existing_df.empty and "subject_id" in existing_df.columns else set()

        records = []
        target_count = max_subjects if max_subjects is not None else 1000

        print(f"Fetching real NASA Burst Chaser subjects across sets {target_sets} (Target: {target_count})...")

        for s_set in target_sets:
            if len(records) >= target_count:
                break
            page = 1
            while len(records) < target_count:
                url = f"https://www.zooniverse.org/api/subjects?subject_set_id={s_set}&page={page}&page_size=50"
                try:
                    req = urllib.request.Request(url, headers={"Accept": "application/vnd.api+json; version=1"})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                except Exception as err:
                    print(f"Error querying subject set {s_set} page {page}: {err}")
                    break

                batch = res.get("subjects", [])
                if not batch:
                    break

                for subj in batch:
                    subj_id = str(subj.get("id"))
                    meta = subj.get("metadata", {}) or {}
                    grb_id = parse_grb_trigger_id(meta)
                    time_coords = parse_time_window_coordinates(meta)
                    label = extract_label_from_metadata(meta)

                    image_url = None
                    locations = subj.get("locations", [])
                    if locations and len(locations) > 0:
                        loc = locations[0]
                        if isinstance(loc, dict):
                            for mime, u in loc.items():
                                if "image" in mime or u.endswith((".png", ".jpg", ".jpeg")):
                                    image_url = u
                                    break
                        elif isinstance(loc, str):
                            image_url = loc

                    records.append({
                        "subject_id": subj_id,
                        "grb_id": grb_id,
                        "image_url": image_url,
                        "t_start": time_coords["t_start"],
                        "t_stop": time_coords["t_stop"],
                        "label": label,
                        "label_id": CLASS_TO_IDX.get(label, -1) if label else -1,
                        "subject_set_id": str(s_set),
                        "raw_metadata": json.dumps(meta),
                    })

                    if len(records) >= target_count:
                        break

                page += 1

        print(f"Retrieved metadata for {len(records)} candidates. Downloading images in parallel...")

        # Multi-threaded image download
        def _download_candidate_image(rec):
            s_id = rec["subject_id"]
            img_url = rec.get("image_url")
            if not img_url:
                rec["image_path"] = None
                return rec

            ext = os.path.splitext(img_url.split("?")[0])[1] or ".png"
            dest = self.images_dir / f"{s_id}{ext}"
            if not (dest.exists() and dest.stat().st_size > 0):
                try:
                    urllib.request.urlretrieve(img_url, dest)
                except Exception as dl_err:
                    print(f"Failed to download subject {s_id}: {dl_err}")
                    rec["image_path"] = None
                    return rec

            rec["image_path"] = str(dest)
            return rec

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            downloaded_records = list(executor.map(_download_candidate_image, records))

        new_df = pd.DataFrame(downloaded_records)
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["subject_id"]).reset_index(drop=True)
        else:
            combined_df = new_df

        combined_df.to_csv(manifest_path, index=False)
        print(f"Successfully synchronized {len(combined_df)} real Burst Chaser candidates to: {manifest_path}")
        return combined_df


class ClassificationIngestionModule:
    """
    Ingests volunteer classification exports and computes consensus labels per subject.
    """

    def __init__(self, workflow_id: int = 25777):
        self.workflow_id = workflow_id

    def parse_classification_export(
        self,
        export_file_path: str,
        consensus_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Parses Zooniverse classifications export (CSV or JSON lines) and aggregates votes.
        """
        export_path = Path(export_file_path)
        if not export_path.exists():
            raise FileNotFoundError(f"Classification export file not found: {export_file_path}")

        if export_path.suffix.lower() == ".csv":
            df_raw = pd.read_csv(export_path, low_memory=False)
        else:
            df_raw = pd.read_json(export_path, lines=True)

        if "workflow_id" in df_raw.columns:
            df_wf = df_raw[df_raw["workflow_id"] == self.workflow_id].copy()
        else:
            df_wf = df_raw.copy()

        print(f"Processing {len(df_wf)} classifications for workflow {self.workflow_id}...")

        votes_by_subject: Dict[str, List[str]] = {}

        for _, row in df_wf.iterrows():
            subj_id_raw = row.get("subject_ids") or row.get("subject_id")
            if pd.isna(subj_id_raw):
                continue
            subj_id = str(subj_id_raw).strip()

            # Parse annotations
            annot_str = row.get("annotations", "")
            try:
                if isinstance(annot_str, str):
                    annots = json.loads(annot_str)
                else:
                    annots = annot_str
            except Exception:
                annots = []

            vote_label = None
            if isinstance(annots, list):
                for a in annots:
                    task = a.get("task", "")
                    val = a.get("value")
                    if task in ["T0", "init", "question"] or val is not None:
                        if val in LABEL_MAPPING:
                            vote_label = LABEL_MAPPING[val]
                            break
                        elif isinstance(val, int) and str(val) in LABEL_MAPPING:
                            vote_label = LABEL_MAPPING[str(val)]
                            break

            if vote_label:
                votes_by_subject.setdefault(subj_id, []).append(vote_label)

        # Aggregate votes
        summary = []
        for s_id, votes in votes_by_subject.items():
            total = len(votes)
            counts = {cls: votes.count(cls) for cls in CLASS_NAMES}
            max_cls = max(counts, key=counts.get)
            max_frac = counts[max_cls] / total if total > 0 else 0.0

            # Label assignment based on consensus
            final_label = max_cls if max_frac >= consensus_threshold else "unclear"

            summary.append({
                "subject_id": s_id,
                "total_votes": total,
                "pulse_votes": counts["pulse"],
                "noise_votes": counts["noise"],
                "unclear_votes": counts["unclear"],
                "consensus_label": final_label,
                "label_id": CLASS_TO_IDX[final_label],
                "confidence": max_frac,
            })

        df_agg = pd.DataFrame(summary)
        return df_agg


class MockBurstChaserLoader:
    """
    Generates synthetic Burst Chaser light-curve plots and volunteer classifications
    for local training, unit testing, and benchmarking.
    """

    def __init__(self, output_dir: str = "data/mock_burst_chaser", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

    def generate_synthetic_light_curve(
        self,
        subject_id: str,
        label: str,
        grb_id: str,
    ) -> Path:
        """
        Renders a realistic synthetic GRB light-curve plot matching Burst Chaser styling:
        - Time (s) vs Counts/s
        - Poisson/Gaussian background noise fluctuations
        - Blue shaded computer-estimated GRB duration interval
        - Red elliptical region of interest (ROI) marker targeting feature
        """
        time = np.linspace(-20, 100, 600)
        baseline = 50.0 + 3.0 * np.sin(time / 20.0)
        noise = self.rng.normal(0, 4.0, size=len(time))
        counts = baseline + noise

        # Computer burst interval (blue band)
        burst_start = 5.0 + self.rng.uniform(-3, 3)
        burst_stop = 45.0 + self.rng.uniform(-3, 5)

        # Region of interest center and marker parameters
        if label == "pulse":
            # Distinct pulse with Norris-like fast rise exponential decay
            t_peak = self.rng.uniform(burst_start + 5, burst_stop - 10)
            amplitude = self.rng.uniform(35.0, 75.0)
            tau1 = self.rng.uniform(0.5, 2.0)
            tau2 = self.rng.uniform(2.0, 6.0)
            pulse_signal = np.where(
                time >= (t_peak - 2 * tau1),
                amplitude * np.exp(-np.abs(time - t_peak) / np.where(time < t_peak, tau1, tau2)),
                0.0,
            )
            counts += pulse_signal
            roi_center_x = t_peak
            roi_center_y = float(baseline[np.argmin(np.abs(time - t_peak))] + amplitude * 0.5)
            roi_width = self.rng.uniform(12.0, 18.0)
            roi_height = amplitude * 1.3
        elif label == "noise":
            # Pure noise fluctuation inside the ROI
            roi_center_x = self.rng.uniform(50.0, 85.0)
            roi_center_y = float(baseline[np.argmin(np.abs(time - roi_center_x))])
            roi_width = self.rng.uniform(10.0, 16.0)
            roi_height = self.rng.uniform(16.0, 24.0)
        else:  # unclear
            # Ambiguous low SNR bump (~1.5-2.2 sigma)
            roi_center_x = self.rng.uniform(burst_start + 2, burst_stop - 2)
            amplitude = self.rng.uniform(7.0, 12.0)
            tau = self.rng.uniform(1.5, 3.5)
            counts += amplitude * np.exp(-0.5 * ((time - roi_center_x) / tau) ** 2)
            roi_center_y = float(baseline[np.argmin(np.abs(time - roi_center_x))] + amplitude * 0.4)
            roi_width = self.rng.uniform(11.0, 17.0)
            roi_height = self.rng.uniform(18.0, 28.0)

        # Render light curve plot
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=100)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f9fafb")

        # Blue shaded interval
        ax.axvspan(burst_start, burst_stop, color="#dbeafe", alpha=0.7, label="Algorithmic Burst Region")

        # Plot count rate with error-bar like stepped light curve
        ax.step(time, counts, where="mid", color="#1e3a8a", linewidth=1.2, label=f"BAT Light Curve ({grb_id})")
        ax.axhline(50.0, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.8)

        # Draw red elliptical marker around the ROI
        ellipse = patches.Ellipse(
            (roi_center_x, roi_center_y),
            width=roi_width,
            height=roi_height,
            fill=False,
            edgecolor="#dc2626",
            linewidth=2.6,
            linestyle="-",
            zorder=10,
        )
        ax.add_patch(ellipse)

        ax.set_xlim(-20, 100)
        y_min = max(0, float(np.min(counts) - 10))
        y_max = float(np.max(counts) + 15)
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel("Time since trigger (seconds)", fontsize=11, fontweight="bold", color="#1f2937")
        ax.set_ylabel("Count rate (counts / s)", fontsize=11, fontweight="bold", color="#1f2937")
        ax.set_title(f"Burst Chaser - Trigger: {grb_id}", fontsize=12, fontweight="bold", color="#111827")
        ax.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

        img_path = self.images_dir / f"{subject_id}.png"
        fig.tight_layout()
        fig.savefig(img_path, dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        return img_path

    def generate_dataset(
        self,
        num_samples: int = 120,
        class_distribution: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Generates a synthetic dataset with realistic class imbalance.
        Default class distribution: 50% pulse, 35% noise, 15% unclear.
        """
        if class_distribution is None:
            class_distribution = {"pulse": 0.50, "noise": 0.35, "unclear": 0.15}

        classes = list(class_distribution.keys())
        probs = [class_distribution[c] for c in classes]
        probs = np.array(probs) / np.sum(probs)

        records = []
        print(f"Generating {num_samples} synthetic Burst Chaser light-curve plots...")

        for i in range(num_samples):
            subj_id = f"mock_{100000 + i}"
            label = self.rng.choice(classes, p=probs)
            trigger_num = 120000 + (i % 80)
            suffix = chr(65 + (i % 4))
            grb_id = f"GRB{trigger_num}{suffix}"

            img_path = self.generate_synthetic_light_curve(subj_id, label, grb_id)

            records.append({
                "subject_id": subj_id,
                "image_path": str(img_path),
                "image_url": None,
                "grb_id": grb_id,
                "t_start": float(self.rng.uniform(0, 10)),
                "t_stop": float(self.rng.uniform(40, 60)),
                "label": label,
                "label_id": CLASS_TO_IDX[label],
                "raw_metadata": json.dumps({"trigger_id": grb_id, "mock": True}),
            })

        df = pd.DataFrame(records)
        manifest_path = self.output_dir / "mock_manifest.csv"
        df.to_csv(manifest_path, index=False)
        print(f"Mock dataset created: {len(df)} samples saved to {manifest_path}")
        print("Class distribution:\n", df["label"].value_counts())
        return df


if __name__ == "__main__":
    loader = MockBurstChaserLoader(output_dir="data/mock_burst_chaser")
    df_mock = loader.generate_dataset(num_samples=30)
    print("Sample record:\n", df_mock.iloc[0])
