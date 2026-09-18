"""
Interactive Active-Learning Annotation & Training Data Expansion Server
for NASA Burst Chaser (Workflow ID: 25777).

Features:
- Web-based interface served via Python's native http.server (zero extra dependencies).
- Real-time model inference using DualStreamBurstChaserClassifier or ResNet-18.
- 1-button and 1-keystroke decision mechanism:
    [1 / Space] : YES (Confirm Model Hypothesis)
    [2]         : Option 2 (Alternative Class A)
    [3]         : Option 3 (Alternative Class B)
    [S]         : Skip current subject
- Automatic queue management and persistence into data/annotated_training_data.csv.
- Synthetic mock candidate generation on demand to expand training sets indefinitely.
"""

import os
import sys
import json
import csv
import time
import argparse
import webbrowser
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional, Set
import pandas as pd
import torch

from data_loader import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, MockBurstChaserLoader
from model import DualStreamBurstChaserClassifier, BurstChaserClassifier
from predict import load_model, predict_single_image


class CandidateManager:
    """
    Manages unannotated light-curve subject queue, tracks verified annotations,
    and handles dynamic candidate generation.
    """

    def __init__(
        self,
        manifest_path: Optional[str] = "data/zooniverse/subjects_manifest.csv",
        images_dir: Optional[str] = "data/zooniverse/images",
        output_csv: str = "data/annotated_training_data.csv",
        auto_replenish: bool = False,
    ):
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.images_dir = Path(images_dir) if images_dir else None
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.auto_replenish = auto_replenish

        self.candidates: List[Dict[str, Any]] = []
        self.annotated_ids: Set[str] = set()
        self.stats = {
            "total_verified": 0,
            "pulses": 0,
            "noise": 0,
            "unclear": 0,
            "agreements": 0,
        }

        self.load_existing_annotations()
        self.load_candidates()

    def load_existing_annotations(self):
        """Loads previously saved annotations from output CSV to prevent duplicate work."""
        if not self.output_csv.exists():
            return

        try:
            df = pd.read_csv(self.output_csv)
            for _, row in df.iterrows():
                subj_id = str(row.get("subject_id", "")).strip()
                if subj_id:
                    self.annotated_ids.add(subj_id)
                label = str(row.get("verified_label", "")).lower()
                if label in self.stats:
                    self.stats[label] += 1
                elif label == "pulse":
                    self.stats["pulses"] += 1
                elif label == "noise":
                    self.stats["noise"] += 1
                elif label == "unclear":
                    self.stats["unclear"] += 1

                if row.get("is_agreement", False) in [True, "True", "true", 1]:
                    self.stats["agreements"] += 1

            self.stats["total_verified"] = len(df)
            print(f"Loaded {len(self.annotated_ids)} existing annotations from {self.output_csv}")
        except Exception as e:
            print(f"Warning: Could not read existing annotations: {e}")

    def load_candidates(self):
        """Populates candidate queue from manifest, image directory, or mock generator."""
        # 1. Load from manifest CSV if present
        if self.manifest_path and self.manifest_path.exists():
            try:
                df = pd.read_csv(self.manifest_path)
                for _, row in df.iterrows():
                    subj_id = str(row.get("subject_id", "")).strip()
                    img_path = row.get("image_path") or row.get("local_image_path")
                    img_url = row.get("image_url")
                    grb_id = str(row.get("grb_id", "GRB_UNKNOWN"))

                    target_src = str(img_path) if img_path and pd.notna(img_path) else (str(img_url) if img_url and pd.notna(img_url) else "")
                    if not target_src:
                        continue

                    self.candidates.append({
                        "subject_id": subj_id,
                        "grb_id": grb_id,
                        "image_source": target_src,
                    })
            except Exception as e:
                print(f"Warning reading manifest: {e}")

        # 2. Add any local real Zooniverse images in images_dir not in manifest
        if self.images_dir and self.images_dir.exists():
            for img_file in self.images_dir.glob("*.png"):
                subj_id = img_file.stem
                if not any(c["subject_id"] == subj_id for c in self.candidates):
                    self.candidates.append({
                        "subject_id": subj_id,
                        "grb_id": "GRB_ARCHIVE",
                        "image_source": str(img_file),
                    })

        # 3. If unannotated candidate pool is low, automatically fetch real candidates from Zooniverse if enabled
        unannotated = [c for c in self.candidates if c["subject_id"] not in self.annotated_ids]
        if self.auto_replenish and len(unannotated) < 10:
            try:
                print(f"Candidate queue has only {len(unannotated)} unannotated subjects; fetching fresh real Burst Chaser candidates from Zooniverse...")
                from data_loader import ZooniverseBurstChaserLoader
                loader = ZooniverseBurstChaserLoader()
                df = loader.fetch_subjects(max_subjects=50)
                for _, row in df.iterrows():
                    subj_id = str(row.get("subject_id", "")).strip()
                    if not any(c["subject_id"] == subj_id for c in self.candidates):
                        target_src = str(row.get("image_path") or row.get("image_url") or "")
                        if target_src:
                            self.candidates.append({
                                "subject_id": subj_id,
                                "grb_id": str(row.get("grb_id", "GRB_REAL")),
                                "image_source": target_src,
                            })
            except Exception as err:
                print(f"Notice during live subject replenishment: {err}")

        print(f"Candidate queue loaded: {len(self.candidates)} total real NASA Burst Chaser subjects available.")

    def get_next_candidate(self) -> Optional[Dict[str, Any]]:
        """Returns the next candidate subject that has not yet been annotated."""
        for cand in self.candidates:
            if cand["subject_id"] not in self.annotated_ids:
                return cand

        # Auto-replenish from Zooniverse if candidate queue is exhausted
        if self.auto_replenish:
            try:
                print("Candidate queue exhausted; fetching next batch of real NASA Burst Chaser subjects from Zooniverse...")
                from data_loader import ZooniverseBurstChaserLoader
                loader = ZooniverseBurstChaserLoader()
                df = loader.fetch_subjects(max_subjects=50)
                for _, row in df.iterrows():
                    subj_id = str(row.get("subject_id", "")).strip()
                    if subj_id not in self.annotated_ids and not any(c["subject_id"] == subj_id for c in self.candidates):
                        target_src = str(row.get("image_path") or row.get("image_url") or "")
                        if target_src:
                            new_cand = {
                                "subject_id": subj_id,
                                "grb_id": str(row.get("grb_id", "GRB_REAL")),
                                "image_source": target_src,
                            }
                            self.candidates.append(new_cand)
                            return new_cand
            except Exception as err:
                print(f"Replenishment error: {err}")

        return None

    def record_annotation(
        self,
        subject_id: str,
        grb_id: str,
        image_source: str,
        model_guess: str,
        model_confidence: float,
        verified_label: str,
        is_agreement: bool,
    ) -> Dict[str, Any]:
        """Appends verified annotation to output CSV and updates session metrics."""
        verified_label = verified_label.lower().strip()
        verified_id = CLASS_TO_IDX.get(verified_label, -1)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "subject_id": subject_id,
            "grb_id": grb_id,
            "image_path": image_source,
            "image_source": image_source,
            "label": verified_label,
            "label_id": verified_id,
            "verified_label": verified_label,
            "verified_label_id": verified_id,
            "model_guess": model_guess,
            "model_confidence": round(float(model_confidence), 4),
            "is_agreement": bool(is_agreement),
            "timestamp": timestamp,
        }

        # Write to CSV
        file_exists = self.output_csv.exists()
        with open(self.output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

        self.annotated_ids.add(str(subject_id))

        # Update stats
        self.stats["total_verified"] += 1
        if verified_label == "pulse":
            self.stats["pulses"] += 1
        elif verified_label == "noise":
            self.stats["noise"] += 1
        elif verified_label == "unclear":
            self.stats["unclear"] += 1

        if is_agreement:
            self.stats["agreements"] += 1

        return self.stats

    def export_combined_dataset(
        self,
        base_csv: str = "data/zooniverse/subjects_manifest.csv",
        output_combined_csv: str = "data/combined_training_dataset.csv",
    ) -> pd.DataFrame:
        """
        Combines personal annotations with a base dataset, prioritizing verified labels.
        Produces a unified training/testing manifest ready for train.py or train_dual_stream.py.
        """
        dfs = []
        if os.path.exists(base_csv):
            df_base = pd.read_csv(base_csv)
            dfs.append(df_base)
        if self.output_csv.exists():
            df_ann = pd.read_csv(self.output_csv)
            dfs.append(df_ann)

        if len(dfs) == 0:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        # Deduplicate on subject_id, keeping the last (verified annotation overrides base)
        if "subject_id" in combined.columns:
            combined = combined.drop_duplicates(subset=["subject_id"], keep="last")

        # Guarantee standard ML schema columns
        if "label" not in combined.columns and "verified_label" in combined.columns:
            combined["label"] = combined["verified_label"]
        if "label_id" not in combined.columns and "verified_label_id" in combined.columns:
            combined["label_id"] = combined["verified_label_id"]
        if "image_path" not in combined.columns and "image_source" in combined.columns:
            combined["image_path"] = combined["image_source"]

        out_path = Path(output_combined_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        print(f"Exported combined training dataset ({len(combined)} samples) to: {output_combined_csv}")
        return combined

    def generate_more_mock(self, count: int = 15) -> int:
        """Generates additional synthetic candidate light curves and enqueues them."""
        mock_loader = MockBurstChaserLoader(output_dir="data/mock_burst_chaser")
        df_new = mock_loader.generate_dataset(num_samples=count)
        added = 0
        for _, row in df_new.iterrows():
            subj_id = str(row["subject_id"])
            if subj_id not in self.annotated_ids and not any(c["subject_id"] == subj_id for c in self.candidates):
                self.candidates.append({
                    "subject_id": subj_id,
                    "grb_id": str(row["grb_id"]),
                    "image_source": str(row["image_path"]),
                })
                added += 1
        return added


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler serving web UI and active-learning REST API.
    """

    model = None
    model_type = "dual_stream"
    device = torch.device("cpu")
    manager: CandidateManager = None
    web_dir = Path("web")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Web UI Root
        if path in ["/", "/index.html"]:
            html_path = self.web_dir / "index.html"
            if not html_path.exists():
                self.send_error(404, "Web interface index.html not found.")
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())
            return

        # 2. Next Unannotated Subject API
        if path == "/api/next":
            cand = self.manager.get_next_candidate()
            if not cand:
                self._send_json({"status": "exhausted", "message": "All subjects annotated."})
                return

            # Run model prediction
            try:
                pred = predict_single_image(
                    image_path_or_url=cand["image_source"],
                    model=self.model,
                    model_type=self.model_type,
                    device=self.device,
                )
            except Exception as e:
                print(f"Prediction error on {cand['image_source']}: {e}")
                pred = {
                    "predicted_label": "pulse",
                    "confidence": 0.50,
                    "probabilities": {"pulse": 0.50, "noise": 0.25, "unclear": 0.25},
                    "cropped": False,
                }

            # Prepare image display URL
            img_src = cand["image_source"]
            if img_src.startswith(("http://", "https://")):
                display_url = img_src
            else:
                display_url = f"/image?path={urllib.parse.quote(img_src)}"

            response_data = {
                "status": "ok",
                "subject": {
                    "subject_id": cand["subject_id"],
                    "grb_id": cand["grb_id"],
                    "image_source": img_src,
                    "image_display_url": display_url,
                    "model_guess": pred["predicted_label"],
                    "confidence": float(pred["confidence"]),
                    "probabilities": pred["probabilities"],
                    "roi_detected": bool(pred["cropped"]),
                }
            }
            self._send_json(response_data)
            return

        # 3. Session Stats API
        if path == "/api/stats":
            self._send_json({"status": "ok", "stats": self.manager.stats})
            return

        # 4. Stream Local Image
        if path == "/image":
            raw_path = query.get("path", [None])[0]
            if not raw_path or not os.path.exists(raw_path):
                self.send_error(404, "Image file not found.")
                return

            # Basic security check
            norm_path = os.path.abspath(raw_path)
            cwd = os.path.abspath(os.getcwd())
            if not norm_path.startswith(cwd):
                self.send_error(403, "Access to file path is restricted.")
                return

            ext = os.path.splitext(norm_path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"

            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with open(norm_path, "rb") as f:
                self.wfile.write(f.read())
            return

        self.send_error(404, "Endpoint not found.")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. Record Annotation API
        if path == "/api/annotate":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(body)
                stats = self.manager.record_annotation(
                    subject_id=data.get("subject_id"),
                    grb_id=data.get("grb_id", "GRB_UNKNOWN"),
                    image_source=data.get("image_source", ""),
                    model_guess=data.get("model_guess", ""),
                    model_confidence=data.get("model_confidence", 0.0),
                    verified_label=data.get("verified_label", "pulse"),
                    is_agreement=data.get("is_agreement", False),
                )
                self._send_json({"status": "ok", "stats": stats})
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, code=400)
            return

        # 2. Generate More Candidates API
        if path == "/api/generate_candidates":
            content_len = int(self.headers.get("Content-Length", 0))
            count = 15
            if content_len > 0:
                try:
                    data = json.loads(self.rfile.read(content_len).decode("utf-8"))
                    count = int(data.get("count", 15))
                except Exception:
                    pass

            added = self.manager.generate_more_mock(count=count)
            self._send_json({"status": "ok", "added": added})
            return

        self.send_error(404, "Endpoint not found.")

    def _send_json(self, data: Dict[str, Any], code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        # Silence routine HTTP access logging for clean terminal
        pass


def run_annotation_server(
    port: int = 8080,
    manifest: Optional[str] = "data/zooniverse/subjects_manifest.csv",
    output: str = "data/annotated_training_data.csv",
    checkpoint: Optional[str] = None,
    dual_stream: bool = True,
    no_browser: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    print("Initializing model for real-time hypothesis generation...")
    ckpt_target = checkpoint or ("checkpoints/dual_stream_best.pth" if dual_stream else "checkpoints/best_model.pth")
    model, meta = load_model(checkpoint_path=ckpt_target, use_dual_stream=dual_stream, device=device)
    model_type = meta.get("model_type", "dual_stream" if dual_stream else "single_stream")

    # Initialize Candidate Manager with auto-replenishment enabled
    manager = CandidateManager(
        manifest_path=manifest,
        images_dir="data/zooniverse/images",
        output_csv=output,
        auto_replenish=True,
    )

    # Configure Handler
    AnnotationRequestHandler.model = model
    AnnotationRequestHandler.model_type = model_type
    AnnotationRequestHandler.device = device
    AnnotationRequestHandler.manager = manager

    # Start Server
    server_address = ("127.0.0.1", port)
    try:
        httpd = HTTPServer(server_address, AnnotationRequestHandler)
    except OSError:
        # Fallback to alternate port if occupied
        port = port + 1
        server_address = ("127.0.0.1", port)
        httpd = HTTPServer(server_address, AnnotationRequestHandler)

    url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 65)
    print("NASA BURST CHASER — ACTIVE ANNOTATION & DATA EXPANSION SERVER")
    print("=" * 65)
    print(f"  Web Interface URL:  {url}")
    print(f"  Model Loaded:       {model_type.upper()} ({device})")
    print(f"  Candidate Queue:    {len(manager.candidates)} subjects")
    print(f"  Verified Storage:   {output} ({manager.stats['total_verified']} verified)")
    print("  Controls:")
    print("    [Space / Enter] : YES (Go Ahead / Confirm Model Hypothesis)")
    print("    [1]             : Option 1 (PULSE)")
    print("    [2]             : Option 2 (NOISE)")
    print("    [3]             : Option 3 (UNSURE)")
    print("    [S]             : Skip Subject")
    print("  Press Ctrl+C in terminal to stop server.\n")

    if not no_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation server...")
        httpd.server_close()
        print("Server shutdown complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="NASA Burst Chaser Active Annotation Hub")
    parser.add_argument("--port", type=int, default=8080, help="Port to run web server on (default: 8080)")
    parser.add_argument("--manifest", type=str, default="data/zooniverse/subjects_manifest.csv", help="Subject manifest path")
    parser.add_argument("--output", type=str, default="data/annotated_training_data.csv", help="CSV path to save verified annotations")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint to generate hypotheses")
    parser.add_argument("--single_stream", action="store_true", help="Use single-stream model instead of dual-stream")
    parser.add_argument("--no_browser", action="store_true", help="Do not automatically open browser on startup")
    parser.add_argument("--export_combined", action="store_true", help="Combine verified annotations with manifest into combined dataset")
    parser.add_argument("--combined_output", type=str, default="data/combined_training_dataset.csv", help="Path to save combined dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.export_combined:
        manager = CandidateManager(manifest_path=args.manifest, output_csv=args.output)
        manager.export_combined_dataset(base_csv=args.manifest, output_combined_csv=args.combined_output)
        sys.exit(0)

    run_annotation_server(
        port=args.port,
        manifest=args.manifest,
        output=args.output,
        checkpoint=args.checkpoint,
        dual_stream=not args.single_stream,
        no_browser=args.no_browser,
    )
