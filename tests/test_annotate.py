"""
Unit and Integration Tests for the Active-Learning Annotation Server and Candidate Manager.
"""

import os
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import torch

from annotate import CandidateManager, AnnotationRequestHandler


class TestAnnotationPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create dummy manifest CSV
        self.manifest_csv = self.temp_path / "test_manifest.csv"
        df = pd.DataFrame([
            {"subject_id": "test_001", "grb_id": "GRB111", "image_path": "dummy1.png"},
            {"subject_id": "test_002", "grb_id": "GRB222", "image_path": "dummy2.png"},
            {"subject_id": "test_003", "grb_id": "GRB333", "image_path": "dummy3.png"},
        ])
        df.to_csv(self.manifest_csv, index=False)

        self.output_csv = self.temp_path / "annotated.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_candidate_manager_queue_and_annotation(self):
        manager = CandidateManager(
            manifest_path=str(self.manifest_csv),
            images_dir=None,
            output_csv=str(self.output_csv),
        )

        self.assertEqual(len(manager.candidates), 3)
        self.assertEqual(manager.stats["total_verified"], 0)

        # Get first candidate
        cand1 = manager.get_next_candidate()
        self.assertIsNotNone(cand1)
        self.assertEqual(cand1["subject_id"], "test_001")

        # Record verification for candidate 1
        stats = manager.record_annotation(
            subject_id=cand1["subject_id"],
            grb_id=cand1["grb_id"],
            image_source=cand1["image_source"],
            model_guess="pulse",
            model_confidence=0.88,
            verified_label="pulse",
            is_agreement=True,
        )

        self.assertEqual(stats["total_verified"], 1)
        self.assertEqual(stats["pulses"], 1)
        self.assertEqual(stats["agreements"], 1)
        self.assertTrue(self.output_csv.exists())

        # Next candidate should now be test_002
        cand2 = manager.get_next_candidate()
        self.assertIsNotNone(cand2)
        self.assertEqual(cand2["subject_id"], "test_002")

        # Record override for candidate 2
        stats = manager.record_annotation(
            subject_id=cand2["subject_id"],
            grb_id=cand2["grb_id"],
            image_source=cand2["image_source"],
            model_guess="pulse",
            model_confidence=0.65,
            verified_label="noise",
            is_agreement=False,
        )

        self.assertEqual(stats["total_verified"], 2)
        self.assertEqual(stats["pulses"], 1)
        self.assertEqual(stats["noise"], 1)
        self.assertEqual(stats["agreements"], 1)

    def test_02_resume_existing_annotations(self):
        # Write 1 existing record to output CSV
        df_existing = pd.DataFrame([{
            "subject_id": "test_001",
            "grb_id": "GRB111",
            "image_source": "dummy1.png",
            "model_guess": "pulse",
            "model_confidence": 0.9,
            "verified_label": "pulse",
            "verified_label_id": 0,
            "is_agreement": True,
            "timestamp": "2026-09-18 12:00:00"
        }])
        df_existing.to_csv(self.output_csv, index=False)

        manager = CandidateManager(
            manifest_path=str(self.manifest_csv),
            images_dir=None,
            output_csv=str(self.output_csv),
        )

        # test_001 should be skipped, so next candidate is test_002
        self.assertEqual(manager.stats["total_verified"], 1)
    def test_03_http_api_endpoints(self):
        import threading
        import urllib.request
        import json
        from http.server import HTTPServer

        manager = CandidateManager(
            manifest_path=str(self.manifest_csv),
            images_dir=None,
            output_csv=str(self.output_csv),
        )

        class DummyModel:
            def __call__(self, *args, **kwargs):
                return None

        AnnotationRequestHandler.manager = manager
        AnnotationRequestHandler.model = DummyModel()
        AnnotationRequestHandler.model_type = "single_stream"
        AnnotationRequestHandler.device = torch.device("cpu")

        server = HTTPServer(("127.0.0.1", 8998), AnnotationRequestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            # 1. Test GET /
            req = urllib.request.Request("http://127.0.0.1:8998/")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode("utf-8")
                self.assertIn("NASA Burst Chaser", html)

            # 2. Test GET /api/stats
            req = urllib.request.Request("http://127.0.0.1:8998/api/stats")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["status"], "ok")
                self.assertEqual(data["stats"]["total_verified"], 0)

            # 3. Test POST /api/annotate
            payload = json.dumps({
                "subject_id": "test_001",
                "grb_id": "GRB111",
                "image_source": "dummy1.png",
                "model_guess": "pulse",
                "model_confidence": 0.85,
                "verified_label": "pulse",
                "is_agreement": True,
            }).encode("utf-8")

            req = urllib.request.Request(
                "http://127.0.0.1:8998/api/annotate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                res_data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(res_data["status"], "ok")
                self.assertEqual(res_data["stats"]["total_verified"], 1)
                self.assertEqual(res_data["stats"]["pulses"], 1)

        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
