"""
tests/test_skeleton_phase1.py
-----------------------------
Unit tests for skeleton-based Phase 1 ingestion.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.kinect_skeleton_loader import KinectSkeletonLoader
from preprocessing.openpose_json_loader import OpenPoseJsonLoader
from preprocessing.skeleton_pipeline import SkeletonPipeline
from preprocessing.skeleton_sequence import SkeletonSequence


def _write_synthetic_kinect_skeleton(path: Path, n_frames: int = 3) -> None:
    """Write a minimal valid .skeleton file."""
    lines = [str(n_frames)]
    for _ in range(n_frames):
        lines.append("1")
        lines.append("0 nan nan nan nan nan nan nan nan nan")
        lines.append("25")
        for j in range(25):
            lines.append(f"{j * 0.01} {j * 0.02} {1.5 + j * 0.001} 0 0 0 0 0 0 0 0 2")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_synthetic_openpose_json(path: Path, frame_idx: int = 0) -> None:
    kp = []
    for j in range(25):
        kp.extend([100.0 + j, 200.0 + j, 0.9])
    data = {
        "version": 1.3,
        "people": [{"person_id": [-1], "pose_keypoints_2d": kp}],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


class TestKinectLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_parse_synthetic_skeleton(self) -> None:
        skel = self.root / "P1_Female_21_Exercise Type 1.skeleton"
        _write_synthetic_kinect_skeleton(skel, n_frames=5)

        loader = KinectSkeletonLoader(label_loader=None)
        seq = loader.parse_skeleton_file(skel, attach_labels=False)

        self.assertEqual(seq.source, "kinect")
        self.assertEqual(seq.num_frames, 5)
        self.assertEqual(seq.num_joints, 25)
        self.assertEqual(seq.num_dims, 3)
        self.assertEqual(seq.motion_name, "Exercise Type 1")

    def test_discover_skeleton_files(self) -> None:
        _write_synthetic_kinect_skeleton(self.root / "a.skeleton")
        _write_synthetic_kinect_skeleton(self.root / "b.skeleton")
        loader = KinectSkeletonLoader()
        files = loader.discover_skeleton_files(self.root)
        self.assertEqual(len(files), 2)


class TestOpenPoseLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_parse_json_file(self) -> None:
        fpath = self.root / "p01s01_c2_000000000000_keypoints.json"
        _write_synthetic_openpose_json(fpath)
        loader = OpenPoseJsonLoader()
        arr = loader.parse_json_file(fpath)
        self.assertEqual(arr.shape, (25, 3))

    def test_load_session(self) -> None:
        for i in range(5):
            fpath = self.root / f"p01s01_c2_{i:012d}_keypoints.json"
            _write_synthetic_openpose_json(fpath, frame_idx=i)

        loader = OpenPoseJsonLoader()
        sessions = loader.discover_sessions(self.root)
        self.assertEqual(len(sessions), 1)

        files = sessions["p01s01_c2"]
        seq = loader.load_session(files, motion_name="C_Front_Away", session_id="p01s01_c2")
        self.assertEqual(seq.num_frames, 5)
        self.assertEqual(seq.source, "openpose")


class TestSkeletonPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "processed"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_and_ingest(self) -> None:
        seq = SkeletonSequence(
            sequence_id="test_seq",
            source="openpose",
            motion_name="test_motion",
            subject_id="p01",
            keypoints=np.random.randn(20, 25, 3).astype(np.float64),
            joint_names=[f"j{i}" for i in range(25)],
            frame_indices=np.arange(20),
            labels={"performance_outcome": 85.0},
        )

        pipeline = SkeletonPipeline(output_dir=self.out)
        result = pipeline.ingest_sequence(seq)
        self.assertTrue(result.success)
        self.assertTrue(Path(result.output_path).exists())

        meta = Path(result.output_path).parent / "metadata.json"
        self.assertTrue(meta.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
