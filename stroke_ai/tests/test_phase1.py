"""
tests/test_phase1.py
---------------------
Unit tests for Phase 1 video processing pipeline.

Run:
    cd stroke_ai
    python -m pytest tests/test_phase1.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.frame_extraction import (
    FrameExtractor,
    batch_process_videos,
    extract_frames,
    resize_frames,
    sample_frames,
)
from preprocessing.video_loader import (
    VideoLoader,
    VideoMetadata,
    get_video_metadata,
    load_video,
    validate_video,
)


def _create_synthetic_video(
    path: Path,
    width: int = 320,
    height: int = 240,
    fps: float = 30.0,
    n_frames: int = 90,
    codec: str = "MJPG",
) -> Path:
    """Write a short synthetic AVI to disk and return its path."""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    rng = np.random.default_rng(42)
    for _ in range(n_frames):
        frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class TestVideoLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.loader = VideoLoader()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_load_video_success(self) -> None:
        vpath = _create_synthetic_video(self.tmp / "test.avi")
        cap, meta = load_video(vpath)
        self.assertTrue(cap.isOpened())
        self.assertIsInstance(meta, VideoMetadata)
        self.assertEqual(meta.file_name, "test.avi")
        cap.release()

    def test_load_video_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_video(self.tmp / "ghost.mp4")

    def test_load_video_unsupported_extension(self) -> None:
        bad = self.tmp / "clip.xyz"
        bad.write_bytes(b"fake")
        with self.assertRaises(ValueError):
            load_video(bad)

    def test_get_video_metadata_fields(self) -> None:
        vpath = _create_synthetic_video(self.tmp / "meta.avi", fps=25.0, n_frames=50)
        meta = get_video_metadata(vpath)
        self.assertEqual(meta.width, 320)
        self.assertEqual(meta.height, 240)
        self.assertAlmostEqual(meta.fps, 25.0, delta=1.0)
        self.assertGreater(meta.duration_seconds, 0.0)
        self.assertGreater(meta.file_size_mb, 0.0)
        self.assertEqual(meta.format, ".avi")

    def test_validate_video_valid(self) -> None:
        vpath = _create_synthetic_video(self.tmp / "valid.avi", n_frames=60)
        _, meta = load_video(vpath)
        meta = validate_video(meta)
        self.assertTrue(meta.is_valid)
        self.assertEqual(meta.validation_errors, [])

    def test_validate_video_too_few_frames(self) -> None:
        vpath = _create_synthetic_video(self.tmp / "tiny.avi", n_frames=2)
        _, meta = load_video(vpath)
        meta.total_frames = 2
        meta = validate_video(meta)
        self.assertFalse(meta.is_valid)
        self.assertTrue(any("frames" in e for e in meta.validation_errors))

    def test_discover_videos_finds_avi(self) -> None:
        _create_synthetic_video(self.tmp / "a.avi")
        _create_synthetic_video(self.tmp / "b.avi")
        found = self.loader.discover_videos(self.tmp)
        names = [p.name for p in found]
        self.assertIn("a.avi", names)
        self.assertIn("b.avi", names)

    def test_batch_load_metadata(self) -> None:
        _create_synthetic_video(self.tmp / "v1.avi")
        _create_synthetic_video(self.tmp / "v2.avi")
        results = self.loader.batch_load_metadata(self.tmp)
        self.assertEqual(len(results), 2)


class TestFrameExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.extractor = FrameExtractor(output_base_dir=self.tmp / "processed")
        self.loader = VideoLoader()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_extract_frames_count(self) -> None:
        vpath = _create_synthetic_video(self.tmp / "ef.avi", fps=30.0, n_frames=90)
        cap, meta = load_video(vpath)
        frames = extract_frames(cap, meta)
        cap.release()
        self.assertGreater(len(frames), 0)
        self.assertLessEqual(len(frames), 90)

    def test_sample_frames_reduces(self) -> None:
        frames = [np.zeros((10, 10, 3), dtype=np.uint8)] * 200
        sampled = sample_frames(frames, max_frames=50)
        self.assertEqual(len(sampled), 50)

    def test_sample_frames_noop_when_small(self) -> None:
        frames = [np.zeros((10, 10, 3), dtype=np.uint8)] * 20
        sampled = sample_frames(frames, max_frames=50)
        self.assertEqual(len(sampled), 20)

    def test_resize_frames_correct_shape(self) -> None:
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        resized = resize_frames(frames, target_width=256, target_height=256)
        self.assertEqual(resized[0].shape, (256, 256, 3))

    def test_process_video_success(self) -> None:
        vpath = _create_synthetic_video(self.tmp / "proc.avi", n_frames=60)
        result = self.extractor.process_video(vpath)
        self.assertTrue(result.success, msg=result.error)
        self.assertGreater(result.frames_saved, 0)

    def test_process_video_frames_on_disk(self) -> None:
        vpath = _create_synthetic_video(self.tmp / "disk.avi", n_frames=60)
        result = self.extractor.process_video(vpath)
        self.assertTrue(result.success)
        saved_files = list(Path(result.output_dir).glob("frame_*"))
        self.assertEqual(len(saved_files), result.frames_saved)

    def test_batch_process_videos(self) -> None:
        for i in range(3):
            _create_synthetic_video(self.tmp / f"v{i}.avi", n_frames=60)
        results = batch_process_videos(
            self.tmp,
            output_base_dir=self.tmp / "processed",
        )
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.success for r in results))

    def test_get_batch_summary(self) -> None:
        for i in range(2):
            _create_synthetic_video(self.tmp / f"s{i}.avi", n_frames=60)
        results = self.extractor.batch_process_videos(self.tmp)
        summary = self.extractor.get_batch_summary(results)
        self.assertEqual(summary["total_videos"], 2)
        self.assertEqual(summary["successful"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
