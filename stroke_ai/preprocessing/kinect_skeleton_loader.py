"""
preprocessing/kinect_skeleton_loader.py
---------------------------------------
Parse Kinect v2 .skeleton files (3D joint positions).

File format
-----------
Line 1: total frame count
Per frame (28 lines):
  - num_bodies (1)
  - body metadata (10 values)
  - num_joints (25)
  - 25 joint lines x 12 values:
      x, y, z, depthX, depthY, colorX, colorY, qw, qx, qy, qz, tracking_state
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.label_loader import LabelLoader
from preprocessing.skeleton_sequence import SkeletonSequence
from utils.config_loader import load_config
from utils.joint_definitions import KINECT_V2_JOINTS
from utils.logger import get_logger

_cfg = load_config()
logger = get_logger(__name__, level=_cfg.logging.level)

LINES_PER_FRAME = 28
JOINTS_PER_BODY = 25
VALUES_PER_JOINT = 12


class KinectSkeletonLoader:
    """Load and parse Kinect .skeleton files into SkeletonSequence objects."""

    def __init__(self, label_loader: Optional[LabelLoader] = ...) -> None:
        if label_loader is ...:
            self._label_loader: Optional[LabelLoader] = LabelLoader()
        else:
            self._label_loader = label_loader

    def parse_skeleton_file(
        self,
        skeleton_path: str | Path,
        attach_labels: bool = True,
    ) -> SkeletonSequence:
        skeleton_path = Path(skeleton_path)
        if not skeleton_path.exists():
            raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")

        with open(skeleton_path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        if not lines:
            raise ValueError(f"Empty skeleton file: {skeleton_path}")

        declared_frames = int(float(lines[0]))
        frames_data: List[np.ndarray] = []
        frame_indices: List[int] = []

        idx = 1
        frame_num = 0
        while idx < len(lines):
            if frame_num >= declared_frames:
                break

            try:
                keypoints, consumed = self._parse_single_frame(lines, idx)
            except (ValueError, IndexError) as exc:
                logger.warning(
                    "Skipping corrupt frame %d in %s: %s",
                    frame_num,
                    skeleton_path.name,
                    exc,
                )
                break

            if keypoints is not None:
                frames_data.append(keypoints)
                frame_indices.append(frame_num)

            idx += consumed
            frame_num += 1

        if not frames_data:
            raise ValueError(f"No valid frames parsed from {skeleton_path}")

        keypoints_arr = np.stack(frames_data, axis=0)  # (T, 25, 3)
        label = None
        if attach_labels and self._label_loader is not None:
            try:
                label = self._label_loader.get_label_for_skeleton_file(skeleton_path)
            except FileNotFoundError:
                logger.warning("Performance CSV not found — skipping labels")

        stem = skeleton_path.stem
        parts = stem.split("_", 3)
        subject_id = parts[0] if parts else stem
        exercise = parts[3] if len(parts) >= 4 else "unknown"

        labels = label.to_dict() if label else {}
        metadata = {
            "file_path": str(skeleton_path.resolve()),
            "declared_frames": declared_frames,
            "parsed_frames": len(frames_data),
            "coordinate_system": "kinect_camera_3d",
            "units": "metres",
        }

        return SkeletonSequence(
            sequence_id=stem,
            source="kinect",
            motion_name=exercise,
            subject_id=subject_id,
            keypoints=keypoints_arr,
            joint_names=KINECT_V2_JOINTS.copy(),
            frame_indices=np.array(frame_indices, dtype=int),
            labels=labels,
            metadata=metadata,
        )

    def _parse_single_frame(
        self, lines: List[str], start: int
    ) -> Tuple[Optional[np.ndarray], int]:
        """Parse one frame starting at `start`. Returns (25,3) array and lines consumed."""
        if start + LINES_PER_FRAME > len(lines):
            # tolerate truncated last frame
            remaining = len(lines) - start
            if remaining < 3:
                raise ValueError("Incomplete frame header")

        pos = start
        _num_bodies = int(float(lines[pos]))
        pos += 1

        # body metadata line
        pos += 1

        num_joints = int(float(lines[pos]))
        pos += 1

        if num_joints != JOINTS_PER_BODY:
            raise ValueError(f"Expected {JOINTS_PER_BODY} joints, got {num_joints}")

        joints = np.full((JOINTS_PER_BODY, 3), np.nan, dtype=np.float64)

        for j in range(num_joints):
            if pos >= len(lines):
                raise ValueError(f"Missing joint data at joint {j}")
            values = lines[pos].split()
            pos += 1
            if len(values) < 3:
                continue
            x, y, z = float(values[0]), float(values[1]), float(values[2])
            joints[j] = [x, y, z]

        consumed = pos - start
        return joints, consumed

    def discover_skeleton_files(self, directory: str | Path) -> List[Path]:
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        files = sorted(directory.glob("*.skeleton"))
        logger.info("Discovered %d Kinect skeleton files in %s", len(files), directory)
        return files

    def batch_load(self, directory: str | Path) -> List[SkeletonSequence]:
        """Load all .skeleton files in a directory."""
        files = self.discover_skeleton_files(directory)
        sequences: List[SkeletonSequence] = []

        for i, fpath in enumerate(files, start=1):
            try:
                seq = self.parse_skeleton_file(fpath)
                sequences.append(seq)
                if i % 50 == 0:
                    logger.info("Kinect batch: loaded %d/%d", i, len(files))
            except Exception as exc:
                logger.error("Failed to load %s: %s", fpath.name, exc)

        logger.info("Kinect batch complete: %d/%d loaded", len(sequences), len(files))
        return sequences
