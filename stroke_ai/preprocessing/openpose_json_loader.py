"""
preprocessing/openpose_json_loader.py
-------------------------------------
Load OpenPose JSON keypoint sequences (2D: x, y, confidence).

Filenames: p01s01_c2_000000000134_keypoints.json
Groups by session prefix (p01s01_c2) and sorts by frame index.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.skeleton_sequence import SkeletonSequence
from utils.config_loader import load_config
from utils.joint_definitions import OPENPOSE_BODY25_JOINTS
from utils.logger import get_logger

_cfg = load_config()
logger = get_logger(__name__, level=_cfg.logging.level)

_FILENAME_RE = re.compile(
    r"^(?P<subject>p\d+s\d+)_(?P<camera>c\d+)_(?P<frame>\d+)_keypoints\.json$",
    re.IGNORECASE,
)


class OpenPoseJsonLoader:
    """Load OpenPose JSON frame folders into SkeletonSequence objects."""

    def __init__(self, min_confidence: Optional[float] = None) -> None:
        scfg = _cfg.skeleton_ingestion
        self._min_confidence = (
            float(min_confidence)
            if min_confidence is not None
            else float(scfg.min_confidence)
        )

    def parse_json_file(self, json_path: str | Path) -> np.ndarray:
        """
        Parse one JSON file -> (25, 3) array [x, y, confidence].
        """
        json_path = Path(json_path)
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        people = data.get("people", [])
        if not people:
            return np.zeros((25, 3), dtype=np.float64)

        kp = people[0].get("pose_keypoints_2d", [])
        arr = np.array(kp, dtype=np.float64).reshape(-1, 3)

        if arr.shape[0] < 25:
            padded = np.zeros((25, 3), dtype=np.float64)
            padded[: arr.shape[0]] = arr
            arr = padded
        elif arr.shape[0] > 25:
            arr = arr[:25]

        # Mask low-confidence joints
        low = arr[:, 2] < self._min_confidence
        arr[low, 0] = 0.0
        arr[low, 1] = 0.0

        return arr

    def _parse_filename(self, path: Path) -> Tuple[str, str, int]:
        m = _FILENAME_RE.match(path.name)
        if not m:
            raise ValueError(f"Unexpected JSON filename: {path.name}")
        session = f"{m.group('subject')}_{m.group('camera')}"
        frame_idx = int(m.group("frame"))
        subject = m.group("subject")
        return session, subject, frame_idx

    def discover_sessions(self, json_dir: str | Path) -> Dict[str, List[Path]]:
        """Group JSON files by session prefix."""
        json_dir = Path(json_dir)
        if not json_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {json_dir}")

        sessions: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)

        for fpath in sorted(json_dir.glob("*_keypoints.json")):
            try:
                session, _, frame_idx = self._parse_filename(fpath)
                sessions[session].append((frame_idx, fpath))
            except ValueError as exc:
                logger.warning("Skipping %s: %s", fpath.name, exc)

        result: Dict[str, List[Path]] = {}
        for session, items in sessions.items():
            items.sort(key=lambda x: x[0])
            result[session] = [p for _, p in items]

        logger.info(
            "Discovered %d OpenPose session(s) in %s", len(result), json_dir
        )
        return result

    def load_session(
        self,
        json_files: List[Path],
        motion_name: str,
        session_id: str,
    ) -> SkeletonSequence:
        """Load an ordered list of JSON files into one SkeletonSequence."""
        if not json_files:
            raise ValueError("No JSON files provided")

        frames: List[np.ndarray] = []
        frame_indices: List[int] = []

        for fpath in json_files:
            _, subject, frame_idx = self._parse_filename(fpath)
            kp = self.parse_json_file(fpath)
            frames.append(kp)
            frame_indices.append(frame_idx)

        keypoints = np.stack(frames, axis=0)
        _, subject_id, _ = self._parse_filename(json_files[0])

        metadata = {
            "num_json_files": len(json_files),
            "coordinate_system": "image_2d",
            "camera": session_id.split("_")[-1] if "_" in session_id else "unknown",
            "file_dir": str(json_files[0].parent),
        }

        return SkeletonSequence(
            sequence_id=session_id,
            source="openpose",
            motion_name=motion_name,
            subject_id=subject_id,
            keypoints=keypoints,
            joint_names=OPENPOSE_BODY25_JOINTS.copy(),
            frame_indices=np.array(frame_indices, dtype=int),
            labels={},
            metadata=metadata,
        )

    def batch_load(
        self, json_dir: str | Path, motion_name: str
    ) -> List[SkeletonSequence]:
        """Load all sessions in a JSON directory."""
        sessions = self.discover_sessions(json_dir)
        sequences: List[SkeletonSequence] = []

        for session_id, files in sessions.items():
            try:
                seq = self.load_session(files, motion_name=motion_name, session_id=session_id)
                sequences.append(seq)
            except Exception as exc:
                logger.error("Failed session %s: %s", session_id, exc)

        logger.info(
            "OpenPose batch [%s]: %d session(s) loaded", motion_name, len(sequences)
        )
        return sequences
