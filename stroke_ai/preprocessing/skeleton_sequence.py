"""
preprocessing/skeleton_sequence.py
----------------------------------
Unified dataclass for skeleton sequences from any data source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SkeletonSequence:
    """
    Normalised skeleton time-series.

    keypoints shape: (num_frames, num_joints, num_dims)
        - Kinect 3D: dims = 3  (x, y, z in metres)
        - OpenPose 2D: dims = 3  (x, y, confidence)
        - Archive 3D: dims = 3  (x, y, z)
    """

    sequence_id: str
    source: str                    # kinect | openpose | archive
    motion_name: str               # e.g. Exercise Type 1, C_Front_Away
    subject_id: str
    keypoints: np.ndarray
    joint_names: List[str]
    frame_indices: np.ndarray
    labels: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.keypoints.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.keypoints.shape[1])

    @property
    def num_dims(self) -> int:
        return int(self.keypoints.shape[2])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "source": self.source,
            "motion_name": self.motion_name,
            "subject_id": self.subject_id,
            "num_frames": self.num_frames,
            "num_joints": self.num_joints,
            "num_dims": self.num_dims,
            "labels": self.labels,
            "metadata": self.metadata,
        }

    def sample_frames(self, max_frames: int) -> "SkeletonSequence":
        """Uniformly sub-sample to at most max_frames."""
        n = self.num_frames
        if n <= max_frames:
            return self

        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        return SkeletonSequence(
            sequence_id=self.sequence_id,
            source=self.source,
            motion_name=self.motion_name,
            subject_id=self.subject_id,
            keypoints=self.keypoints[indices],
            joint_names=self.joint_names,
            frame_indices=self.frame_indices[indices],
            labels=self.labels,
            metadata={**self.metadata, "sampled_from": n, "sampled_to": max_frames},
        )
