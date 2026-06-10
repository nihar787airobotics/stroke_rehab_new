"""
preprocessing/skeleton_pipeline.py
----------------------------------
Unified Phase 1 pipeline for skeleton datasets (no video, no IMU).

Sources
-------
1. Kinect .skeleton  — 3D joint positions + performance labels
2. OpenPose JSON     — 2D keypoints (C_Front_Away, C_Sag_Left, C_Sag_Right)
3. Archive Excel     — 3D joint positions per motion (when available)

Output
------
data/processed/{source}/{sequence_id}/keypoints.npz
data/processed/manifest.csv
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.archive_joint_loader import ArchiveJointLoader
from preprocessing.kinect_skeleton_loader import KinectSkeletonLoader
from preprocessing.openpose_json_loader import OpenPoseJsonLoader
from preprocessing.skeleton_sequence import SkeletonSequence
from utils.config_loader import load_config
from utils.logger import get_logger

_cfg = load_config()
_log_file = str(_PROJECT_ROOT / _cfg.logging.file)
logger = get_logger(__name__, level=_cfg.logging.level, log_file=_log_file)


@dataclass
class IngestionResult:
    """Result of ingesting one skeleton sequence."""

    sequence_id: str
    source: str
    motion_name: str
    subject_id: str
    output_path: str = ""
    num_frames: int = 0
    num_joints: int = 0
    success: bool = False
    error: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "sequence_id": self.sequence_id,
            "source": self.source,
            "motion_name": self.motion_name,
            "subject_id": self.subject_id,
            "output_path": self.output_path,
            "num_frames": self.num_frames,
            "num_joints": self.num_joints,
            "success": self.success,
            "error": self.error,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass
class BatchIngestionSummary:
    total: int = 0
    successful: int = 0
    failed: int = 0
    results: List[IngestionResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "total_frames": sum(r.num_frames for r in self.results if r.success),
            "failed_items": [
                {"sequence_id": r.sequence_id, "error": r.error}
                for r in self.results
                if not r.success
            ],
        }


class SkeletonPipeline:
    """
    Phase 1 skeleton data ingestion pipeline.

    Processes Kinect, OpenPose, and Archive datasets into a unified
    numpy format under data/processed/.
    """

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
    ) -> None:
        self._cfg = load_config(config_path)
        self._scfg = self._cfg.skeleton_ingestion
        self._output_base = Path(output_dir or _PROJECT_ROOT / self._cfg.paths.data_processed)
        self._output_base.mkdir(parents=True, exist_ok=True)

        self._kinect = KinectSkeletonLoader()
        self._openpose = OpenPoseJsonLoader()
        self._archive = ArchiveJointLoader()

        logger.info("SkeletonPipeline output: %s", self._output_base)

    def save_sequence(self, seq: SkeletonSequence) -> Path:
        """Save a SkeletonSequence to disk as .npz + metadata.json."""
        max_frames = int(self._scfg.max_frames)
        if seq.num_frames > max_frames:
            seq = seq.sample_frames(max_frames)

        out_dir = self._output_base / seq.source / seq.sequence_id
        out_dir.mkdir(parents=True, exist_ok=True)

        npz_path = out_dir / "keypoints.npz"
        np.savez_compressed(
            npz_path,
            keypoints=seq.keypoints,
            frame_indices=seq.frame_indices,
            joint_names=np.array(seq.joint_names, dtype=object),
        )

        meta_path = out_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    **seq.to_dict(),
                    "joint_names": seq.joint_names,
                    "labels": seq.labels,
                    "metadata": seq.metadata,
                },
                fh,
                indent=2,
                default=str,
            )

        logger.debug("Saved %s -> %s", seq.sequence_id, npz_path)
        return npz_path

    def ingest_sequence(self, seq: SkeletonSequence) -> IngestionResult:
        t0 = time.perf_counter()
        result = IngestionResult(
            sequence_id=seq.sequence_id,
            source=seq.source,
            motion_name=seq.motion_name,
            subject_id=seq.subject_id,
        )

        min_frames = int(self._scfg.min_frames)
        if seq.num_frames < min_frames:
            result.error = (
                f"Too few frames: {seq.num_frames} < minimum {min_frames}"
            )
            result.elapsed_seconds = time.perf_counter() - t0
            return result

        try:
            out_path = self.save_sequence(seq)
            result.output_path = str(out_path)
            result.num_frames = min(seq.num_frames, int(self._scfg.max_frames))
            result.num_joints = seq.num_joints
            result.success = True
        except Exception as exc:
            result.error = str(exc)
            logger.error("Ingest failed %s: %s", seq.sequence_id, exc)

        result.elapsed_seconds = time.perf_counter() - t0
        return result

    def process_kinect(self) -> List[IngestionResult]:
        ds = self._cfg.datasets
        kinect_dir = Path(ds.stroke_rehab_root) / ds.kinect_skeleton_dir

        if not kinect_dir.is_dir():
            logger.warning("Kinect directory not found: %s", kinect_dir)
            return []

        sequences = self._kinect.batch_load(kinect_dir)
        return [self.ingest_sequence(seq) for seq in sequences]

    def process_openpose(self) -> List[IngestionResult]:
        results: List[IngestionResult] = []
        openpose_cfg = dict(self._cfg.datasets.openpose)

        for motion_name, json_dir in openpose_cfg.items():
            json_path = Path(str(json_dir))
            if not json_path.is_dir():
                logger.warning("OpenPose dir not found [%s]: %s", motion_name, json_path)
                continue

            sequences = self._openpose.batch_load(json_path, motion_name=motion_name)
            results.extend(self.ingest_sequence(seq) for seq in sequences)

        return results

    def process_archive(self) -> List[IngestionResult]:
        if not self._archive._root.is_dir():
            logger.warning(
                "Archive dataset not found: %s (skip until you add it)",
                self._archive._root,
            )
            return []

        sequences = self._archive.batch_load()
        return [self.ingest_sequence(seq) for seq in sequences]

    def batch_process_all(
        self,
        include_kinect: bool = True,
        include_openpose: bool = True,
        include_archive: bool = True,
    ) -> BatchIngestionSummary:
        """
        Process all available skeleton datasets.

        IMU data is never loaded — camera/skeleton only.
        """
        all_results: List[IngestionResult] = []

        if include_kinect:
            logger.info("=== Processing Kinect skeleton data ===")
            all_results.extend(self.process_kinect())

        if include_openpose:
            logger.info("=== Processing OpenPose JSON data ===")
            all_results.extend(self.process_openpose())

        if include_archive:
            logger.info("=== Processing Archive joint data ===")
            all_results.extend(self.process_archive())

        summary = BatchIngestionSummary(
            total=len(all_results),
            successful=sum(1 for r in all_results if r.success),
            failed=sum(1 for r in all_results if not r.success),
            results=all_results,
        )

        if self._scfg.save_manifest:
            self._write_manifest(all_results)

        logger.info(
            "Batch complete: %d/%d succeeded",
            summary.successful,
            summary.total,
        )
        return summary

    def _write_manifest(self, results: List[IngestionResult]) -> Path:
        rows = []
        for r in results:
            if not r.success:
                continue
            meta_path = Path(r.output_path).parent / "metadata.json"
            labels = {}
            extra = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                    labels = meta.get("labels", {})
                    extra = meta.get("metadata", {})

            rows.append(
                {
                    "sequence_id": r.sequence_id,
                    "source": r.source,
                    "motion_name": r.motion_name,
                    "subject_id": r.subject_id,
                    "num_frames": r.num_frames,
                    "num_joints": r.num_joints,
                    "output_path": r.output_path,
                    "performance_outcome": labels.get("performance_outcome"),
                    "control_factor": labels.get("control_factor"),
                    "cohort": extra.get("cohort"),
                }
            )

        manifest_path = self._output_base / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        logger.info("Manifest saved: %s (%d rows)", manifest_path, len(rows))
        return manifest_path


def batch_process_skeletons(
    output_dir: Optional[str | Path] = None,
    include_kinect: bool = True,
    include_openpose: bool = True,
    include_archive: bool = True,
) -> BatchIngestionSummary:
    """Module-level entry point for Phase 1 skeleton ingestion."""
    pipeline = SkeletonPipeline(output_dir=output_dir)
    return pipeline.batch_process_all(
        include_kinect=include_kinect,
        include_openpose=include_openpose,
        include_archive=include_archive,
    )
