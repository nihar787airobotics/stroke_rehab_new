"""
examples/run_phase1_skeleton.py
-------------------------------
Phase 1 — Skeleton Data Ingestion (no video, no IMU).

Processes your available datasets:
  1. Kinect .skeleton (3D joints + performance labels)
  2. OpenPose JSON (C_Front_Away, C_Sag_Left, C_Sag_Right)
  3. Archive Joint_Positions (when you add C:/datasets/archive/)

Run from stroke_ai/:
    python examples/run_phase1_skeleton.py

Output:
    data/processed/kinect/<sequence_id>/keypoints.npz
    data/processed/openpose/<session_id>/keypoints.npz
    data/processed/manifest.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.skeleton_pipeline import SkeletonPipeline
from utils.config_loader import load_config


def main() -> None:
    cfg = load_config()
    print("=" * 60)
    print("Stroke AI — Phase 1: Skeleton Data Ingestion")
    print("=" * 60)
    print("\nConfigured dataset paths:")
    ds = cfg.datasets
    print(f"  Kinect:    {ds.stroke_rehab_root}/{ds.kinect_skeleton_dir}")
    for name, path in dict(ds.openpose).items():
        exists = Path(str(path)).is_dir()
        status = "OK" if exists else "NOT FOUND"
        print(f"  {name:14} {path}  [{status}]")
    print(f"  Archive:   {ds.archive_root}  [{'OK' if Path(ds.archive_root).is_dir() else 'NOT FOUND'}]")
    print("\nIMU data: NOT used (by design)\n")

    pipeline = SkeletonPipeline()
    summary = pipeline.batch_process_all(
        include_kinect=True,
        include_openpose=True,
        include_archive=True,
    )

    print("\n--- Summary ---")
    print(json.dumps(summary.to_dict(), indent=2))

    if summary.successful == 0:
        print(
            "\nNo sequences processed. Check paths in configs/config.yaml "
            "and ensure datasets are on disk."
        )
        return

    print(f"\nProcessed {summary.successful} sequence(s).")
    print(f"Manifest: {pipeline._output_base / 'manifest.csv'}")


if __name__ == "__main__":
    main()
