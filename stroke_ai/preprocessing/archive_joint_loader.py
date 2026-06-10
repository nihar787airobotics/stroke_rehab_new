"""
preprocessing/archive_joint_loader.py
-------------------------------------
Load 3D joint positions from the hemiplegia archive dataset.

Expected layout:

archive/data_new/
├── H01/
│   ├── Motion1/
│   │   ├── Joint_Positions.csv
│   │   └── Labels.csv
│   └── ...
├── P01/
│   └── ...
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.skeleton_sequence import SkeletonSequence
from utils.config_loader import load_config
from utils.joint_definitions import ARCHIVE_GENERIC_JOINTS
from utils.logger import get_logger

_cfg = load_config()
logger = get_logger(__name__, level=_cfg.logging.level)

JOINTS_PER_FRAME = 25


class ArchiveJointLoader:
    """
    Loader for Hemiplegia Archive dataset.

    Output shape:
        (num_frames, 25, 3)
    """

    def __init__(
        self,
        archive_root: Optional[str | Path] = None,
    ) -> None:

        self._root = Path(
            archive_root or _cfg.datasets.archive_root
        )

    def _read_excel(
        self,
        path: Path,
    ) -> pd.DataFrame:

        suffix = path.suffix.lower()

        if suffix == ".csv":

            return pd.read_csv(
                path,
                header=None,
            )

        return pd.read_excel(
            path,
            header=None,
        )

    def _reshape_positions(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        """
        Convert raw joint table into:

            (T, 25, 3)

        Archive CSV format:

            x y z
            x y z
            ...
            25 rows = 1 frame
        """

        cols = {
            str(c).lower(): c
            for c in df.columns
        }

        x_col = next(
            (
                cols[k]
                for k in cols
                if k in ("x", "joint_x", "pos_x")
            ),
            None,
        )

        y_col = next(
            (
                cols[k]
                for k in cols
                if k in ("y", "joint_y", "pos_y")
            ),
            None,
        )

        z_col = next(
            (
                cols[k]
                for k in cols
                if k in ("z", "joint_z", "pos_z")
            ),
            None,
        )

        if all([x_col, y_col, z_col]):

            arr = df[
                [x_col, y_col, z_col]
            ].to_numpy(
                dtype=np.float64
            )

        else:

            numeric = df.select_dtypes(
                include=[np.number]
            )

            if numeric.shape[1] < 3:

                raise ValueError(
                    f"Expected at least 3 numeric columns, got {numeric.shape[1]}"
                )

            arr = numeric.iloc[:, :3].to_numpy(
                dtype=np.float64
            )

        n_rows = arr.shape[0]

        if n_rows % JOINTS_PER_FRAME == 0:

            num_frames = (
                n_rows // JOINTS_PER_FRAME
            )

            logger.info(
                "Archive reshape: %d rows -> (%d,%d,3)",
                n_rows,
                num_frames,
                JOINTS_PER_FRAME,
            )

            return arr.reshape(
                num_frames,
                JOINTS_PER_FRAME,
                3,
            )

        logger.warning(
            "Archive rows (%d) not divisible by %d. "
            "Falling back to single-joint trajectory.",
            n_rows,
            JOINTS_PER_FRAME,
        )

        return arr.reshape(
            n_rows,
            1,
            3,
        )

    def _load_labels(
        self,
        motion_dir: Path,
    ) -> dict:

        for name in (
            "Labels.xls",
            "Labels.xlsx",
            "Labels.csv",
        ):

            label_path = motion_dir / name

            if not label_path.exists():
                continue

            try:

                df = self._read_excel(
                    label_path
                )

                values = (
                    df.iloc[:, 0]
                    .dropna()
                    .tolist()
                )

                return {
                    "frame_labels": values,
                    "label_file": str(
                        label_path
                    ),
                }

            except Exception as exc:

                logger.warning(
                    "Failed loading labels %s: %s",
                    label_path,
                    exc,
                )

        return {}

    def load_motion(
        self,
        subject_dir: Path,
        motion_name: str,
    ) -> SkeletonSequence:

        motion_dir = subject_dir / motion_name

        joint_path = None

        for name in (
            "Joint_Positions.xls",
            "Joint_Positions.xlsx",
            "Joint_Positions.csv",
        ):

            candidate = motion_dir / name

            if candidate.exists():
                joint_path = candidate
                break

        if joint_path is None:

            raise FileNotFoundError(
                f"Joint_Positions not found in {motion_dir}"
            )

        df = self._read_excel(
            joint_path
        )

        keypoints = self._reshape_positions(
            df
        )

        labels = self._load_labels(
            motion_dir
        )

        subject_id = subject_dir.name

        cohort = (
            "healthy"
            if subject_id.upper().startswith(
                "H"
            )
            else "patient"
        )

        metadata = {
            "file_path": str(
                joint_path.resolve()
            ),
            "motion_dir": str(
                motion_dir
            ),
            "cohort": cohort,
            "coordinate_system": "archive_3d",
            "units": "metres",
        }

        return SkeletonSequence(
            sequence_id=f"{subject_id}_{motion_name}",
            source="archive",
            motion_name=motion_name,
            subject_id=subject_id,
            keypoints=keypoints,
            joint_names=ARCHIVE_GENERIC_JOINTS.copy(),
            frame_indices=np.arange(
                keypoints.shape[0],
                dtype=int,
            ),
            labels=labels,
            metadata=metadata,
        )

    def discover_subjects(
        self,
    ) -> List[Path]:

        if not self._root.exists():

            logger.warning(
                "Archive root not found: %s",
                self._root,
            )

            return []

        return sorted(
            [
                p
                for p in self._root.iterdir()
                if p.is_dir()
            ]
        )

    def discover_motions(
        self,
        subject_dir: Path,
    ) -> List[str]:

        motions = []

        for d in sorted(
            subject_dir.iterdir()
        ):

            if (
                d.is_dir()
                and any(
                    (
                        d / name
                    ).exists()
                    for name in (
                        "Joint_Positions.xls",
                        "Joint_Positions.xlsx",
                        "Joint_Positions.csv",
                    )
                )
            ):
                motions.append(
                    d.name
                )

        return motions

    def batch_load(
        self,
    ) -> List[SkeletonSequence]:

        sequences = []

        subjects = (
            self.discover_subjects()
        )

        for subject_dir in subjects:

            motions = self.discover_motions(
                subject_dir
            )

            for motion in motions:

                try:

                    seq = self.load_motion(
                        subject_dir,
                        motion,
                    )

                    sequences.append(
                        seq
                    )

                except Exception as exc:

                    logger.error(
                        "Failed %s/%s: %s",
                        subject_dir.name,
                        motion,
                        exc,
                    )

        logger.info(
            "Archive batch complete: %d sequence(s)",
            len(sequences),
        )

        return sequences