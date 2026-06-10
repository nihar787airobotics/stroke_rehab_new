"""
preprocessing/label_loader.py
-----------------------------
Load participant metadata and performance scores for Kinect dataset.
IMU data is intentionally excluded from this project.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config_loader import load_config
from utils.logger import get_logger

_cfg = load_config()
logger = get_logger(__name__, level=_cfg.logging.level)


@dataclass
class PerformanceLabel:
    """Performance scores for one participant + exercise."""

    participant_id: str
    exercise_type: str
    performance_outcome: float
    control_factor: float
    gender: Optional[str] = None
    age: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "participant_id": self.participant_id,
            "exercise_type": self.exercise_type,
            "performance_outcome": self.performance_outcome,
            "control_factor": self.control_factor,
            "gender": self.gender,
            "age": self.age,
        }


class LabelLoader:
    """Load and query CSV labels for the Kinect rehabilitation dataset."""

    PARTICIPANT_COL = "Partcipant Id"  # typo preserved from source file

    def __init__(self, dataset_root: Optional[str | Path] = None) -> None:
        ds = _cfg.datasets
        self._root = Path(dataset_root or ds.stroke_rehab_root)
        self._performance_path = self._root / ds.performance_scores
        self._participants_path = self._root / ds.participants_info
        self._performance_df: Optional[pd.DataFrame] = None
        self._participants_df: Optional[pd.DataFrame] = None
        logger.info("LabelLoader root: %s", self._root)

    def _normalise_id(self, pid: str) -> str:
        """P1, p01, P01 -> P1 for consistent lookup."""
        pid = str(pid).strip()
        m = re.match(r"^[Pp](\d+)$", pid)
        if m:
            return f"P{int(m.group(1))}"
        return pid

    def load_performance_scores(self) -> pd.DataFrame:
        if self._performance_df is not None:
            return self._performance_df

        if not self._performance_path.exists():
            raise FileNotFoundError(f"Performance scores not found: {self._performance_path}")

        df = pd.read_csv(self._performance_path)
        df[self.PARTICIPANT_COL] = df[self.PARTICIPANT_COL].astype(str).map(self._normalise_id)
        self._performance_df = df
        logger.info("Loaded %d performance score rows", len(df))
        return df

    def load_participants_info(self) -> pd.DataFrame:
        if self._participants_df is not None:
            return self._participants_df

        if not self._participants_path.exists():
            raise FileNotFoundError(f"Participants info not found: {self._participants_path}")

        df = pd.read_csv(self._participants_path)
        df[self.PARTICIPANT_COL] = df[self.PARTICIPANT_COL].astype(str).map(self._normalise_id)
        self._participants_df = df
        logger.info("Loaded %d participant info rows", len(df))
        return df

    def get_label_for_skeleton_file(self, skeleton_path: str | Path) -> Optional[PerformanceLabel]:
        """
        Parse filename like 'P1_Female_21_Exercise Type 1.skeleton' and return labels.
        """
        skeleton_path = Path(skeleton_path)
        stem = skeleton_path.stem
        parts = stem.split("_", 3)
        if len(parts) < 4:
            logger.warning("Cannot parse skeleton filename: %s", skeleton_path.name)
            return None

        participant_id = self._normalise_id(parts[0])
        gender = parts[1]
        age = int(parts[2])
        exercise_type = parts[3]

        perf = self.load_performance_scores()
        row = perf[
            (perf[self.PARTICIPANT_COL] == participant_id)
            & (perf["Exercise Type"] == exercise_type)
        ]

        if row.empty:
            logger.warning(
                "No performance label for %s / %s", participant_id, exercise_type
            )
            return PerformanceLabel(
                participant_id=participant_id,
                exercise_type=exercise_type,
                performance_outcome=float("nan"),
                control_factor=float("nan"),
                gender=gender,
                age=age,
            )

        r = row.iloc[0]
        return PerformanceLabel(
            participant_id=participant_id,
            exercise_type=exercise_type,
            performance_outcome=float(r["Performance Outcome"]),
            control_factor=float(r["Control Factor"]),
            gender=gender,
            age=age,
        )

    def get_all_labels(self) -> List[PerformanceLabel]:
        perf = self.load_performance_scores()
        info = self.load_participants_info()
        info_map = info.set_index(self.PARTICIPANT_COL).to_dict(orient="index")

        labels: List[PerformanceLabel] = []
        for _, row in perf.iterrows():
            pid = row[self.PARTICIPANT_COL]
            meta = info_map.get(pid, {})
            labels.append(
                PerformanceLabel(
                    participant_id=pid,
                    exercise_type=str(row["Exercise Type"]),
                    performance_outcome=float(row["Performance Outcome"]),
                    control_factor=float(row["Control Factor"]),
                    gender=meta.get("Gender"),
                    age=int(meta["Age"]) if meta.get("Age") is not None else None,
                )
            )
        return labels
