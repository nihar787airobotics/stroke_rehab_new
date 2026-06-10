"""
preprocessing/video_loader.py
------------------------------
Handles video I/O: loading, validation, metadata extraction, and batch discovery.

Classes
-------
VideoMetadata : Dataclass holding per-video metadata.
VideoLoader     : Main class for load / validate / metadata / batch operations.

Module functions
----------------
load_video, validate_video, get_video_metadata
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config_loader import load_config
from utils.logger import get_logger

_cfg = load_config()
_log_file = str(_PROJECT_ROOT / _cfg.logging.file)
logger = get_logger(__name__, level=_cfg.logging.level, log_file=_log_file)


@dataclass
class VideoMetadata:
    """All descriptive properties extracted from a single video file."""

    file_path: str = ""
    file_name: str = ""
    file_size_mb: float = 0.0
    format: str = ""
    fps: float = 0.0
    total_frames: int = 0
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    resolution: str = ""
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialise to plain dictionary."""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size_mb": round(self.file_size_mb, 3),
            "format": self.format,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 3),
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"[{status}] {self.file_name} | "
            f"{self.width}x{self.height} | "
            f"{self.fps:.1f} fps | "
            f"{self.duration_seconds:.1f}s | "
            f"{self.total_frames} frames | "
            f"{self.file_size_mb:.2f} MB"
        )


class VideoLoader:
    """Loads, validates, and describes rehabilitation exercise videos."""

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        self._cfg = load_config(config_path)
        self._vcfg = self._cfg.video
        logger.info("VideoLoader initialised with config: %s", config_path or "default")

    def load_video(
        self, video_path: str | Path
    ) -> Tuple[cv2.VideoCapture, VideoMetadata]:
        """
        Open a video file and return the capture handle + metadata.

        Raises
        ------
        FileNotFoundError : File does not exist.
        ValueError        : File extension not supported.
        RuntimeError      : OpenCV could not open the file.
        """
        video_path = Path(video_path)
        logger.info("Loading video: %s", video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        ext = video_path.suffix.lower()
        supported = [s.lower() for s in self._vcfg.supported_formats]
        if ext not in supported:
            raise ValueError(f"Unsupported format '{ext}'. Supported: {supported}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open: {video_path}")

        meta = self.get_video_metadata(video_path, cap)
        logger.debug("Loaded -> %s", meta)
        return cap, meta

    def validate_video(self, meta: VideoMetadata) -> VideoMetadata:
        """
        Run constraint checks against VideoMetadata and annotate it.

        Checks: FPS, duration, resolution, minimum frame count.
        """
        errors: List[str] = []
        vcfg = self._vcfg
        fcfg = self._cfg.frame_extraction

        if not (vcfg.min_fps <= meta.fps <= vcfg.max_fps):
            errors.append(
                f"FPS {meta.fps:.2f} out of range [{vcfg.min_fps}, {vcfg.max_fps}]"
            )

        if not (vcfg.min_duration_seconds <= meta.duration_seconds <= vcfg.max_duration_seconds):
            errors.append(
                f"Duration {meta.duration_seconds:.2f}s out of range "
                f"[{vcfg.min_duration_seconds}, {vcfg.max_duration_seconds}]"
            )

        if not (vcfg.min_width <= meta.width <= vcfg.max_width):
            errors.append(
                f"Width {meta.width} out of range [{vcfg.min_width}, {vcfg.max_width}]"
            )

        if not (vcfg.min_height <= meta.height <= vcfg.max_height):
            errors.append(
                f"Height {meta.height} out of range [{vcfg.min_height}, {vcfg.max_height}]"
            )

        if meta.total_frames < fcfg.min_frames:
            errors.append(
                f"Total frames {meta.total_frames} < minimum {fcfg.min_frames}"
            )

        meta.validation_errors = errors
        meta.is_valid = len(errors) == 0

        level = logger.info if meta.is_valid else logger.warning
        level(
            "Validation %s for %s%s",
            "PASSED" if meta.is_valid else "FAILED",
            meta.file_name,
            "" if meta.is_valid else f" — errors: {errors}",
        )
        return meta

    def get_video_metadata(
        self,
        video_path: str | Path,
        cap: Optional[cv2.VideoCapture] = None,
    ) -> VideoMetadata:
        """Extract complete metadata from a video file."""
        video_path = Path(video_path)
        owns_cap = cap is None

        if owns_cap:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open for metadata: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            duration = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
            file_size_mb = (
                video_path.stat().st_size / (1024 * 1024) if video_path.exists() else 0.0
            )

            meta = VideoMetadata(
                file_path=str(video_path.resolve()),
                file_name=video_path.name,
                file_size_mb=file_size_mb,
                format=video_path.suffix.lower(),
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration,
                width=width,
                height=height,
                resolution=f"{width}x{height}",
            )
        finally:
            if owns_cap:
                cap.release()

        logger.debug("Metadata extracted: %s", meta)
        return meta

    def discover_videos(
        self,
        directory: str | Path,
        recursive: bool = False,
    ) -> List[Path]:
        """Find all supported video files in a directory."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        supported = {s.lower() for s in self._vcfg.supported_formats}
        pattern = "**/*" if recursive else "*"
        found: List[Path] = [
            p
            for p in sorted(directory.glob(pattern))
            if p.is_file() and p.suffix.lower() in supported
        ]

        logger.info(
            "Discovered %d video(s) in '%s' (recursive=%s)",
            len(found),
            directory,
            recursive,
        )
        return found

    def batch_load_metadata(
        self,
        directory: str | Path,
        recursive: bool = False,
    ) -> List[VideoMetadata]:
        """Load and validate metadata for every video in a directory."""
        paths = self.discover_videos(directory, recursive=recursive)
        if not paths:
            logger.warning("No videos found in: %s", directory)
            return []

        results: List[VideoMetadata] = []
        for idx, vpath in enumerate(paths, start=1):
            logger.info("Processing %d/%d -> %s", idx, len(paths), vpath.name)
            try:
                cap, meta = self.load_video(vpath)
                cap.release()
                meta = self.validate_video(meta)
            except Exception as exc:
                logger.error("Failed to load %s: %s", vpath.name, exc)
                meta = VideoMetadata(
                    file_path=str(vpath),
                    file_name=vpath.name,
                    is_valid=False,
                    validation_errors=[str(exc)],
                )
            results.append(meta)

        valid = sum(1 for m in results if m.is_valid)
        logger.info("Batch metadata complete: %d valid / %d total", valid, len(results))
        return results


# ---------------------------------------------------------------------------
# Module-level API (delegates to VideoLoader)
# ---------------------------------------------------------------------------
_default_loader: Optional[VideoLoader] = None


def _get_loader(config_path: Optional[str | Path] = None) -> VideoLoader:
    global _default_loader
    if config_path is not None:
        return VideoLoader(config_path)
    if _default_loader is None:
        _default_loader = VideoLoader()
    return _default_loader


def load_video(
    video_path: str | Path,
    config_path: Optional[str | Path] = None,
) -> Tuple[cv2.VideoCapture, VideoMetadata]:
    """Open a video and return (VideoCapture, VideoMetadata)."""
    return _get_loader(config_path).load_video(video_path)


def validate_video(
    meta: VideoMetadata,
    config_path: Optional[str | Path] = None,
) -> VideoMetadata:
    """Validate video metadata against configured constraints."""
    return _get_loader(config_path).validate_video(meta)


def get_video_metadata(
    video_path: str | Path,
    cap: Optional[cv2.VideoCapture] = None,
    config_path: Optional[str | Path] = None,
) -> VideoMetadata:
    """Extract metadata from a video file."""
    return _get_loader(config_path).get_video_metadata(video_path, cap)


if __name__ == "__main__":
    import json

    loader = VideoLoader()
    raw_dir = _PROJECT_ROOT / "data" / "raw"
    videos = loader.discover_videos(raw_dir)

    if not videos:
        print(f"No videos found in {raw_dir}. Place .mp4/.avi/.mov files there to test.")
    else:
        for vp in videos:
            cap, meta = loader.load_video(vp)
            cap.release()
            meta = loader.validate_video(meta)
            print(json.dumps(meta.to_dict(), indent=2))
