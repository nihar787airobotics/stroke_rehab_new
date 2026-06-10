"""
preprocessing/frame_extraction.py
----------------------------------
Extracts, samples, and resizes frames from rehabilitation videos.

Pipeline
--------
VideoCapture -> extract_frames -> sample_frames -> resize_frames -> save

Classes
-------
FrameExtractionResult : Summary of one video's extraction pass.
FrameExtractor        : Main extraction engine.

Module functions
----------------
extract_frames, sample_frames, resize_frames, batch_process_videos
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.video_loader import VideoLoader, VideoMetadata
from utils.config_loader import load_config
from utils.logger import get_logger

_cfg = load_config()
_log_file = str(_PROJECT_ROOT / _cfg.logging.file)
logger = get_logger(__name__, level=_cfg.logging.level, log_file=_log_file)


@dataclass
class FrameExtractionResult:
    """Summary of one video's extraction pass."""

    video_path: str = ""
    video_name: str = ""
    output_dir: str = ""
    raw_frames_extracted: int = 0
    frames_after_sampling: int = 0
    frames_saved: int = 0
    original_width: int = 0
    original_height: int = 0
    resized_width: int = 0
    resized_height: int = 0
    elapsed_seconds: float = 0.0
    success: bool = False
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            "video_path": self.video_path,
            "video_name": self.video_name,
            "output_dir": self.output_dir,
            "raw_frames_extracted": self.raw_frames_extracted,
            "frames_after_sampling": self.frames_after_sampling,
            "frames_saved": self.frames_saved,
            "original_resolution": f"{self.original_width}x{self.original_height}",
            "resized_resolution": f"{self.resized_width}x{self.resized_height}",
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "success": self.success,
            "error": self.error,
        }

    def __str__(self) -> str:
        status = "OK" if self.success else f"FAIL({self.error})"
        return (
            f"[{status}] {self.video_name} | "
            f"extracted={self.raw_frames_extracted} "
            f"sampled={self.frames_after_sampling} "
            f"saved={self.frames_saved} | "
            f"{self.elapsed_seconds:.2f}s"
        )


class FrameExtractor:
    """Extracts frames from rehabilitation videos and saves them to disk."""

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        output_base_dir: Optional[str | Path] = None,
    ) -> None:
        self._cfg = load_config(config_path)
        self._fcfg = self._cfg.frame_extraction
        self._loader = VideoLoader(config_path)

        if output_base_dir is not None:
            self._output_base = Path(output_base_dir)
        else:
            self._output_base = _PROJECT_ROOT / self._cfg.paths.data_processed

        self._output_base.mkdir(parents=True, exist_ok=True)
        logger.info("FrameExtractor initialised | output_base=%s", self._output_base)

    def extract_frames(
        self,
        cap: cv2.VideoCapture,
        meta: VideoMetadata,
    ) -> List[np.ndarray]:
        """
        Pull frames from an open VideoCapture at the configured target FPS.

        Computes frame-step from native FPS vs target FPS and reads every nth frame.
        """
        target_fps: float = float(self._fcfg.target_fps)
        source_fps: float = max(meta.fps, 1.0)
        frame_step: int = max(1, int(round(source_fps / target_fps)))

        logger.debug(
            "extract_frames | source_fps=%.1f target_fps=%.1f step=%d",
            source_fps,
            target_fps,
            frame_step,
        )

        frames: List[np.ndarray] = []
        frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                frames.append(frame)
            frame_idx += 1

        logger.debug(
            "extract_frames -> %d frames (read %d total frames)",
            len(frames),
            frame_idx,
        )
        return frames

    def sample_frames(
        self,
        frames: List[np.ndarray],
        max_frames: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Sub-sample a frame list to at most max_frames using uniform spacing."""
        if max_frames is None:
            max_frames = int(self._fcfg.max_frames)

        n = len(frames)
        if n == 0:
            return []

        if n <= max_frames:
            logger.debug("sample_frames | %d <= max %d — no sampling needed", n, max_frames)
            return frames

        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        sampled = [frames[i] for i in indices]
        logger.debug("sample_frames | %d -> %d frames (uniform)", n, len(sampled))
        return sampled

    def resize_frames(
        self,
        frames: List[np.ndarray],
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Resize a list of frames to target (W, H)."""
        w = int(target_width or self._fcfg.resize_width)
        h = int(target_height or self._fcfg.resize_height)

        if not frames:
            return []

        orig_h, orig_w = frames[0].shape[:2]
        if orig_w == w and orig_h == h:
            logger.debug("resize_frames | already %dx%d — skip", w, h)
            return frames

        resized = [
            cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames
        ]
        logger.debug(
            "resize_frames | %dx%d -> %dx%d (%d frames)",
            orig_w,
            orig_h,
            w,
            h,
            len(resized),
        )
        return resized

    def _make_output_dir(self, video_path: Path) -> Path:
        out_dir = self._output_base / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _save_frames(
        self,
        frames: List[np.ndarray],
        output_dir: Path,
        fmt: Optional[str] = None,
        quality: Optional[int] = None,
    ) -> int:
        fmt = fmt or str(self._fcfg.output_format).lower().lstrip(".")
        quality = quality if quality is not None else int(self._fcfg.quality)

        encode_params: List[int] = []
        if fmt in ("jpg", "jpeg"):
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif fmt == "png":
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, quality // 10)]

        saved = 0
        digits = len(str(len(frames)))
        for i, frame in enumerate(frames):
            filename = output_dir / f"frame_{str(i).zfill(digits)}.{fmt}"
            ok = cv2.imwrite(str(filename), frame, encode_params)
            if ok:
                saved += 1
            else:
                logger.warning("Failed to save frame %d -> %s", i, filename)

        logger.debug("Saved %d/%d frames to %s", saved, len(frames), output_dir)
        return saved

    def process_video(
        self,
        video_path: str | Path,
        skip_invalid: bool = True,
    ) -> FrameExtractionResult:
        """
        Full pipeline for a single video:
        load -> validate -> extract -> sample -> resize -> save
        """
        video_path = Path(video_path)
        result = FrameExtractionResult(
            video_path=str(video_path),
            video_name=video_path.name,
        )
        t0 = time.perf_counter()

        try:
            cap, meta = self._loader.load_video(video_path)
            meta = self._loader.validate_video(meta)

            if not meta.is_valid:
                msg = f"Validation failed: {meta.validation_errors}"
                logger.warning(msg)
                if skip_invalid:
                    result.error = msg
                    result.elapsed_seconds = time.perf_counter() - t0
                    cap.release()
                    return result
                raise ValueError(msg)

            result.original_width = meta.width
            result.original_height = meta.height

            frames = self.extract_frames(cap, meta)
            cap.release()
            result.raw_frames_extracted = len(frames)

            if not frames:
                raise RuntimeError("No frames extracted from video.")

            frames = self.sample_frames(frames)
            result.frames_after_sampling = len(frames)

            frames = self.resize_frames(frames)
            w = int(self._fcfg.resize_width)
            h = int(self._fcfg.resize_height)
            result.resized_width = w
            result.resized_height = h

            out_dir = self._make_output_dir(video_path)
            result.output_dir = str(out_dir)
            saved = self._save_frames(frames, out_dir)
            result.frames_saved = saved
            result.success = saved > 0

            logger.info(
                "process_video OK: %s -> %d frames saved to %s",
                video_path.name,
                saved,
                out_dir,
            )

        except Exception as exc:
            logger.error("process_video FAILED: %s — %s", video_path.name, exc)
            result.error = str(exc)
            result.success = False

        result.elapsed_seconds = time.perf_counter() - t0
        return result

    def batch_process_videos(
        self,
        input_dir: str | Path,
        recursive: bool = False,
        skip_invalid: bool = True,
    ) -> List[FrameExtractionResult]:
        """Process every video in a directory."""
        input_dir = Path(input_dir)
        video_paths = self._loader.discover_videos(input_dir, recursive=recursive)

        if not video_paths:
            logger.warning("No videos found in: %s", input_dir)
            return []

        results: List[FrameExtractionResult] = []
        total = len(video_paths)

        logger.info("Starting batch processing: %d video(s)", total)
        for idx, vpath in enumerate(video_paths, start=1):
            logger.info("Batch %d/%d: %s", idx, total, vpath.name)
            res = self.process_video(vpath, skip_invalid=skip_invalid)
            results.append(res)

        success_count = sum(1 for r in results if r.success)
        total_frames = sum(r.frames_saved for r in results)
        logger.info(
            "Batch complete: %d/%d succeeded | %d total frames saved",
            success_count,
            total,
            total_frames,
        )
        return results

    def get_batch_summary(self, results: List[FrameExtractionResult]) -> Dict:
        """Aggregate statistics across a list of results."""
        return {
            "total_videos": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "total_frames_saved": sum(r.frames_saved for r in results),
            "total_elapsed_seconds": round(sum(r.elapsed_seconds for r in results), 3),
            "failed_videos": [
                {"name": r.video_name, "error": r.error}
                for r in results
                if not r.success
            ],
        }


# ---------------------------------------------------------------------------
# Module-level API (delegates to FrameExtractor)
# ---------------------------------------------------------------------------
_default_extractor: Optional[FrameExtractor] = None


def _get_extractor(
    config_path: Optional[str | Path] = None,
    output_base_dir: Optional[str | Path] = None,
) -> FrameExtractor:
    global _default_extractor
    if config_path is not None or output_base_dir is not None:
        return FrameExtractor(config_path, output_base_dir)
    if _default_extractor is None:
        _default_extractor = FrameExtractor()
    return _default_extractor


def extract_frames(
    cap: cv2.VideoCapture,
    meta: VideoMetadata,
    config_path: Optional[str | Path] = None,
) -> List[np.ndarray]:
    """Extract frames at configured target FPS from an open VideoCapture."""
    return _get_extractor(config_path).extract_frames(cap, meta)


def sample_frames(
    frames: List[np.ndarray],
    max_frames: Optional[int] = None,
    config_path: Optional[str | Path] = None,
) -> List[np.ndarray]:
    """Uniformly sub-sample frames to at most max_frames."""
    return _get_extractor(config_path).sample_frames(frames, max_frames)


def resize_frames(
    frames: List[np.ndarray],
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    config_path: Optional[str | Path] = None,
) -> List[np.ndarray]:
    """Resize frames to target dimensions."""
    return _get_extractor(config_path).resize_frames(frames, target_width, target_height)


def batch_process_videos(
    input_dir: str | Path,
    recursive: bool = False,
    skip_invalid: bool = True,
    config_path: Optional[str | Path] = None,
    output_base_dir: Optional[str | Path] = None,
) -> List[FrameExtractionResult]:
    """Process all videos in input_dir and save frames to data/processed/."""
    return _get_extractor(config_path, output_base_dir).batch_process_videos(
        input_dir, recursive=recursive, skip_invalid=skip_invalid
    )


if __name__ == "__main__":
    import json

    extractor = FrameExtractor()
    raw_dir = _PROJECT_ROOT / "data" / "raw"
    results = extractor.batch_process_videos(raw_dir)
    print(json.dumps(extractor.get_batch_summary(results), indent=2))
    for res in results:
        print(res)
