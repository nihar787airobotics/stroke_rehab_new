"""Video preprocessing: loading, validation, and frame extraction."""

from preprocessing.frame_extraction import (
    FrameExtractionResult,
    FrameExtractor,
    batch_process_videos,
    extract_frames,
    resize_frames,
    sample_frames,
)
from preprocessing.video_loader import VideoLoader, VideoMetadata, get_video_metadata, load_video, validate_video

__all__ = [
    "VideoLoader",
    "VideoMetadata",
    "FrameExtractor",
    "FrameExtractionResult",
    "load_video",
    "validate_video",
    "get_video_metadata",
    "extract_frames",
    "sample_frames",
    "resize_frames",
    "batch_process_videos",
]
