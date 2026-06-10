"""Preprocessing: video (optional) and skeleton data ingestion (Phase 1)."""

from preprocessing.frame_extraction import (
    FrameExtractionResult,
    FrameExtractor,
    batch_process_videos,
    extract_frames,
    resize_frames,
    sample_frames,
)
from preprocessing.kinect_skeleton_loader import KinectSkeletonLoader
from preprocessing.openpose_json_loader import OpenPoseJsonLoader
from preprocessing.archive_joint_loader import ArchiveJointLoader
from preprocessing.skeleton_pipeline import (
    BatchIngestionSummary,
    IngestionResult,
    SkeletonPipeline,
    batch_process_skeletons,
)
from preprocessing.skeleton_sequence import SkeletonSequence
from preprocessing.label_loader import LabelLoader, PerformanceLabel
from preprocessing.video_loader import VideoLoader, VideoMetadata, get_video_metadata, load_video, validate_video

__all__ = [
    # Video (optional — when RGB videos available)
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
    # Skeleton (primary Phase 1 path)
    "SkeletonSequence",
    "SkeletonPipeline",
    "IngestionResult",
    "BatchIngestionSummary",
    "KinectSkeletonLoader",
    "OpenPoseJsonLoader",
    "ArchiveJointLoader",
    "LabelLoader",
    "PerformanceLabel",
    "batch_process_skeletons",
]
