"""
examples/run_phase1.py
----------------------
Example usage for Phase 1: Video Processing Pipeline.

Steps:
1. Place rehabilitation videos in data/raw/  (.mp4, .avi, .mov, .mkv)
2. Run this script from the stroke_ai/ directory:

       python examples/run_phase1.py

3. Extracted frames are saved to data/processed/<video_name>/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.frame_extraction import FrameExtractor, batch_process_videos
from preprocessing.video_loader import VideoLoader


def main() -> None:
    raw_dir = _PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    loader = VideoLoader()
    videos = loader.discover_videos(raw_dir)

    if not videos:
        print(f"No videos found in {raw_dir}")
        print("Place .mp4/.avi/.mov files in data/raw/ and re-run.")
        return

    print(f"Found {len(videos)} video(s) in {raw_dir}\n")

    print("--- Video Metadata ---")
    for vpath in videos:
        cap, meta = loader.load_video(vpath)
        cap.release()
        meta = loader.validate_video(meta)
        print(meta)
        if not meta.is_valid:
            print(f"  Errors: {meta.validation_errors}")

    print("\n--- Frame Extraction ---")
    extractor = FrameExtractor()
    results = batch_process_videos(raw_dir)
    summary = extractor.get_batch_summary(results)

    print(json.dumps(summary, indent=2))
    for res in results:
        print(res)


if __name__ == "__main__":
    main()
