import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering.feature_extractor import FeatureExtractor

manifest_path = Path("data/processed/kinect")

sample = next(manifest_path.rglob("keypoints.npz"))

print("Loading:")
print(sample)

data = FeatureExtractor.load_keypoints(sample)

kp = data["keypoints"]

print("\nShape:", kp.shape)
print("Frames:", FeatureExtractor.sequence_length(kp))
print("Joints:", FeatureExtractor.num_joints(kp))
print("Coordinates:", FeatureExtractor.coordinate_dims(kp))

FeatureExtractor.validate_keypoints(kp)

print("\nFeatureExtractor working correctly")