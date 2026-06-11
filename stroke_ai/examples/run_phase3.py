import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.motion_features import MotionFeatures
from feature_engineering.angle_features import AngleFeatures

sample = next(
    Path("data/processed/kinect").rglob("keypoints.npz")
)

data = FeatureExtractor.load_keypoints(sample)

kp = data["keypoints"]

motion = MotionFeatures.extract(kp)

angles = AngleFeatures.extract(kp)

print("\n=== Motion Features ===\n")

for k, v in motion.items():
    print(f"{k}: {v}")

print("\n=== Angle Features ===\n")

for k, v in angles.items():
    print(f"{k}: {v}")