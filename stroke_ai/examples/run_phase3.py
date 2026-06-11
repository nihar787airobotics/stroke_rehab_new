import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.motion_features import MotionFeatures

sample = next(
    Path("data/processed/kinect").rglob("keypoints.npz")
)

print("Loading:")
print(sample)

data = FeatureExtractor.load_keypoints(sample)

kp = data["keypoints"]

print("\nShape:", kp.shape)

features = MotionFeatures.extract(kp)

print("\nMotion Features\n")

for k, v in features.items():
    print(f"{k}: {v}")