import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.motion_features import MotionFeatures
from feature_engineering.angle_features import AngleFeatures
from feature_engineering.symmetry_features import SymmetryFeatures
from feature_engineering.trajectory_features import TrajectoryFeatures


# ==========================================================
# Load Sample Sequence
# ==========================================================

sample = next(
    Path("data/processed/kinect").rglob("keypoints.npz")
)

print("Loading:")
print(sample)

data = FeatureExtractor.load_keypoints(sample)

kp = data["keypoints"]

print("\nShape:", kp.shape)
print("Frames:", kp.shape[0])
print("Joints:", kp.shape[1])
print("Coordinates:", kp.shape[2])

# ==========================================================
# Extract Features
# ==========================================================

motion = MotionFeatures.extract(kp)

angles = AngleFeatures.extract(kp)

symmetry = SymmetryFeatures.extract(kp)

trajectory = TrajectoryFeatures.extract(kp)

# ==========================================================
# Motion Features
# ==========================================================

print("\n=== Motion Features ===\n")

for k, v in motion.items():
    print(f"{k}: {v}")

# ==========================================================
# Angle Features
# ==========================================================

print("\n=== Angle Features ===\n")

for k, v in angles.items():
    print(f"{k}: {v}")

# ==========================================================
# Symmetry Features
# ==========================================================

print("\n=== Symmetry Features ===\n")

for k, v in symmetry.items():
    print(f"{k}: {v}")

# ==========================================================
# Trajectory Features
# ==========================================================

print("\n=== Trajectory Features ===\n")

for k, v in trajectory.items():
    print(f"{k}: {v}")

# ==========================================================
# Summary
# ==========================================================

total_features = (
    len(motion)
    + len(angles)
    + len(symmetry)
    + len(trajectory)
)

print("\n===================================")
print("PHASE 3 FEATURE EXTRACTION SUMMARY")
print("===================================")

print(f"Motion Features      : {len(motion)}")
print(f"Angle Features       : {len(angles)}")
print(f"Symmetry Features    : {len(symmetry)}")
print(f"Trajectory Features  : {len(trajectory)}")

print("-----------------------------------")
print(f"Total Features       : {total_features}")
print("===================================")