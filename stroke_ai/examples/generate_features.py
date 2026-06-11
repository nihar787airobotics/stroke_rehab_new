import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.motion_features import MotionFeatures
from feature_engineering.angle_features import AngleFeatures
from feature_engineering.symmetry_features import SymmetryFeatures
from feature_engineering.trajectory_features import TrajectoryFeatures


manifest_path = Path("data/processed/manifest.csv")

df = pd.read_csv(manifest_path)

rows = []

print(f"Found {len(df)} sequences")

for idx, row in df.iterrows():

    npz_path = Path(row["output_path"])

    try:

        data = FeatureExtractor.load_keypoints(npz_path)

        kp = data["keypoints"]

        motion = MotionFeatures.extract(kp)
        angles = AngleFeatures.extract(kp)
        symmetry = SymmetryFeatures.extract(kp)
        trajectory = TrajectoryFeatures.extract(kp)

        feature_row = {
            "sequence_id": row["sequence_id"],
            "source": row["source"],
            "subject_id": row["subject_id"],
            "motion_name": row["motion_name"],
        }

        feature_row.update(motion)
        feature_row.update(angles)
        feature_row.update(symmetry)
        feature_row.update(trajectory)

        if "performance_outcome" in row:
            feature_row["performance_outcome"] = row["performance_outcome"]

        rows.append(feature_row)

        if (idx + 1) % 50 == 0:
            print(
                f"Processed {idx+1}/{len(df)}"
            )

    except Exception as e:

        print(
            f"Failed: {row['sequence_id']} -> {e}"
        )

features_df = pd.DataFrame(rows)

output_dir = Path("data/features")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_file = output_dir / "features.csv"

features_df.to_csv(
    output_file,
    index=False
)

print("\n================================")
print("FEATURE GENERATION COMPLETE")
print("================================")
print("Rows:", len(features_df))
print("Columns:", len(features_df.columns))
print("Saved:", output_file)