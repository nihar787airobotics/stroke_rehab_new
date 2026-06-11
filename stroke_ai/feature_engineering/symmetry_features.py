import numpy as np


class SymmetryFeatures:

    @staticmethod
    def extract(keypoints):
        """
        Kinect 25-joint skeleton
        """

        features = {}

        # =========================
        # Joint Pairs
        # =========================

        pairs = {
            "shoulder": (4, 8),
            "elbow": (5, 9),
            "wrist": (6, 10),
            "hip": (12, 16),
            "knee": (13, 17),
            "ankle": (14, 18)
        }

        for name, (left_idx, right_idx) in pairs.items():

            left = keypoints[:, left_idx, :]
            right = keypoints[:, right_idx, :]

            distance = np.linalg.norm(
                left - right,
                axis=1
            )

            features[f"{name}_symmetry_mean"] = float(
                np.mean(distance)
            )

            features[f"{name}_symmetry_max"] = float(
                np.max(distance)
            )

            features[f"{name}_symmetry_std"] = float(
                np.std(distance)
            )

        return features