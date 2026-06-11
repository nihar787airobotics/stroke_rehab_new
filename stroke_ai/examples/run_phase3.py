import numpy as np


class MotionFeatures:

    @staticmethod
    def extract(keypoints):
        """
        keypoints shape:
        (T, J, 3)
        """

        velocity = np.diff(keypoints, axis=0)

        speed = np.linalg.norm(
            velocity,
            axis=2
        )

        features = {
            "mean_speed": float(np.mean(speed)),
            "max_speed": float(np.max(speed)),
            "std_speed": float(np.std(speed)),
        }

        return features