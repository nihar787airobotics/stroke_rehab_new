import numpy as np


class TemporalFeatures:
    """
    Temporal statistics extracted from a skeleton sequence.
    """

    @staticmethod
    def extract(keypoints):
        """
        Input:
            keypoints -> (T, J, 3)

        Returns:
            dict of temporal features
        """

        num_frames = keypoints.shape[0]

        features = {
            "num_frames": float(num_frames),
            "sequence_duration": float(num_frames),
        }

        return features