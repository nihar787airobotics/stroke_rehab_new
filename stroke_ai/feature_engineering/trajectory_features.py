import numpy as np


class TrajectoryFeatures:

    @staticmethod
    def _path_metrics(points):
        """
        points shape:
        (T,3)
        """

        velocity = np.diff(points, axis=0)

        segment_lengths = np.linalg.norm(
            velocity,
            axis=1
        )

        path_length = np.sum(segment_lengths)

        displacement = np.linalg.norm(
            points[-1] - points[0]
        )

        efficiency = (
            displacement / path_length
            if path_length > 0
            else 0.0
        )

        return (
            float(path_length),
            float(displacement),
            float(efficiency)
        )

    @staticmethod
    def extract(keypoints):

        features = {}

        joints = {
            "left_wrist": 6,
            "right_wrist": 10,
            "left_hand": 7,
            "right_hand": 11
        }

        for name, idx in joints.items():

            points = keypoints[:, idx, :]

            path_length, displacement, efficiency = (
                TrajectoryFeatures._path_metrics(points)
            )

            features[f"{name}_path_length"] = path_length

            features[f"{name}_displacement"] = displacement

            features[f"{name}_efficiency"] = efficiency

        return features