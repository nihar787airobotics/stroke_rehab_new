import numpy as np


class AngleFeatures:

    @staticmethod
    def calculate_angle(a, b, c):
        """
        Angle ABC in degrees
        """

        ba = a - b
        bc = c - b

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba == 0 or norm_bc == 0:
            return 0.0

        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)

        cosine = np.clip(cosine, -1.0, 1.0)

        angle = np.degrees(
            np.arccos(cosine)
        )

        return float(angle)

    @staticmethod
    def extract(keypoints):
        """
        keypoints shape:
        (T, 25, 3)
        """

        left_elbow = []
        right_elbow = []

        for frame in keypoints:

            # LEFT ELBOW
            left_elbow.append(
                AngleFeatures.calculate_angle(
                    frame[4],   # ShoulderLeft
                    frame[5],   # ElbowLeft
                    frame[6]    # WristLeft
                )
            )

            # RIGHT ELBOW
            right_elbow.append(
                AngleFeatures.calculate_angle(
                    frame[8],   # ShoulderRight
                    frame[9],   # ElbowRight
                    frame[10]   # WristRight
                )
            )

        left_elbow = np.array(left_elbow)
        right_elbow = np.array(right_elbow)

        features = {

            "left_elbow_mean":
                float(np.mean(left_elbow)),

            "left_elbow_max":
                float(np.max(left_elbow)),

            "left_elbow_min":
                float(np.min(left_elbow)),

            "left_elbow_rom":
                float(
                    np.max(left_elbow) -
                    np.min(left_elbow)
                ),

            "right_elbow_mean":
                float(np.mean(right_elbow)),

            "right_elbow_max":
                float(np.max(right_elbow)),

            "right_elbow_min":
                float(np.min(right_elbow)),

            "right_elbow_rom":
                float(
                    np.max(right_elbow) -
                    np.min(right_elbow)
                ),

            "elbow_symmetry":
                float(
                    abs(
                        np.mean(left_elbow) -
                        np.mean(right_elbow)
                    )
                ),
        }

        return features