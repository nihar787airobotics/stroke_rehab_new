import numpy as np


class AngleFeatures:

    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculate angle ABC in degrees.

        Parameters:
            a, b, c : np.ndarray (3,)
                Three 3D points.

        Returns:
            float
                Angle ABC in degrees.
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
        Extract elbow and shoulder angle features.

        Input:
            keypoints -> (T, 25, 3)

        Output:
            dict of angle-based features
        """

        # ==========================
        # ELBOW ANGLES
        # ==========================

        left_elbow = []
        right_elbow = []

        # ==========================
        # SHOULDER ANGLES
        # ==========================

        left_shoulder = []
        right_shoulder = []

        for frame in keypoints:

            # --------------------------------
            # LEFT ELBOW
            # ShoulderLeft -> ElbowLeft -> WristLeft
            # --------------------------------

            left_elbow.append(
                AngleFeatures.calculate_angle(
                    frame[4],   # ShoulderLeft
                    frame[5],   # ElbowLeft
                    frame[6]    # WristLeft
                )
            )

            # --------------------------------
            # RIGHT ELBOW
            # ShoulderRight -> ElbowRight -> WristRight
            # --------------------------------

            right_elbow.append(
                AngleFeatures.calculate_angle(
                    frame[8],   # ShoulderRight
                    frame[9],   # ElbowRight
                    frame[10]   # WristRight
                )
            )

            # --------------------------------
            # LEFT SHOULDER
            # SpineShoulder -> ShoulderLeft -> ElbowLeft
            # --------------------------------

            left_shoulder.append(
                AngleFeatures.calculate_angle(
                    frame[20],  # SpineShoulder
                    frame[4],   # ShoulderLeft
                    frame[5]    # ElbowLeft
                )
            )

            # --------------------------------
            # RIGHT SHOULDER
            # SpineShoulder -> ShoulderRight -> ElbowRight
            # --------------------------------

            right_shoulder.append(
                AngleFeatures.calculate_angle(
                    frame[20],  # SpineShoulder
                    frame[8],   # ShoulderRight
                    frame[9]    # ElbowRight
                )
            )

        left_elbow = np.array(left_elbow)
        right_elbow = np.array(right_elbow)

        left_shoulder = np.array(left_shoulder)
        right_shoulder = np.array(right_shoulder)

        features = {

            # ==================================
            # LEFT ELBOW
            # ==================================

            "left_elbow_mean":
                float(np.mean(left_elbow)),

            "left_elbow_max":
                float(np.max(left_elbow)),

            "left_elbow_min":
                float(np.min(left_elbow)),

            "left_elbow_rom":
                float(
                    np.max(left_elbow)
                    -
                    np.min(left_elbow)
                ),

            # ==================================
            # RIGHT ELBOW
            # ==================================

            "right_elbow_mean":
                float(np.mean(right_elbow)),

            "right_elbow_max":
                float(np.max(right_elbow)),

            "right_elbow_min":
                float(np.min(right_elbow)),

            "right_elbow_rom":
                float(
                    np.max(right_elbow)
                    -
                    np.min(right_elbow)
                ),

            # ==================================
            # ELBOW SYMMETRY
            # ==================================

            "elbow_symmetry":
                float(
                    abs(
                        np.mean(left_elbow)
                        -
                        np.mean(right_elbow)
                    )
                ),

            # ==================================
            # LEFT SHOULDER
            # ==================================

            "left_shoulder_mean":
                float(np.mean(left_shoulder)),

            "left_shoulder_max":
                float(np.max(left_shoulder)),

            "left_shoulder_min":
                float(np.min(left_shoulder)),

            "left_shoulder_rom":
                float(
                    np.max(left_shoulder)
                    -
                    np.min(left_shoulder)
                ),

            # ==================================
            # RIGHT SHOULDER
            # ==================================

            "right_shoulder_mean":
                float(np.mean(right_shoulder)),

            "right_shoulder_max":
                float(np.max(right_shoulder)),

            "right_shoulder_min":
                float(np.min(right_shoulder)),

            "right_shoulder_rom":
                float(
                    np.max(right_shoulder)
                    -
                    np.min(right_shoulder)
                ),

            # ==================================
            # SHOULDER SYMMETRY
            # ==================================

            "shoulder_symmetry":
                float(
                    abs(
                        np.mean(left_shoulder)
                        -
                        np.mean(right_shoulder)
                    )
                ),
        }

        return features