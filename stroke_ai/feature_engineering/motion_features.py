import numpy as np


class MotionFeatures:

    @staticmethod
    def extract(keypoints):
        """
        keypoints shape:
        (T, J, 3)
        """

        features = {}

        # ----------------------------
        # Velocity
        # ----------------------------
        velocity = np.diff(keypoints, axis=0)

        speed = np.linalg.norm(
            velocity,
            axis=2
        )

        features["mean_speed"] = float(np.mean(speed))
        features["max_speed"] = float(np.max(speed))
        features["min_speed"] = float(np.min(speed))
        features["std_speed"] = float(np.std(speed))
        features["median_speed"] = float(np.median(speed))

        # ----------------------------
        # Acceleration
        # ----------------------------
        acceleration = np.diff(
            velocity,
            axis=0
        )

        accel_mag = np.linalg.norm(
            acceleration,
            axis=2
        )

        features["mean_acceleration"] = float(
            np.mean(accel_mag)
        )

        features["max_acceleration"] = float(
            np.max(accel_mag)
        )

        features["std_acceleration"] = float(
            np.std(accel_mag)
        )

        # ----------------------------
        # Jerk
        # ----------------------------
        jerk = np.diff(
            acceleration,
            axis=0
        )

        jerk_mag = np.linalg.norm(
            jerk,
            axis=2
        )

        features["mean_jerk"] = float(
            np.mean(jerk_mag)
        )

        features["max_jerk"] = float(
            np.max(jerk_mag)
        )

        features["std_jerk"] = float(
            np.std(jerk_mag)
        )

        # ----------------------------
        # Range of Motion
        # ----------------------------
        features["x_range"] = float(
            np.max(keypoints[:, :, 0]) -
            np.min(keypoints[:, :, 0])
        )

        features["y_range"] = float(
            np.max(keypoints[:, :, 1]) -
            np.min(keypoints[:, :, 1])
        )

        features["z_range"] = float(
            np.max(keypoints[:, :, 2]) -
            np.min(keypoints[:, :, 2])
        )

        # ----------------------------
        # Path Length
        # ----------------------------
        path_length = np.sum(speed)

        features["path_length"] = float(
            path_length
        )

        # ----------------------------
        # Displacement
        # ----------------------------
        start_center = np.mean(
            keypoints[0],
            axis=0
        )

        end_center = np.mean(
            keypoints[-1],
            axis=0
        )

        displacement = np.linalg.norm(
            end_center - start_center
        )

        features["displacement"] = float(
            displacement
        )

        # ----------------------------
        # Efficiency
        # ----------------------------
        if path_length > 0:
            efficiency = displacement / path_length
        else:
            efficiency = 0.0

        features["efficiency"] = float(
            efficiency
        )

        # ----------------------------
        # Stability
        # ----------------------------
        center = np.mean(
            keypoints,
            axis=1
        )

        center_std = np.std(
            center,
            axis=0
        )

        features["stability_x"] = float(center_std[0])
        features["stability_y"] = float(center_std[1])
        features["stability_z"] = float(center_std[2])

        return features