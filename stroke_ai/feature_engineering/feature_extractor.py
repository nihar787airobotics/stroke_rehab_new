from pathlib import Path
import numpy as np


class FeatureExtractor:
    """
    Base feature extraction utilities.
    Every Phase 3 module will inherit from this.
    """

    @staticmethod
    def load_keypoints(npz_path):
        data = np.load(npz_path, allow_pickle=True)

        return {
            "keypoints": data["keypoints"],
            "frame_indices": data["frame_indices"],
            "joint_names": data["joint_names"],
        }

    @staticmethod
    def sequence_length(keypoints):
        return keypoints.shape[0]

    @staticmethod
    def num_joints(keypoints):
        return keypoints.shape[1]

    @staticmethod
    def coordinate_dims(keypoints):
        return keypoints.shape[2]

    @staticmethod
    def flatten_sequence(keypoints):
        return keypoints.reshape(keypoints.shape[0], -1)

    @staticmethod
    def validate_keypoints(keypoints):
        if len(keypoints.shape) != 3:
            raise ValueError(
                f"Expected (T,J,C), got {keypoints.shape}"
            )

        if np.isnan(keypoints).any():
            raise ValueError("NaN values detected")

        return True