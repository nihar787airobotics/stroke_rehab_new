"""
utils/joint_definitions.py
--------------------------
Joint name constants for Kinect v2 (25) and OpenPose BODY_25 (25).
"""

from __future__ import annotations

from typing import Dict, List

# Kinect v2 — 25 joints (matches .skeleton file order)
KINECT_V2_JOINTS: List[str] = [
    "SpineBase",
    "SpineMid",
    "Neck",
    "Head",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HandRight",
    "HipLeft",
    "KneeLeft",
    "AnkleLeft",
    "FootLeft",
    "HipRight",
    "KneeRight",
    "AnkleRight",
    "FootRight",
    "SpineShoulder",
    "HandTipLeft",
    "ThumbLeft",
    "HandTipRight",
    "ThumbRight",
]

# OpenPose BODY_25
OPENPOSE_BODY25_JOINTS: List[str] = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
]

# Archive hemiplegia dataset — generic 25-joint layout (XYZ rows grouped by frame)
ARCHIVE_GENERIC_JOINTS: List[str] = [f"Joint_{i:02d}" for i in range(25)]

JOINT_COUNT: Dict[str, int] = {
    "kinect": 25,
    "openpose": 25,
    "archive": 25,
}
