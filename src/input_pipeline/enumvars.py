from enum import Enum
from mediapipe.tasks.python.vision import RunningMode


class Directory(Enum):
    VIDEO = "../../data/raw_videos"
    VIDEO_OUTPUT = "../../data/processed_videos"
    POSE_DATA_NPY = "../../data/posedata_output/npy"
    POSE_DATA_CSV = "../../data/posedata_output/csv"
    POSE_MODEL= "../model/pose_task/pose_landmarker_lite.task"
    POSE_FULL_MODEL= "../model/pose_task/pose_landmarker_full.task"
    POSE_HEAVY_MODEL= "../model/pose_task/pose_landmarker_heavy.task"

class Pose_config(Enum):
    STATIC_IMAGE_MODE = False
    MODEL_COMPLEXITY = 1
    SEGMENT_MASK = False
    MIN_DETETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    VISION_RUNNING_MODE = RunningMode.VIDEO
