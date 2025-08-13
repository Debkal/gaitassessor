from enum import Enum


class Directory(Enum):
    VIDEO = "../../data/raw_videos"
    VIDEOOUTPUT = "../../data/processed_videos"
    POSEDATANPY = "../../data/posedata_output/npy"
    POSEDATACSV = "../../data/posedata_output/csv"


class Pose_config(Enum):
    STATIC_IMAGE_MODE = False
    MODEL_COMPLEXITY = 1
    SEGMENT_MASK = False
    MIN_DETETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
