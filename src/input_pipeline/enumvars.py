from enum import Enum


class Directory(Enum):
    VIDEO = "../../data/raw_videos"
    POSEDATANPY = "../../data/posedata_output/npy"
    POSEDATACSV = "../../data/posedata_output/csv"


print(Directory.VIDEO.value)
