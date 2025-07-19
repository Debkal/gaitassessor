from enum import Enum

class Directory(Enum):
    VIDEO = "/data/raw_video"
    POSEDATANPY ="/data/posedata_output/npy"
    POSEDATACSV ="/data/posedata_output/csv"

print(Directory.VIDEO.value)
