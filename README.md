## Gait Assesor

This is a machine learning application that extracts the pose of a person's gait with an input of a video file(.mp4)<br>

### Dependencies

Anaconda installed on your operating system.

<ul>
    <li>Mediapipe</li>
    <li>TensorFlow</li>
    <li>Numpy</li>
</ul>

#### Setup
```
conda env create -f environment.yml
conda activate gaitenv
```
#### Running the project
Make sure your file tree structure is arranged in the following format
```
├── data
│   ├── manifests
│   ├── posedata_output
│   │   ├── csv
│   │   ├── fullclipnpy
│   │   └── npy
│   ├── processed_videos
│   │
│   └── raw_videos
│       ├── walking_doublestep1.mp4
├── environment.yml
├── LICENSE
├── __pycache__
├── README.md
├── requirements.txt
└── src
    ├── input_pipeline
    │   ├── enums.py
    │   ├── enumvars.py
    │   ├── pose_extractor.py
    │   └── visualise_gait.py
    └── model
        ├── pose_task
        │    └── pose_landmarker_heavy.task
        └── gait_assessment
```
This application is a representation of desired feature extraction using Mediapipe. Mediapipe uses a bottom up pose estimation algorithm. See the model card for more information [BlazePose GHUM](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf).
It is a visual and interactive representation of what I will be going for in the future. As I work on gait feature extraction at the lowest level.

##### This is a work in progress and licensed under Apache 2.0
