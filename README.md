## Gait Assesor

This is a machine learning model that makes an assessment of a person's gait with an input of a video   file(.mp4)<br>

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
```


This machine learning model is a hybrid model using a Convolutional Neural Network and Recurrent Neural Network
in tandom to execute an assessment of the gait state of any person. It will return a boolean for its overall assessment. True for a gait within normal deviation and False for a gait with irregular deviation. For both cases it will digress into key frames that lead to its assessment. It will respond with a description of the key frame and what could possibly indicate a break down in gait at these key frames.

##### This is a work in progress and licensed under Apache 2.0
