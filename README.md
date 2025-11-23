## GaitAssessor - TSN Branch

This branch contains a work in progress for learning machine learning neural networks. This branch specifically uses the Temporal Segmentation Network from [MMAction2](https://github.com/open-mmlab/mmaction2).

### Setup

Anaconda or miniconda for environment handling. This may take some time to configure.

```
git clone https://github.com/open-mmlab/mmaction2.git
conda env create -f environment.yml
conda activate mmaction2_dev
pip install -e .
```

### Running the Project
Make sure your Directory tree is setup as shown at minimum.The data directory can be outside the root folder but, make sure the config file points to your data directory.
```
|--mmaction2
|__|--data/
|_____|--train/
|_____|--val/
|_____|--test/
|--|--configs/
|--|--tests/
|--|--tools/
|--|--work_dirs/
|--|--mmaction/
```
### Configuring the model
Go to configs directory and choose the best fit approach for your data. Than make sure to go to the deepest folder until you run across the "model_name.py". Edit the file to configure the data to train and validate on. Than adjust the epoch parameters to your liking. For example for TSN a config to choose is configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py .
