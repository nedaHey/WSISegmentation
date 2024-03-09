# WSISegmentation

## Intro

This work aims to perform tissue segmentation from a dataset that includes the seven most common canine cutaneous tumor types.
We propose two semantic segmentation models for histopathology WSI's, which are based on encoder-decoder. The proposed models are a single-resolution UNet [link](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) model as well as a novel multi-resolution network based on the popular UNet, namely HookNet [link](https://www.sciencedirect.com/science/article/pii/S1361841520302541). As expected, training the models with different levels of WSI's could help us identify the appropriate level for learning each cancer type.


## Code
We have three main functionalities:
1. Data preparation
2. Visualization
3. Model: defining model, defining metrics, training

Data-related functionalities (pre-processing, data preparation, and augmentation) are done using classes defined in the `dataset.py` module. This module uses `SlideContainer` defined in `utils.py`.

Model creation, compiling, checkpointing, and tracking is done in the `models` sub-package.

Visualization tools are defined within the `vistools.py` module.

Also, configurations can be found on the `config.py` file.


## Installation

- Install `tensorflow-gpu>=2.0`
- Install `image-classifiers`: `pip install image-classifiers`
- Install `OpenSlide` [Link](https://openslide.org)
- `pip install numpy pandas matplotlib scikit-learn EXCAT-Sync opencv-python albumentations`

## Training

For training, you need to pass 4 arguments: 

1. `dataset_dir` which is `/path/to/the/dataset/directory`
2. `subset_file_path` which is `datasets.xls` file's path
3. `out_dir` directory for saving checkpoints and logs
4. `model_name`

For example, for training the `UNet` model on the 3-class case (cancer, tissue, and background), set the parameters on `config.py` and train using:

`python3 train_unet_cancer.py --dataset_dir {} --subset_file_path {} --out_dir {} --model_name {}`
