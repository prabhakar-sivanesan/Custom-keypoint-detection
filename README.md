# Keypoint detection on custom dataset

## Introduction

Most of the keypoint detection model and repositories are trained on [COCO](https://cocodataset.org/#keypoints-2020) or [MPII](http://human-pose.mpi-inf.mpg.de/#overview) human pose dataset or facial keypoints. There were no tangible guide to train a keypoint detection model on custom dataset other than human pose or facial keypoints.  
And hence this repository will primarily focus on keypoint detection on custom dataset using [Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Here we have used a combination of [Centernet](https://arxiv.org/abs/1904.07850)-[hourglass](https://arxiv.org/abs/1603.06937) network therefore the model can provide both bounding boxes and keypoint data as an output during inference.  

## Table of contents

- Preparing dataset
  - Data collection
  - Annotation
  - Processing dataset
- Model preparation
  - Pretrained model
  - Parameter changes in pipeline.config
  - Creating label map
- Training
- Inference


## Preparing dataset

### Data collection

### Annotation

### Processing dataset

## Model preparation

### Pretrained model

### Parameter changes in pipeline.config

### Creating label map

## Training

## Inference
