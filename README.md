# Keypoint detection on custom dataset

## Introduction

Most of the keypoint detection model and repositories are trained on [COCO](https://cocodataset.org/#keypoints-2020) or [MPII](http://human-pose.mpi-inf.mpg.de/#overview) human pose dataset or facial keypoints. There were no tangible guide to train a keypoint detection model on custom dataset other than human pose or facial keypoints.  
And hence this repository will primarily focus on keypoint detection training on custom dataset using [Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Here we have used a combination of [Centernet](https://arxiv.org/abs/1904.07850)-[hourglass](https://arxiv.org/abs/1603.06937) network therefore the model can provide both bounding boxes and keypoint data as an output during inference.  We will using the transfer learning technique on [centernet-hourglass104](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz) pre-trained model trained on coco dataset to speed-up the training process. 

## Table of contents

- [Preparing dataset](#preparing-dataset)
  - [Data collection](#data-collection)
  - [Annotation](#annotation)
  - [Processing dataset](#processing-dataset)
- [Model preparation](#model-preparation)
  - [Pretrained model](#pretrained-model)
  - [Parameter changes in config file](#parameter-changes-in-config-file)
  - [Creating label map](#creating-label-map)
- [Training](#training)
- [Inference](#inference)


## Preparing dataset

Create a folder structure similar to this order  
```
Custom-keypoint-detection
|_ dataset
    |_ images (folder to place all the images)
    |_ annotations (folder to place the annotation file)
    |_ tfrecord (folder to place tfrecord)
```


### Data collection

Collect all your images and place it into your ```dataset/images``` folder.

### Annotation

### Processing dataset

## Model preparation

### Pretrained model

### Parameter changes in config file

### Creating label map

## Training

## Inference
