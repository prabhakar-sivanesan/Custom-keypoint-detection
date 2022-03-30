# Custom-keypoint-detection

## Introduction

Most of the keypoint detection model and repositories are trained on COCO or MPII human pose dataset or facial keypoints. There were no tangible guide to train a keypoint detection model on custom dataset other than human pose or facial keypoints.
And hence this repository will primarily focus on keypoint detection on custom dataset using Tensorflow object detection API. Here we have used a combination of Centernet-hourglass network therefore the model can provide both bounding boxes and keypoint data as an output during inference.
