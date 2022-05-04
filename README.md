# Keypoint detection training using Tensorflow Object detection API

## Introduction

Most of the keypoint detection model and repositories are trained on [COCO](https://cocodataset.org/#keypoints-2020) or [MPII](http://human-pose.mpi-inf.mpg.de/#overview) human pose dataset or facial keypoints. There were no tangible guide to train a keypoint detection model on custom dataset other than human pose or facial keypoints.  
And hence this repository will primarily focus on keypoint detection training on custom dataset using [Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Here we have used a combination of [Centernet](https://arxiv.org/abs/1904.07850)-[hourglass](https://arxiv.org/abs/1603.06937) network therefore the model can provide both bounding boxes and keypoint data as an output during inference.  
We will be using the transfer learning technique on [centernet-hourglass104](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz) pre-trained model trained on coco dataset to speed-up the training process. 

## Table of contents

- [Preparing dataset](#preparing-dataset)
  - [Data collection](#data-collection)
  - [Annotation](#annotation)
  - [Processing dataset](#processing-dataset)
  - [Generate TFRecord](#generate-tfrecord)
  - [Creating label map](#creating-label-map)
- [Model preparation](#model-preparation)
  - [Pretrained model](#pretrained-model)
  - [Parameter changes in config file](#parameter-changes-in-config-file)
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
Our intention in this project is to detect cutting plier and it's 5 keypoints. Basically you can replace it with any object you need to detect.

### Data collection

Collect all your images and place it into your ```dataset/images``` folder. Make sure all the images are in same format, preferably .jpg/jpeg.

![sample1](utils/image_1.jpg)
![sample2](utils/image_2.jpg)
![sample3](utils/image_3.jpg)
![sample4](utils/image_4.jpg)

### Annotation

The TF2 object detection pipeline requires the dataset for centernet-hourglass network to be annotated on [coco data format](https://cocodataset.org/#format-data) as it's pretrained model is initially trained on COCO dataset.  
I have used [coco-annotator](https://github.com/jsbroks/coco-annotator), a web-based annotation tool that let's you annotate bounding boxes, keypoints, etc which also allows us to automatically download the annotations in coco data format. The [setup and installtion](https://github.com/jsbroks/coco-annotator/wiki/Getting-Started) using docker is super easy, where you can follow these steps to do so.

Run ```docker-compose up``` on terminal from the coco_annotator project directory. Once it's fired up, open ```http://localhost:5000``` on your web browser to go to COCO annotator web interface.

#### Create dataset

Go to *Datasets* tab and create a new dataset. Give a *dataset name* and click **Create Dataset**.  

![create dataset](/utils/create_dataset.gif)  

It will automatically create a folder inside ```coco-annotator/datasets/(dataset_name)```. Now copy all the images from ```Custom-keypoint-detection/dataset/images``` to ```coco-annotator/datasets/(dataset_name)```. This will automatically import all the images into the coco-annotator tool.  

#### Create categories

Next step is to create the **_categories (labels)_** for our dataset to annotate. We create categories only for the objects that needs to be detected using bounding box. We won't create separate categories for keypoints, it will be a subset of the object itself.  
Link **_categories_** to the dataset by the **_Edit_** option.

![create categories](/utils/create_category.gif)  

#### Image annotation

Move to ```Datasets``` tab and click on the images to start annotating. Draw the bounding box first and press ```right arrow``` on the keyboard to annotate keypoints. Follow the same keypoints order while annotating which shows on the right side panel. 

![annotate dataset](utils/annotate_dataset.gif)

After annotating the required images, download the annotation data through ```Datasets -> Download COCO```. All the annotation data will be saved and downloaded as a ```(dataset_name).json``` file.

![dowload dataset](utils/download_coco.jpg)

You can find the [annotation file](dataset/annotations/Plier%20keypoint.json) for our dataset in ```dataset/annotations/``` folder. By looking at the data structure in annotation file you will get any idea how to prepare your own dataset in coco format.

### Processing dataset

As you can see we have all the images annotated and saved in a file. But for training, we need two type of dataset namely, training and validation data. So, in order to prepare your dataset for training, we need to split the dataset into set. 

```
$ python cocosplit.py -h
usage: cocosplit.py [-h] -s SPLIT [--having-annotations]
                    coco_annotations train test

Splits COCO annotations file into training and test sets.

positional arguments:
  coco_annotations      Path to COCO annotations file.
  train                 Where to store COCO training annotations
  test                  Where to store COCO test annotations

optional arguments:
  -h, --help            show this help message and exit
  -s SPLIT              A percentage of a split; a number in (0, 1)
  --having-annotations  Ignore all images without annotations. Keep only these
                        with at least one annotation
```
```
python split_coco_dataset.py -s 0.7  dataset/annotations/Plier\ keypoint.json 
            dataset/annotations/train_data.json dataset/annotations/validation_data.json
```

This script will split the data with 70% into training data and 30% into validation data.  
Thanks to [cocosplit repo from akarazniewicz](https://github.com/akarazniewicz/cocosplit), most part of the script is inspired from his work and just made few adjustments to it.

### Generate TFRecord

Tensorflow object detection api itself provides an example python script to [generate TFRecord](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py) for coco based annotations. But the script is primarily written for coco dataset which contains human pose keypoints. So with few changes to it, we can use it for any custom dataset.  
To do that, edit the ```_COCO_KEYPOINTS_NAMES``` list in [line no 87](https://github.com/tensorflow/models/blob/a3727dae1371fd4b93b80599bdce0e3d57600a25/research/object_detection/dataset_tools/create_coco_tf_record.py#L87) with our keypoints data with the same order it appears on the [annotation file](https://github.com/prabhakar-sivanesan/Custom-keypoint-detection/blob/3d3d729d03160ff60b6a9fa29e2b79e67a35b9fd/dataset/annotations/validation_data.json#L4731). **This is an important step which needs to be verified.**  
Change it from
```
_COCO_KEYPOINT_NAMES = [
    b'nose', b'left_eye', b'right_eye', b'left_ear', b'right_ear',
    b'left_shoulder', b'right_shoulder', b'left_elbow', b'right_elbow',
    b'left_wrist', b'right_wrist', b'left_hip', b'right_hip',
    b'left_knee', b'right_knee', b'left_ankle', b'right_ankle'
]
```
to
```
_COCO_KEYPOINT_NAMES = [
    b'plier_right_handle', b'plier_left_handle', b'plier_middle',
    b'plier_right_head', b'plier_left_head'
]
```
For reference, I have added the ```generate_tfrecord_from_coco.py``` script to the repository. Once this change is done, you can run the script to generate tfrecord for train and validataion dataset. 
Run  
```
!python dataset_tools/create_coco_tf_record.py --train_image_dir "dataset/images" \
--test_image_dir "dataset/images" \
--val_image_dir "dataset/images" \
--train_annotations_file "dataset/annotations/train_data.json" \
--testdev_annotations_file "dataset/annotations/validation_data.json" \
--val_annotations_file "dataset/annotations/validation_data.json" \
--train_keypoint_annotations_file "dataset/annotations/train_data.json" \
--val_keypoint_annotations_file "dataset/annotations/validation_data.json" \
--output_dir "dataset/tfrecord"
```


### Creating label map

The next step is to create label map, which contains the label and ID correlation data for each category we have annotated along with it's keypoint. There is a specific format we have to adhere for TF2 object detection api to rightly recognise this label map.

```
item {
  id: 1
  name: "category_1"
  display_name: "category_1 display name"
  keypoints {
    id: 0
    label: "keypoint 0 name"
  }
  keypoints {
    id: 1
    label: "keypoint 1 name"
  }
  ...
  ...
  keypoints {
    id: 7
    label: "keypoint 7 name"
  }
}
item{
  id: 2
  name: "category_2"
  display_name: "category_2 display name"
  keypoints {
    id: 8
    label: "keypoint 8 name"
  }
  ...
  ...
  keypoints {
    id: 11
    label: "keypoint 11 name"
  }
}
item{
  id: 3
  name: "category_3"
  display_name: "category_3 display name"
}
```
A few things to note:  

1. The name of the keypoints is arbitrary,so some of them share the same label text. What matters is the ID.
2. Keypoint IDs start from 0, unlike item IDs that start from 1
3. Keypoint IDs represent the position of each keypoint in the vector we'll write in the TFRecord file, so here I am implicitly defining the fact that the first 7 keypoints in the vector will be related to category_1 and the last 4 to category_2. Only one of the two groups will be defined for each sample, the other will contain zeroes (and it won't be used during training).

It is not mandatory for all the classes to have a keypoint data annotated with it. There can be other classes too without any keypoint annotaion. Model will only train to detect bounding boxes for those objects. For reference, I have added the label map for our dataset at ```dataset/label_map.pbtxt```.

## Model preparation

### Pretrained model

Download the [centernet-hourglass104 keypoints 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz) pre-trained model from [TF2 model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). You can also use [centernet-hourglass104 keypoints 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_kpts_coco17_tpu-32.tar.gz) pretrained model.  
Extract and place the pre-trained model inside the ```/pretrained_models``` folder. Your folder structure should look like this 
```
Custom-keypoint-detection
|_ dataset
    |_ images (folder to place all the images)
    |_ annotations (folder to place the annotation file)
    |_ tfrecord (folder to place tfrecord)
|_ pretrained_model
    |_ centernet_hg104_512x512_kpts_coco17_tpu-32
        |_checkpoint
        |_saved_model
        |_pipeline.config
```
### Parameter changes in config file

Configuration in the ```pipeline.config``` file has to be edited based on the dataset annotation and training parameters. Firstly, for each category with keypoints defined, we have to add a ```keypoint_estimation_task``` block in the ```center_net``` block. 

```
keypoint_estimation_task {
      task_name: "category_1_keypoint_detection"
      task_loss_weight: 1.0
      loss {
        localization_loss {
          l1_localization_loss {
          }
        }
        classification_loss {
          penalty_reduced_logistic_focal_loss {
            alpha: 2.0
            beta: 4.0
          }
        }
      }
      keypoint_class_name: "category_1"
      keypoint_regression_loss_weight: 0.1
      keypoint_heatmap_loss_weight: 1.0
      keypoint_offset_loss_weight: 1.0
      offset_peak_radius: 3
      per_keypoint_offset: true
    }
```
Point ```keypoint_label_map_path``` parameter to the label map path.  
Change ```fine_tune_checkpoint``` parameter to the pre-trained model checkpoint file.  
Then edit the train and eval input readers to load keypoints adding ```num_keypoints: 5``` (cumulative number of keypoints in all category) to the ```input_reader``` block.  

```
train_input_reader: {
  label_map_path: "path_to_label_map.pbtxt"
  tf_record_input_reader {
    input_path: "path_to_train.record"
  }
  num_keypoints: 5
}

eval_input_reader: {
  label_map_path: "path_to_label_map.pbtxt"
  tf_record_input_reader {
    input_path: "path_validation.record"
  }
  num_keypoints: 5
}
```
Finally, for proper evaluation metrics, you need to add keypoints information to the ```eval_config``` block. ```keypoint_edge``` is for visualization purpose.  

```eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
  num_visualizations: 10
  max_num_boxes_to_visualize: 20
  min_score_threshold: 0.2
  parameterized_metric {
    coco_keypoint_metrics {
      class_label: "category_1"
      keypoint_label_to_sigmas {
        key: "keypoint_0" # add exact keypoint name of keypoint id 0 from labelmap
        value: 5 # defaults value is 5
      }
      keypoint_label_to_sigmas {
        key: "keypoint_1" # add exact keypoint name of keypoint id 1 from labelmap
        value: 5
      }
      keypoint_label_to_sigmas {
        key: "keypoint_2" # add exact keypoint name of keypoint id 2 from labelmap
        value: 5
      }
      keypoint_label_to_sigmas {
        key: "keypoint_3" # add exact keypoint name of keypoint id 3 from labelmap
        value: 5
      }
      keypoint_label_to_sigmas {
        key: "keypoint_4" # add exact keypoint name of keypoint id 4 from labelmap
        value: 5
      }
    }
  }
  parameterized_metric { # keep adding another block for more category.
    coco_keypoint_metrics {
      class_label: "category_2"
      keypoint_label_to_sigmas {
        key: "another_keypint"
        value: 5
      }
      ...
      ...
      keypoint_label_to_sigmas {
        key: "another_keypoint"
        value: 5
      }
    }
  }
  keypoint_edge { # add exact keypoint mapping 
    start: 0 
    end: 1
  }
  keypoint_edge { 
    start: 1
    end: 2
  }
  keypoint_edge { 
    start: 0
    end: 2
  }
  keypoint_edge { 
    start: 2
    end: 3
  }
  keypoint_edge { 
    start: 2
    end: 4
  }
  keypoint_edge { 
    start: 3
    end: 4
  }
}
```

## Training

## Inference

### To-do's
