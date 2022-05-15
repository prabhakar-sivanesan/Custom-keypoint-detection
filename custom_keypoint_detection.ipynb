{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "custom_keypoint_detection.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "machine_shape": "hm"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "##**Custom Keypoint detection using Tensorflow Object detection API**\n",
        "\n",
        "This notebook helps you train a keypoint detection model on custom dataset annotated in coco format. We are using Centernet-hourglass  model.\n",
        "This codebase contains scripts for TF record generation, model training and inference."
      ],
      "metadata": {
        "id": "KuRiew3bYFoH"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Mount Google drive"
      ],
      "metadata": {
        "id": "QqEaX760XyUs"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "MajTHP2VQcEz"
      },
      "outputs": [],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/gdrive', force_remount=True)"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Download Tensorflow model repository into the gdrive workspace"
      ],
      "metadata": {
        "id": "Zodjk7LcY4-_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#and clone the TensorFlow Model Garden repository\n",
        "!git clone https://github.com/tensorflow/models.git\n",
        "\n",
        "#cd into the TensorFlow directory in your Google Drive\n",
        "%cd '/content/gdrive/My Drive/TensorFlow'"
      ],
      "metadata": {
        "id": "bauEop5RQhXi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Installing dependencies"
      ],
      "metadata": {
        "id": "YkxT3lDXZLsa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!apt-get install protobuf-compiler python-lxml python-pil\n",
        "!pip install Cython pandas tf-slim lvis"
      ],
      "metadata": {
        "id": "IsCIWLfBQ6Vn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#reinstall opencv with headless\n",
        "!pip list | grep opencv\n",
        "!pip uninstall opencv-python -y\n",
        "!pip install \"opencv-python-headless<4.3\"\n",
        "!pip list | grep opencv"
      ],
      "metadata": {
        "id": "zQRrga76RyP8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#cd into 'TensorFlow/models/research'\n",
        "%cd '/content/gdrive/My Drive/TensorFlow/models/research/'\n",
        "!protoc object_detection/protos/*.proto --python_out=."
      ],
      "metadata": {
        "id": "G5frJiV2Q8_h"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!cp object_detection/packages/tf2/setup.py .\n",
        "!python -m pip install --use-feature=2020-resolver ."
      ],
      "metadata": {
        "id": "VjH0qRpQRA0a"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Verify installation is done correctly\n",
        "#cd into 'TensorFlow/models/research/object_detection/builders/'\n",
        "%cd '/content/gdrive/My Drive/TensorFlow/models/research/object_detection/builders/'\n",
        "!python model_builder_tf2_test.py\n",
        "from object_detection.utils import label_map_util\n",
        "from object_detection.utils import visualization_utils as viz_utils\n",
        "print('Done')"
      ],
      "metadata": {
        "id": "HnrHAbS-RD7s"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Download pretrained model from TF model garden"
      ],
      "metadata": {
        "id": "qRKkQBpZaNgd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "%cd \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/pretrained_model\"\n",
        "!wget \"http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz\"\n",
        "!tar -xvzf \"centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz\""
      ],
      "metadata": {
        "id": "Lyi8FOwXRJBC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Generate TFrecord from coco annotation dataset\n",
        "\n",
        "This step has to be executed after the dataset is split for training, validation and testing."
      ],
      "metadata": {
        "id": "ptG_okf3bYpO"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "%cd \"/content/gdrive/MyDrive/TensorFlow/models/research/object_detection\"\n",
        "!python dataset_tools/create_coco_tf_record.py --train_image_dir \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/dataset/images\" \\\n",
        "--test_image_dir \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/dataset/images\" \\\n",
        "--val_image_dir \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/dataset/images\" \\\n",
        "--train_annotations_file \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/dataset/annotations/train_data.json\" \\\n",
        "--testdev_annotations_file \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/dataset/annotations/validation_data.json\" \\\n",
        "--val_annotations_file \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/dataset/annotations/validation_data.json\" \\\n",
        "--train_keypoint_annotations_file \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/dataset/annotations/train_data.json\" \\\n",
        "--val_keypoint_annotations_file \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/dataset/annotations/validation_data.json\" \\\n",
        "--output_dir \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/dataset/tfrecord\""
      ],
      "metadata": {
        "id": "4CgjxhpjWkh6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Run Tensorboard\n",
        "Open tensorboard to monitor the training performance."
      ],
      "metadata": {
        "id": "7ngdYnp3bwao"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "%cd '/content/gdrive/MyDrive/projects/custom_keypoint_detection/'\n",
        "#start the Tensorboard\n",
        "#%load_ext tensorboard\n",
        "%reload_ext tensorboard\n",
        "%tensorboard --logdir=output_models/"
      ],
      "metadata": {
        "id": "46W5wwu7dAcE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Model Training"
      ],
      "metadata": {
        "id": "iX5QrZ3xb8A3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "%cd \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/\"\n",
        "!python /content/gdrive/MyDrive/TensorFlow/models/research/object_detection/model_main_tf2.py \\\n",
        "--model_dir \"output_models\" \\\n",
        "--pipeline_config_path \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/dataset/pipeline.config\""
      ],
      "metadata": {
        "id": "pWeaxwJiYaYT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Export saved model\n",
        "Export saved model from the trained checkpoint."
      ],
      "metadata": {
        "id": "u_RmLIunb_y6"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "%cd \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/\"\n",
        "!python /content/gdrive/MyDrive/TensorFlow/models/research/object_detection/exporter_main_v2.py \\\n",
        "--input_type image_tensor \\\n",
        "--pipeline_config_path \"/content/gdrive/MyDrive/projects/custom_keypoint_detection/dataset/pipeline.config\" \\\n",
        "--trained_checkpoint_dir \"output_models\" \\\n",
        "--output_directory \"saved_model\""
      ],
      "metadata": {
        "id": "CWqWfxXvypDk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Run inference"
      ],
      "metadata": {
        "id": "_6a0chQ9cQg2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "%cd \"/content/gdrive/MyDrive/projects/custom_keypoint_detection\"\n",
        "!python inference.py"
      ],
      "metadata": {
        "id": "MUyn6pDr7Huz"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}