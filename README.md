# CNN-LSM
The code for my paper: Research on Image Segmentation Method Based on Level Set and Deep Learning.

### Prerequisite

1. Pascal VOC 2012 dataset, download from: <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit>
2. Pre-trained ResNet18 model: download from https://download.pytorch.org/models/resnet18-5c106cde.pth, then put it in the main directory.
3. CNN model that has been trained: download from 

### Setup

1. Install `miniconda`
2. Do `conda env create`
3. Enter the env `source activate cnn-levelset`
4. Install [TensorFlow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html), or for CPU, run `chmod +x tools/setup_tf.sh && ./setup_tf.sh`
5. Run `python experiment_{localizer|segmenter}.py`
4. Install [TensorFlow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html)
  1. For TF installation without GPU, run `chmod +x tools/tf_setup.sh && sh tools/tf_setup.sh`
5. Download Pascal VOC 2012 dataset, from: <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit>

### Reproducing experimental results

1. Download dataset that is being used for the paper. [Download from here](https://drive.google.com/open?id=0BzFf_WMmDYN8dUdYZE9iMEZXS0k), then unzip it in the main project directory. See `data/README.txt` for documentations of these features
2. Change `cnnlevelset/config.py`
3. Run `python experiment.py`
