# CNN-LSM
The code for my paper: Research on Image Segmentation Method Based on Level Set and Deep Learning.

### Prerequisite

1. Pascal VOC 2012 dataset, download from: <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit>
2. Pre-trained ResNet18 model: download from https://download.pytorch.org/models/resnet18-5c106cde.pth, then put it in the main directory.
3. CNN model that has been trained: download from https://drive.google.com/file/d/14RRd_ZW7fshunrMYlVONTMJEycq58iV4/view?usp=sharing ,then put it in `data/model_resnet_singleobj`.

### Setup

1. Install `miniconda`
2. Do `conda create -n pytorch python=3.8`
3. Enter the env `conda activate pytorch `
4. install torch,torchvision ,torchaudio and other necessary packages, we will run `cnn-lsm/__init__.py` and `cnn-lsm/__init__1.py` in the `pytorch` env.
5. Do `conda create -n tf python=3.7`
6. enter the env `conda activate tf`
7. Install TensorFlow, Keras and other necessary packages, we will run `experiment_localizer.py` and `experiment.py` in the `tf` env.

### Reproducing experimental results
1. Please ensure that you have downloaded Pascal VOC 2012 dataset, Pre-trained ResNet18 model and CNN model(as we have mentioned above).
2. First, enter the env `pytorch`, run `python cnn-lsm/__init__.py` and `python cnn-lsm/__init__1.py` to get `data/features` and `data/labels`.
3. Second, enter the env `tf`, run `python experiment.py`.

### Use your own dataset
1. First, update `data/dataset` with your own dataset
2. Second, enter the env `pytorch`, run `python cnn-lsm/__init__.py` and `python cnn-lsm/__init__1.py` to get `data/features` and `data/labels`
3. Then, enter the env `tf`, run `python experiment_localizer.py` to train CNN model
4. Run `python experiment.py` to test your CNN model
