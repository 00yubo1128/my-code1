# Data documentation

## Using pretrained models

Using these dataset and models below is useful to reproducing experimental results.

#### Dataset
`data/dataset` contains all images names from VOC2012 that are being used for this project. It is filtered by considering only images that have single object.

#### Labels
`data/labels` contains pickle files that can be loaded with Numpy. These files are in the shape of `N x 20 x 5` where `N` is the number of data, 20 is the number of classes in VOC2012, and 5 is the label plus bounding box, i.e. first column is the class label, and the remaining 4 columns are for bounding box. You can run `cnn-lsm/__init__.py` to get the `labels_segmentation_train.npy` and `labels_segmentation_test.npy` files, and run `cnn-lsm/__init__1.py` to get the `labels_segmentation.npy` file.

#### Features
`data/features` contains pickle files thhat can be loaded with Numpy. These files are features that extracted from pre-trained ResNet18 (Download from: https://download.pytorch.org/models/resnet18-5c106cde.pth ). You can use the dataset and run `cnn-lsm/__init__.py` to get the `resnet_features_segmentatin_train.npy` and `resnet_features_segmentation_test.npy` files.

#### Models
`data/model_resnet_singleobj` contains the trained CNN that used in my paper, you can download from here: https://drive.google.com/file/d/14RRd_ZW7fshunrMYlVONTMJEycq58iV4/view?usp=sharing .
