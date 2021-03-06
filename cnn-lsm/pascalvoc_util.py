############# 定义了一个PascalVOC类，里面包含了之后会用到的许多函数，如计算评估指标等 #################


import numpy as np
import pandas as pd
import xmltodict
from skimage import io, transform, color, draw
from collections import defaultdict


class PascalVOC(object):
    """
    Pascal VOC dataset utility.

    Arguments
    ---------
        voc_dir: string
            Indicating path of the Pascal VOC devkit.
    """

    labels = [
        'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
        'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
    ] #PASCAL VOC2012数据集中的类别
    label2idx = {lbl: idx for idx, lbl in enumerate(labels)}
    idx2label = {idx: lbl for idx, lbl in enumerate(labels)} #映射
    img_size = (224, 224)

    def __init__(self, voc_dir):
        self.voc_dir = voc_dir.rstrip('/')
        self.imageset_dir = './autodl-tmp/data/dataset/'
        self.img_dir = voc_dir + '/JPEGImages/'
        self.bbox_dir = voc_dir + '/Annotations/'
        self.segmentation_dir = voc_dir + '/SegmentationObject/'
        self.feature_dir = './autodl-tmp/data/features/'
        self.label_dir = './autodl-tmp/data/labels/'
        self.feature_prefix = 'resnet_features_'
        self.label_prefix = 'labels_'
        self.trainset_name = 'segmentation_train.txt'
        self.testset_name = 'segmentation_test.txt'
        self.trainset, self.testset = self._load()
        self.mb_idx = 0
        self.mb_test_idx=0
     
    #返回小批量训练集数据及相应标签
    def next_image_minibatch(self, size, random=True, reset=False):
        X = self.trainset

        if random:
            mb = X.sample(size)
        else:
            if reset:
                self.mb_idx = 0

            mb = X[self.mb_idx:self.mb_idx+size]
            self.mb_idx += size

            if self.mb_idx >= X.size:
                self.mb_idx = 0

        return self.load_images(mb), self.load_annotations(mb)
    
    #返回测试集小批量数据及标签
    def next_test_image_minibatch(self, size, random=True, reset=False):
        X = self.testset

        if random:
            mb = X.sample(size)
        else:
            if reset:
                self.mb_test_idx = 0
            
            mb = X[self.mb_test_idx:self.mb_test_idx+size]
            self.mb_test_idx += size

            if self.mb_test_idx >= X.size:
                self.mb_test_idx = 0

        return self.load_images(mb), self.load_annotations(mb)

    def get_test_data(self, size, random=True):
        if random:
            imgs = self.testset.sample(size)
        else:
            imgs = self.testset.head(size)

        X_img = self.load_images(imgs)
        X, y = self.load_features_testset()

        idxes = imgs.index.tolist()
        X, y = X[idxes], y[idxes]

        y_seg = self.load_segmentation_label()

        return X_img, X, y, y_seg
    
    #根据图像名字读取图像数据并resize为224x224大小
    def load_images(self, img_names):
        X = [transform.resize(io.imread(self._img_path(img)), self.img_size) for img in img_names[0]]

        return np.array(X)
    
    #根据图像名字读取标签数据
    def load_annotations(self, img_names):
        y = [np.column_stack(self.get_class_bbox(img)) for img in img_names[0]]

        return np.array(y)
    
    #加载已保存的测试集分割标签
    def load_segmentation_label(self):
        return np.load(self.label_dir + 'labels_segmentation.npy')
    
    #根据图像名字读取测试集的分割标签
    def load_segmentation_label_from_imgs(self, img_names):
        def preprocess(img_name):
            img = io.imread(self.segmentation_dir + '/' + img_name + '.png')
            img = transform.resize(img, self.img_size)
            img = color.rgb2gray(img)
            img = (img != 0)
            return img

        y = [preprocess(img) for img in img_names[0]]

        return np.array(y)

    #在原始图像上画出边界框
    def draw_bbox(self, img, bbox, color=[1, 0, 0], line_width=3):
        xmin, ymin, xmax, ymax = bbox
        h, w = img.shape[:2]

        xmin = int(round(xmin * w))
        xmax = int(round(xmax * w))
        ymin = int(round(ymin * h))
        ymax = int(round(ymax * h))

        img_bbox = np.copy(img)

        img_bbox[ymin-line_width:ymin, xmin-line_width:xmax+line_width] = color
        img_bbox[ymax:ymax+line_width, xmin-line_width:xmax+line_width] = color
        img_bbox[ymin-line_width:ymax+line_width, xmin-line_width:xmin] = color
        img_bbox[ymin-line_width:ymax+line_width, xmax:xmax+line_width] = color

        return img_bbox
    
    # 根据图像名字读取其真实标签数据：类别和边框
    def get_class_bbox(self, img_name):
        with open(self._label_path(img_name), 'r') as f:
            xml = xmltodict.parse(f.read())

        img_size = xml['annotation']['size']
        img_w, img_h = float(img_size['width']), float(img_size['height'])

        objs = xml['annotation']['object']

        if type(objs) is not list:
            objs = [objs]

        clses = np.zeros_like(self.labels, dtype=np.float)
        bboxes = np.zeros(shape=[len(self.labels), 4], dtype=np.float)
        bbox_cls = defaultdict(list)

        for obj in objs:
            idx = self.label2idx[obj['name']]
            clses[idx] = 1

            bndbox = obj['bndbox']
            bbox = (bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax'])
            bbox = self._normalize_bbox(bbox, (img_w, img_h))
            bbox = np.array(bbox, dtype=np.float)
            bbox_cls[idx].append(bbox)

        for k, v in bbox_cls.items():
            sample_idx=np.random.randint(0,len(v))
            bboxes[k]=v[sample_idx]

        return clses, bboxes

    def load_features_trainset(self):
        return self._load_features(self.trainset_name)

    def load_features_testset(self):
        return self._load_features(self.testset_name)
    
    #计算分割准确率的函数
    def segmentation_accuracy(self, y_pred, y_true):
        return np.mean(y_pred == y_true)

    #计算分割精度的函数
    def segmentation_precision(self, y_pred, y_true):
        tp = np.sum(y_true & y_pred)
        fp = np.sum(~y_true & y_pred)
        return tp / (tp + fp + 1e-8)
    
    #计算分割召回率的函数
    def segmentation_recall(self, y_pred, y_true):
        tp = np.sum(y_true & y_pred)
        fn = np.sum(y_true & ~y_pred)
        return tp / (tp + fn + 1e-8)

    #返回分割精度、召回率和F1分数
    def segmentation_prec_rec_f1(self, y_pred, y_true):
        p = self.segmentation_precision(y_pred, y_true)
        r = self.segmentation_recall(y_pred, y_true)
        f1 = 2 * p * r / (p + r + 1e-8)
        return p, r, f1
    
    def _load_features(self, dataset_name):
        dataset_name = dataset_name.split('.')[0]
        X = np.load(self.feature_dir + self.feature_prefix + dataset_name + '.npy')
        y = np.load(self.label_dir + self.label_prefix + dataset_name + '.npy')
        return X, y

    def _load(self):
        train = self._read_dataset(self.imageset_dir + self.trainset_name)
        test = self._read_dataset(self.imageset_dir + self.testset_name)
        return train, test

    def _read_dataset(self, filename):
        return pd.read_csv(filename, header=None, delim_whitespace=True)

    def _img_path(self, img):
        return '{}/{}.jpg'.format(self.img_dir, img)

    def _label_path(self, img):
        return '{}/{}.xml'.format(self.bbox_dir, img)

    def _normalize_bbox(self, bbox, img_dim):
        w, h = img_dim
        xmin, ymin, xmax, ymax = bbox

        def normalize(x, s):
            return float(x) / s

        xmin, ymin = normalize(xmin, w), normalize(ymin, h)
        xmax, ymax = normalize(xmax, w), normalize(ymax, h)

        return [xmin, ymin, xmax, ymax]
