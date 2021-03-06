################ CNN网络训练 ###############3


from keras.models import load_model
from keras.optimizers import SGD, Adam
from skimage.io import imshow
from pascalvoc_util import PascalVOC
from localizer import Localizer
from generator import pascal_datagen, pascal_datagen_singleobj
import config as cfg

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.python.control_flow_ops = tf

nb_epoch = 60
pascal = PascalVOC(voc_dir=cfg.PASCAL_PATH)

if len(sys.argv) > 1:
    if sys.argv[1] == 'test':
        X_img_test, X_test, y_test = pascal.get_test_data(10, random=True)

        localizer = Localizer(model_path=cfg.MODEL_PATH)
        cls_preds, bbox_preds = localizer.predict(X_test)

        for img, y, cls_pred, bbox_pred in zip(X_img_test, y_test, cls_preds, bbox_preds):
            label = pascal.idx2label[np.argmax(cls_pred)]

            print(label)

            img = img.reshape(224, 224, 3)
            imshow(pascal.draw_bbox(img, bbox_pred))
            plt.show()

    sys.exit(0)

X_train, y_train = pascal.load_features_trainset()

y_cls = y_train[:, :, 0]
y_reg = y_train[:, :, 1:]
idxes = np.argmax(y_cls, axis=1)
y_reg = y_reg[range(y_train.shape[0]), idxes]
y_train = [y_cls, y_reg]

localizer = Localizer()
localizer.train(X_train, y_train, nb_epoch=nb_epoch)
#localizer.model.save("./autodl-tmp/data/model/model_resnet_singleobj.h5")
