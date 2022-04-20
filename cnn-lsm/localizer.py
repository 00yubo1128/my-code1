######### 编写了CNN网络模型的架构 ############ 


from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.regularizers import l2
from config import *

import keras.backend as K
import tensorflow as tf
import numpy as np


def smooth_l1(x):
    x = tf.abs(x)

    x = tf.where(
        tf.less(x, 1),
        tf.multiply(tf.square(x), 0.5),
        tf.subtract(x, 0.5)
    )

    x = tf.reshape(x, shape=[-1, 4])
    x = tf.reduce_sum(x, 1)

    return x

#定义回归分支上的损失函数
def reg_loss(y_true, y_pred):  
    return smooth_l1(y_true - y_pred)

#定义学习率调节器
def scheduler(epoch):
    if 0 <= epoch < 20:
        return 1e-3

    if 20 <= epoch < 35:
        return 1e-4

    if 35 <= epoch < 55:
        return 1e-5

    return 1e-6

#定义一个类用于搭建CNN网络的架构
class Localizer(object):

    custom_objs = {'reg_loss': reg_loss}

    def __init__(self, model_path=None):
        if model_path is not None:
            self.model = self.load_model(model_path)
        else:
            # ResNet18 last conv features
            inputs = Input(shape=(7, 7, 512))
            x = Convolution2D(128, 1, 1)(inputs)
            x = Flatten()(x)

            # Cls head
            h_cls = Dense(256, activation='relu', W_regularizer=l2(l=0.01))(x)
            h_cls = Dropout(p=0.5)(h_cls)
            cls_head = Dense(20, activation='softmax', name='cls')(h_cls)

            # Reg head
            h_reg = Dense(256, activation='relu', W_regularizer=l2(l=0.01))(x)
            h_reg = Dropout(p=0.5)(h_reg)
            reg_head = Dense(4, activation='linear', name='reg')(h_reg)

            # Joint model
            self.model = Model(input=inputs, output=[cls_head, reg_head])
    
    #定义模型的训练函数
    def train(self, X, y, optimizer='adam', nb_epoch=160):
        self.model.compile(optimizer='adam',
                           loss={'cls': 'categorical_crossentropy', 'reg': reg_loss},
                           loss_weights={'cls': 1., 'reg': 1.},
                           metrics={'cls': 'accuracy'})

        callbacks = [ModelCheckpoint(MODEL_PATH),
                     LearningRateScheduler(scheduler)]

        import pickle

        history = self.model.fit(X, y, batch_size=64, nb_epoch=nb_epoch, callbacks=callbacks)

        with open('./autodl-tmp/data/training_log_DCLSM.bin', 'wb') as f:
            pickle.dump(history.history, f)
        self.model.save("./autodl-tmp/data/model_resnet_singleobj.h5") #最后将训练好的模型保存在data/model_resnet_singleobj.h5文件下

    #定义预测函数
    def predict(self, X): 
        return self.model.predict(X)
    
    #定义模型加载方法
    def load_model(self, model_path):
        return load_model(model_path, custom_objects=self.custom_objs)
