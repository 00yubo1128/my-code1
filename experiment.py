############# 在测试数据集上测试的代码 ###################


from pascalvoc_util import PascalVOC
from localizer import Localizer
from collections import defaultdict

import config as cfg
import tensorflow as tf
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


tf.python.control_flow_ops = tf

pascal = PascalVOC(cfg.PASCAL_PATH)

X_img_test, X_test, y_test, y_seg = pascal.get_test_data(10000, False)
cls_y_test = y_test[:, :, 0]

N = float(X_img_test.shape[0])

localizer = Localizer(model_path=cfg.MODEL_PATH)

start = time.time()
cls_preds, bbox_preds = localizer.predict(X_test)
end = time.time()
print('CNN time: {:.4f}'.format(end - start))
print('Average: {:.4f}'.format((end - start) / N))

cls_acc = np.mean(np.argmax(cls_preds, axis=1) == np.argmax(cls_y_test, axis=1))
print(cls_acc)

K.clear_session()


from segmenter import *


if len(sys.argv) > 1 and sys.argv[1] == 'show':
    show = True
else:
    show = False

bbox_res, border_res, cnn_res = defaultdict(list), defaultdict(list), defaultdict(list)
i = 0

for img, y, cls_pred, bbox_pred, ys in zip(X_img_test, y_test, cls_preds, bbox_preds, y_seg):
    if (show&(i==25)) :
        label = pascal.idx2label[np.argmax(cls_pred)]

        print(label)

        img = img.reshape(224, 224, 3)
        plt.imsave('{}.png'.format(i),pascal.draw_bbox(img,bbox_pred))
        #plt.imsave('{}.png'.format(i),img)
        plt.imsave('ground_truth.png',ys)

        phi = phi_from_bbox(img, bbox_pred)
        #phi=default_phi(img)
        #mask=(phi<0)
        #plt.imsave('{}.png'.format(i),mask)
        levelset_segment(img, phi=phi, dt=1, v=1, sigma=5, alpha=100000, n_iter=80, print_after=80)
        
        input()
    else:
        start = time.time()
        phi = phi_from_bbox(img, bbox_pred)
        mask = (phi < 0)
        end = time.time()
        bbox_res['time'].append(end - start)
        bbox_res['accuracy'].append(pascal.segmentation_accuracy(mask, ys))
        p, r, f1 = pascal.segmentation_prec_rec_f1(mask, ys)
        bbox_res['precision'].append(p)
        bbox_res['recall'].append(r)
        bbox_res['f1'].append(f1)

        start = time.time()
        phi = default_phi(img)
        mask=levelset_segment(img, phi=phi, dt=1, v=1, sigma=5, alpha=100000, n_iter=80, print_after=None)
        end = time.time()
        border_res['time'].append(end - start)
        border_res['accuracy'].append(pascal.segmentation_accuracy(mask, ys))
        p, r, f1 = pascal.segmentation_prec_rec_f1(mask, ys)
        border_res['precision'].append(p)
        border_res['recall'].append(r)
        border_res['f1'].append(f1)

        start = time.time()
        phi = phi_from_bbox(img, bbox_pred)
        mask=levelset_segment(img, phi=phi, dt=1, v=1, sigma=5, alpha=100000, n_iter=80, print_after=None)
        end = time.time()
        cnn_res['time'].append(end - start)
        cnn_res['accuracy'].append(pascal.segmentation_accuracy(mask, ys))
        p, r, f1 = pascal.segmentation_prec_rec_f1(mask, ys)
        cnn_res['precision'].append(p)
        cnn_res['recall'].append(r)
        cnn_res['f1'].append(f1)

    i += 1
    #if i>25:
        #break
        
    print(i)

if not show:
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print(metric)
        print('----------------')
        print('Bbox: {}'.format(np.mean(bbox_res[metric])))
        print('Border: {}'.format(np.mean(border_res[metric])))
        print('CNN: {}'.format(np.mean(cnn_res[metric])))
        print()

    print('Time')
    print('---------------------')
    print('Bbox: {}'.format(np.mean(bbox_res['time'])))
    print('Border: {}'.format(np.mean(border_res['time'])))
    print('CNN: {}'.format(np.mean(cnn_res['time'])))
    print()
