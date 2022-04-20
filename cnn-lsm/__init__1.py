###### 得到测试集上的分割标签并保存在data/labels/labels_segmentation文件下 ######

import numpy as np
from pascalvoc_util import PascalVOC
import config as cfg

pascal = PascalVOC(cfg.PASCAL_PATH)#创建PASCALVOC实例
segmentation_labels=pascal.load_segmentation_label_from_imgs(pascal.testset)
#print(np.shape(segmentation_labels)[0])

np.save("./autodl-tmp/data/labels/labels_segmentation.npy",segmentation_labels)#保存分割标签