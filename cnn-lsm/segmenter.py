"""
Image segmenter using Level Set Method given the initial bounding box.
"""
import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import op 
from skimage import color
from scipy.ndimage.filters import gaussian_gradient_magnitude

#默认初始轮廓为整张图像的边界
def default_phi(x):
    phi = np.ones(x.shape[:2])
    phi[5:-5, 5:-5] = -1.
    return phi


#由预测目标边界框构造相应的初始水平集函数phi
def phi_from_bbox(img, bbox):
    xmin, ymin, xmax, ymax = bbox
    h, w = img.shape[:2]

    xmin = int(round(xmin * w))
    xmax = int(round(xmax * w))
    ymin = int(round(ymin * h))
    ymax = int(round(ymax * h))

    phi = np.ones(img.shape[:2])
    phi[ymin:ymax, xmin:xmax] = -1

    return phi


#边缘指示函数g
def stopping_fun(x, alpha):
    return 1. / (1. + alpha * op.norm(op.grad(x))**2)


#GAC分割模型，返回预测分割蒙版
def levelset_segment(img, phi=None, dt=1, v=1, sigma=1, alpha=1, n_iter=80, print_after=None):
    img_ori = img.copy()
    img = color.rgb2gray(img)

    img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)

    g = stopping_fun(img_smooth, alpha)
    dg = op.grad(g)

    if phi is None:
        phi = default_phi(img)

    for i in range(n_iter):
        dphi = op.grad(phi)
        dphi_norm = op.norm(dphi)
        kappa = op.curvature(phi)

        smoothing = g * kappa * dphi_norm #平滑项
        balloon = g * dphi_norm * v #气球力
        attachment = op.dot(dphi, dg) #“吸引”力

        dphi_t = smoothing + balloon + attachment

        # Solve level set geodesic equation PDE
        phi = phi + dt * dphi_t

        if print_after is not None and i != 0 and i % print_after == 0:
            plt.imshow(img_ori, cmap='Greys_r')
            plt.contour(phi, 0, colors='r', linewidths=[3])
            plt.draw()
            plt.show()

    if print_after is not None:
        #fig,ax=plt.subplots(figsize=(3,3))
        #ax.imshow(img_ori,cmap='Greys_r')
        #ax.contour(phi,0,colors='red')
        #plt.savefig('2.png')
        mask=phi<0
        plt.imsave('2.png',mask)

    return (phi < 0)


