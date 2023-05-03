import cv2 as cv 
import numpy as np 


def laplacian_pyramid(img, max_l=np.PINF):
    levels = min(int(np.log2(min(img.shape[:2])) + 1), max_l)
    laplacian_pyramid = [img]
    for i in range(levels - 1):
        im_blur = cv.GaussianBlur(img, (5, 5), 0)
        tmp = im_blur[::2, ::2, ...]
        expand_gauss = cv.GaussianBlur(cv.resize(tmp, laplacian_pyramid[-1].shape[1::-1], interpolation=cv.INTER_LINEAR), (5, 5), 0)
        laplacian_pyramid[-1] = laplacian_pyramid[-1].astype(np.int32) - expand_gauss
        laplacian_pyramid.append(tmp)
    return laplacian_pyramid

