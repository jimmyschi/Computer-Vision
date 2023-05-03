import cv2 as cv 
import numpy as np 


def gaussian_pyramid(img, max_l=np.PINF, up_scaled=False):
    levels = min(int(np.log2(min(img.shape[:2])) + 1), max_l)
    gauss_pyramid = [img]
    subscale = lambda i: cv.pyrUp(cv.pyrDown(i), i.shape) if up_scaled else cv.pyrDown(i)
    [gauss_pyramid.append(subscale(gauss_pyramid[-1])) for i in range(levels - 1)]
    return gauss_pyramid