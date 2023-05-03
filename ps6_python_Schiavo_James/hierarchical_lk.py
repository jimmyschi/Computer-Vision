import cv2 as cv 
import numpy as np

from gaussian_pyramid import gaussian_pyramid
from lucas_kanade import lucas_kanade

def hierarchical_lk(img_1, img_2, k_shape=None, max_l=np.PINF, up_scaled=False):
    gauss_1 = gaussian_pyramid(img_1, up_scaled=up_scaled)
    gauss_2 = gaussian_pyramid(img_2, up_scaled=up_scaled)
    max_l = min(len(gauss_1), len(gauss_2), max_l)
    flow = np.zeros(gauss_1[-1].shape[:2] + (2,)).astype(np.float32)
    for i in range(max_l - 1, -1, -1):
        dst_size = gauss_2[i].shape[:2] + (2,)
        flow = 2.0 * cv.GaussianBlur(cv.resize(flow, dst_size[1::-1], interpolation=cv.INTER_LINEAR), (5, 5), 0)
        h, w = flow.shape[:2]
        flow_map = -flow.copy()
        flow_map[:, :, 0] += np.arange(w)
        flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]
        warped = cv.remap(gauss_1[i], flow_map.astype(np.float32), None, cv.INTER_LINEAR)
        flow += lucas_kanade(warped, gauss_2[i], k_shape)
    return flow