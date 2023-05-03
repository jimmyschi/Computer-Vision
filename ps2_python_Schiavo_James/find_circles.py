import cv2 as cv 
import numpy as np 

from hough_circles_acc import hough_circles_acc
from hough_peaks import hough_peaks

def find_circles(img_edges, numpeaks, r= np.arange(35,45,1)):
    radii = np.zeros((numpeaks,1))
    for i in range(r.shape[0]):
        H = hough_circles_acc(img_edges,r[i])
        H_out = cv.convertScaleAbs(H, alpha=(255.0/H.max()))
        H_out = cv.cvtColor(H_out,cv.COLOR_GRAY2RGB)
        centers = hough_peaks(H_out,numpeaks)
        radii[i] = r[i]
    return centers,radii


#img_3 = cv.imread("./input/ps2-input1.png")
#filtered_3 = cv.GaussianBlur(img_3,(5,5),5)
#img_edges = cv.Canny(filtered_3,100,200)
#centers, radii = find_circles(img_edges)
#for i in range(centers.shape[0]):
    x = int(centers[i][0])
    y = int(centers[i][1])
    im_out = cv.circle(img_3,((x,y)),radius=radii[i],color=(0,0,255))
    cv.imwrite("./output/ps2-5-b-1.png",im_out)
