import cv2 as cv 
import numpy as np 
import math

from hough_peaks import hough_peaks

def hough_circles_acc(img,radius):
    #print(img.shape)
    theta = np.deg2rad(np.arange(0,360))
    H = np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #check if it's an edge
            if img[i][j] > 0:
                for l in range(theta.shape[0]):
                    a = j - radius*math.cos(theta[l])
                    b = i + radius*math.sin(theta[l])
                    if a < img.shape[0]:
                        H[int(a)][int(b)] = H[int(a)][int(b)] + 1
    return H

#img_in = cv.imread("./input/ps2-input1.png")
#img_smooth = cv.imread("./output/ps2-4-a-1.png")
#cv.imwrite("./output/ps2-5-a-1.png",img_smooth)
#img_edges = cv.imread("./output/ps2-4-b-1.png")
#cv.imwrite("./output/ps2-5-a-2.png",img_edges)
#img_edges = cv.cvtColor(img_edges,cv.COLOR_BGR2GRAY)
#H = hough_circles_acc(img_edges,20)
#peaks = hough_peaks(H,130)
#H_out = cv.convertScaleAbs(H, alpha=(255.0/H.max()))
#H_out = cv.cvtColor(H_out,cv.COLOR_GRAY2RGB)
#for i in range(peaks.shape[0]):
    #r = int(peaks[i][0])
    #t = int(peaks[i][1])
    #im_out = cv.circle(img_in,((r,t)),radius=20,color=(0,0,255))
#cv.imwrite("./output/ps2-5-a-3.png",im_out)
