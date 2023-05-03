import cv2 as cv 
import numpy as np
import math
from hough_lines_acc import hough_lines_acc
from hough_peaks import hough_peaks

def hough_lines_draw(img_in,img_out,peaks,rho,theta):
    for q in range(peaks.shape[0]):
        r = int(peaks[q][0])
        t = int(peaks[q][1])
        if theta[t] != 0:
            x1 = 1
            x2 = img_in.shape[1]
            y1 = (rho[r] - x1*math.cos(theta[t]))/math.sin(theta[t])
            y2 = (rho[r] - x2*math.cos(theta[t]))/math.sin(theta[t])
        else:
            y1 = 1
            y2 = img_in.shape[0]
            x1 = (rho[r] - y1*math.sin(theta[t]))/math.cos(theta[t])
            x2 = (rho[r] - y2*math.sin(theta[t]))/math.cos(theta[t])
        hough_lines = cv.line(img_in,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
        cv.imwrite(img_out,hough_lines)

#img_in = cv.imread("./input/ps2-input0.png")
#BW = cv.imread("./output/ps2-1-a-1.png")
#BW = cv.cvtColor(BW,cv.COLOR_BGR2GRAY)
#H, theta, rho = hough_lines_acc(BW)
#peaks = hough_peaks(H,10)
#hough_lines_draw(img_in,"./output/ps2-2-c-1.png",peaks,rho,theta)
