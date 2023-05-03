import cv2 as cv 
import numpy as np
from hough_lines_acc import hough_lines_acc

def hough_peaks(H,numpeaks):
    peaks = np.zeros((numpeaks,2))
    H_new = H
    for i in range(numpeaks):
        peaks_idx = np.unravel_index(H_new.argmax(), H_new.shape)
        rho_idx = peaks_idx[0]
        theta_idx = peaks_idx[1]
        peaks[i][0] = rho_idx
        peaks[i][1] = theta_idx
        H_new[rho_idx][theta_idx] = 0
    return peaks

#BW = cv.imread("./output/ps2-1-a-1.png")
#BW = cv.cvtColor(BW,cv.COLOR_BGR2GRAY)
#H, theta, rho = hough_lines_acc(BW)
#peaks = hough_peaks(H,10)

#H_out = cv.convertScaleAbs(H, alpha=(255.0/H.max()))
#H_out = cv.cvtColor(H_out,cv.COLOR_GRAY2RGB)
#for i in range(peaks.shape[0]):
    #r = int(peaks[i][0])
    #t = int(peaks[i][1])
    #im_out = cv.circle(H_out,(t,r),radius=1,color=(0,0,255),thickness=-1)
#cv.imwrite("./output/ps2-2-b-1.png",im_out)
#print("Peaks: " + str(peaks))


