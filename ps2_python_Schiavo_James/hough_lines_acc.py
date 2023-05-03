import cv2 as cv 
import numpy as np
import math

def hough_lines_acc(BW):
    theta = np.deg2rad(np.arange(-90,90))
    diagonal_size = math.sqrt(BW.shape[0]**2 + BW.shape[1]**2)
    print("diag: " + str(diagonal_size))
    rho = np.linspace(-diagonal_size,diagonal_size,2*diagonal_size + 1)
    print("rho: " + str(rho))
    H = np.zeros((rho.shape[0],theta.shape[0]))
    print("H shape: "  +str(H.shape))
    #print("theta: " + str(theta))
    for i in range(BW.shape[0]):
        #y-cord = i
        for j in range(BW.shape[1]):
            #x-cord = j
            #check if it's an edge pixel
            if BW[i][j] > 0:
                for t in range(theta.shape[0]):
                    #d = rho
                    r = j*np.cos(theta[t]) + i*np.sin(theta[t])
                    #voting
                    H[int(r + diagonal_size)][t] += 1
    """"
    print("H: " + str(H))
    result = np.where(H == np.amax(H))
    d_max = result[0]
    theta_max = result[1]
    print("d*: " + str(d_max))
    print("theta*: " + str(theta_max))
    """
    return H, theta, rho


#BW = cv.imread("./output/ps2-1-a-1.png")
#print("BW shape: " + str(BW.shape))'
#BW = cv.cvtColor(BW,cv.COLOR_BGR2GRAY)
#print("BW shape: " + str(BW.shape))
#H, theta, rho = hough_lines_acc(BW)
#print("Final H: " + str(H))
#print("Final theta: " + str(theta))
#print("Final rho: " + str(rho))
#im_out = cv.convertScaleAbs(H, alpha=(255.0/H.max()))
#cv.imwrite("./output/ps2-2-a-1.png",im_out)