import cv2 as cv 
import numpy as np 

def sift(img, harris_corners=None):
    print("harris_corners shape: " + str(harris_corners.shape))
    print("harris_corners: " + str(harris_corners))
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    print("img shape: " + str(img.shape))
    key_points = [cv.KeyPoint(x=int(i), y=int(j), size=10, octave=0) for (i, j) in harris_corners]
    sift = cv.SIFT_create()
    _, descriptors = sift.compute(img, key_points)
    return key_points, descriptors
