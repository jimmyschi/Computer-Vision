import numpy as np
import cv2 as cv 
from matplotlib import pyplot as plt
from kmeans_multiple import kmeans_multiple

def Segment_kmeans(im_in,K,iters,R):
    im_in = cv.normalize(im_in.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    im_in = cv.resize(im_in,(100,100))
    #print("im_in: " + str(im_in))
    X = np.reshape(im_in,(im_in.shape[0]*im_in.shape[1],3))
    #print(X)
    ids,means,ssd = kmeans_multiple(X,K,iters,R)
    print("ids: " + str(ids))
    print("means: " + str(means))
    print("ssd: " + str(means))
    print(means)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(K):
                k_val = k + 1
                if ids[i] == k_val:
                    X[i][j] = means[k][j]
    #print("X: " + str(X))
    im_out = cv.convertScaleAbs(X, alpha=(255.0/X.max()))
    im_out = np.reshape(im_out,im_in.shape)
    print("im_out: " + str(im_out))
    return im_out



image1 = cv.imread("./input/ps1_images-1/im1.jpg")
image2 = cv.imread("./input/ps1_images-1/im2.jpg")
image3 = cv.imread("./input/ps1_images-1/im3.png")


K = [3,5,7]
iters = [7,15,30]
R = [5,15,20]
count = 1

for k in K:
    for it in iters:
        for r in R:
            image1_out = Segment_kmeans(image1,k,it,r)
            cv.imwrite("./output/ps0-1-c-" + str(count) + ".jpg",image1_out)
            count = count + 1
            image2_out = Segment_kmeans(image2,k,it,r)
            cv.imwrite("./output/ps0-1-c-" + str(count) + ".jpg",image2_out)
            count = count + 1
            image3_out = Segment_kmeans(image3,k,it,r)
            cv.imwrite("./output/ps0-1-c-" + str(count) + ".png",image3_out)
            count = count + 1

