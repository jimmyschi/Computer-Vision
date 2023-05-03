from dis import dis
import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt
import random as rand
import torch 
from scipy.spatial.distance import cdist


def kmeans_single(X,K,iters):
    ids = np.zeros((X.shape[0],1))
    means = np.zeros((K,X.shape[1]))  
    # intialize the centers
    for j in range(X.shape[1]):
        minimum = min(X[:,j])
        maximum = max(X[:,j])
        target_range = maximum - minimum
        for k in range(K):
            means[k][j] = rand.uniform(0,target_range)
    for it in range(iters):
        dist = cdist(X,means)
        ssd = 0
        for m in range(dist.shape[0]):
            min_dist = dist[m][0]
            min_cluster = 1
            for n in range(dist.shape[1]):
                if dist[m][n] < min_dist:
                    min_dist = dist[m][n]
                    min_cluster = n + 1
            ids[m] = min_cluster  # now we have assignments for all data samples
        for o in range(X.shape[0]):
            sum = 0
            count = 0
            for p in range(X.shape[1]):
                for q in range(K):
                    k_val = q + 1
                    if ids[o] == k_val:
                        #print("new means")
                        sum = sum + X[o][p]
                        count = count + 1
                        means[q][p] = sum/count              
    ssd_vector = np.min(cdist(X,means),axis=1)
    for p in range(ssd_vector.shape[0]):
        ssd = ssd + ssd_vector[p]
    return ids, means, ssd 
    

#X = np.random.randint(100,size=(15,3))
#K = 3
#iters = 10
#ids,means,ssd = kmeans_single(X,K,iters)
#print(ids)
#print(means)
#print("ssd: " + str(ssd))