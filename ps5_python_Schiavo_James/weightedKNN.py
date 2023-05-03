from dis import dis
from enum import unique
from re import X
import sqlite3
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import math 

def weightedKNN(X_train,y_train,X_test,sigma):
    y_predict = np.zeros((X_test.shape[0],1))
    classes = 3
    
    #dist = np.zeros((X_train.shape[0],1))
    sigma_2 = sigma * sigma
    for k in range(X_test.shape[0]):
        weighted_count = np.zeros((classes,1))
        weight = np.zeros((X_train.shape[0],1))
        for j in range(X_train.shape[0]):
            dist = 0
            for m in range(X_train.shape[1]):
                dist += (X_train[j][m] - X_test[k][m])**2
            dist = math.sqrt(dist)
            dist_2 = dist * dist
            weight[j] = math.exp(-dist_2/(sigma_2))
            for i in range(classes):
                target = i + 1
                if y_train[j] == target:
                    weighted_count[i] += 1*weight[j] 
        y_predict[k] = np.argmax(weighted_count)+1
    return y_predict

