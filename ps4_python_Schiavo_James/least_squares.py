import cv2 as cv 
import numpy as np 

def least_squares(norm_2d,norm_3d):
    a = []
    for (x, y, z), (u, v) in zip(norm_3d, norm_2d):
        a.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        a.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    A = np.asarray(a)
    #print("A shape: " + str(A.shape))
    #print("A: " + str(A))
    W, V = np.linalg.eig(np.matmul(np.transpose(A), A))
    M = V[:, W.argmin()]
    #print("M!!!!!!!! " + str(M))
    M_normA = np.array([[M[0],M[1],M[2],M[3]],
                       [M[4],M[5],M[6],M[7]],
                       [M[8],M[9],M[10],M[11]]])
    return M_normA
