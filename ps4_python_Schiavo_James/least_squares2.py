import numpy as np
import cv2 as cv 

def least_squares(points_a,points_b):
    f = []
    for (u, v), (u_, v_) in zip(points_a, points_b):
        f.append([u_ * u, u_ * v, u_, v_ * u, v_ * v, v_, u, v, 1])
    F = np.asarray(f)
    print("F shape: " + str(F.shape))
    print("f: " + str(F))
    b = -F[:, -1]
    a = F[:, 0:-1]
    m, res, _, _ = np.linalg.lstsq(a, b, rcond=None)
    f_temp = np.append(m, 1)
    F_final = np.array([[f_temp[0],f_temp[1],f_temp[2]],
                       [f_temp[3],f_temp[4],f_temp[5]],
                       [f_temp[6],f_temp[7],f_temp[8]]])
    return F_final


    
