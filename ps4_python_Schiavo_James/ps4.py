import cv2 as cv 
import numpy as np
import math 
import random

#1a
two_d_points_file = open("./input/pts2d-norm-pic_a.txt",'r')
three_d_points_file = open("./input/pts3d-norm.txt",'r')
two_d_picb_file = open("./input/pts2d-pic_b.txt",'r')
two_d_pica_file = open("./input/pts2d-pic_a.txt",'r')
three_d_file = open("./input/pts3d.txt",'r')

two_d_points = two_d_points_file.readlines()
three_d_points = three_d_points_file.readlines()
print(two_d_points)
print(three_d_points)
print(len(two_d_points))
print(len(three_d_points))

two_d_picb_points = two_d_picb_file.readlines()
two_d_pica_points = two_d_pica_file.readlines()
three_d_points_new = three_d_file.readlines()

norm_2d = np.zeros((20,2))
for i in range(len(two_d_points)):
    string = two_d_points[i]
    new_string = string.split(" ")
    print(new_string)
    u_bool = False
    for j in range(len(new_string)):
        if new_string[j] != '' and u_bool == False:
            u_bool = True
            u = new_string[j]
            print("u: " + str(u))
            norm_2d[i][0] = float(u)
        elif new_string[j] != '' and u_bool == True:
            v = new_string[j]
            print("v: " + str(v))
            norm_2d[i][1] = float(v)

picb_2d = np.zeros((20,2))
for i in range(len(two_d_picb_points)):
    string = two_d_picb_points[i]
    new_string = string.split(" ")
    print(new_string)
    u_bool = False
    for j in range(len(new_string)):
        if new_string[j] != '' and u_bool == False:
            u_bool = True
            u = new_string[j]
            print("u: " + str(u))
            picb_2d[i][0] = float(u)
        elif new_string[j] != '' and u_bool == True:
            v = new_string[j]
            print("v: " + str(v))
            picb_2d[i][1] = float(v)

pica_2d = np.zeros((20,2))
for i in range(len(two_d_pica_points)):
    string = two_d_pica_points[i]
    new_string = string.split(" ")
    print(new_string)
    u_bool = False
    for j in range(len(new_string)):
        if new_string[j] != '' and u_bool == False:
            u_bool = True
            u = new_string[j]
            print("u: " + str(u))
            pica_2d[i][0] = float(u)
        elif new_string[j] != '' and u_bool == True:
            v = new_string[j]
            print("v: " + str(v))
            pica_2d[i][1] = float(v)
            
    
norm_3d = np.zeros((20,3))
for i in range(len(three_d_points)):
    string = three_d_points[i]
    new_string = string.split(" ")
    print(new_string)
    x_bool = False
    y_bool = False
    for j in range(len(new_string)):
        if new_string[j] != '' and x_bool == False and y_bool == False:
            x = new_string[j]
            norm_3d[i][0] = float(x)
            print("x: " + str(x))
            x_bool = True
        elif new_string[j] != '' and x_bool == True and y_bool == False:
            y = new_string[j]
            norm_3d[i][1] = float(y)
            print("y: " + str(y))
            y_bool = True
        elif new_string[j] != '' and x_bool == True and y_bool == True:
            z = new_string[j]
            norm_3d[i][2] = float(z)
            print("z: " + str(z))

new_3d = np.zeros((20,3))
for i in range(len(three_d_points_new)):
    string = three_d_points_new[i]
    new_string = string.split(" ")
    print(new_string)
    x_bool = False
    y_bool = False
    for j in range(len(new_string)):
        if new_string[j] != '' and x_bool == False and y_bool == False:
            x = new_string[j]
            new_3d[i][0] = float(x)
            print("x: " + str(x))
            x_bool = True
        elif new_string[j] != '' and x_bool == True and y_bool == False:
            y = new_string[j]
            new_3d[i][1] = float(y)
            print("y: " + str(y))
            y_bool = True
        elif new_string[j] != '' and x_bool == True and y_bool == True:
            z = new_string[j]
            new_3d[i][2] = float(z)
            print("z: " + str(z))
    
#1a
from least_squares import least_squares
print("norm_2d: " + str(norm_2d))
print("norm_3d: " + str(norm_3d))
M_normA = least_squares(norm_2d,norm_3d)
print("M_normA shape: " + str(M_normA.shape))
print("M_normaA: " + str(M_normA))

#TODO: 
print("norm_3d shape: " + str(norm_3d.shape))
first_xyz = np.ones((4,1))
first_xyz[0] = norm_3d[0][0]
first_xyz[1] = norm_3d[0][1]
first_xyz[2] = norm_3d[0][2]
print("first_xyz: " + str(first_xyz))
first_test_point = np.matmul(M_normA,first_xyz)
print("first_test_point: " + str(first_test_point))
u_first = first_test_point[0]/first_test_point[2]
v_first = first_test_point[1]/first_test_point[2]
print("u_first: " + str(u_first))
print("v_first: " + str(v_first))
last_xyz = np.ones((4,1))
last_xyz[0] = norm_3d[19][0]
last_xyz[1] = norm_3d[19][1]
last_xyz[2] = norm_3d[19][2]
print("last_xyz: " + str(last_xyz))
last_test_point = np.matmul(M_normA,last_xyz)
print("last_test_point: " + str(last_test_point))
u_last = last_test_point[0]/last_test_point[2]
v_last = last_test_point[1]/last_test_point[2]
print("u_last: " + str(u_last))
print("v_last: " + str(v_last))

#CHECK RESIDUALS
dist = 0
for i in range(norm_3d.shape[0]):
    xyz_resid = np.ones((4,1))
    xyz_resid[0] = norm_3d[i][0]
    xyz_resid[1] = norm_3d[i][1]
    xyz_resid[2] = norm_3d[i][2]
    test_point = np.matmul(M_normA,xyz_resid)
    u_resid = test_point[0]/test_point[2]
    #print("u_resid: " + str(u_resid))
    v_resid = test_point[1]/test_point[2]
    #print("v_resid: " + str(v_resid))
    dist += (u_resid - v_resid)**2
    #print("dist: " + str(dist))
residual = math.sqrt(dist)
print("Residual: " + str(residual))



#1b
k = [8,12,16]
point_size = list(range(0,20))
#print(point_size)
k_count = 0
low_residual = 99999
residual1 = np.zeros((10,3))
dist = 0
for p_set in k:
    for i in range(10):
        for p_set in k:
            #2d and 3d 
            new_2d_points = np.zeros((p_set,2))
            new_3d_points = np.zeros((p_set,3))
            two_d_points_index = []
            two_d_points_index.append(random.sample(point_size, p_set))
            #print("two_d_points_index: " + str(two_d_points_index))
            for j in range(p_set):
                #u and v values appended to new 2d points list
                #x, y, and z values appended to new 3d points list
                #print("FAIL: " + str(two_d_points_index[0][j]))
                new_2d_points[j][0] = norm_2d[two_d_points_index[0][j]][0]
                new_2d_points[j][1] = norm_2d[two_d_points_index[0][j]][1]
                new_3d_points[j][0] = norm_3d[two_d_points_index[0][j]][0]
                new_3d_points[j][1] = norm_3d[two_d_points_index[0][j]][1]
                new_3d_points[j][2] = norm_3d[two_d_points_index[0][j]][2]
            M = least_squares(new_2d_points,new_3d_points)
            #print("M: " + str(M))
            for r in range(4):
                xyz_resid1 = np.ones((4,1))
                xyz_resid1[0] = new_3d_points[r][0]
                xyz_resid1[1] = new_3d_points[r][1]
                xyz_resid1[2] = new_3d_points[r][2]
                test_point1 = np.matmul(M,xyz_resid1)
                u_resid1 = test_point1[0]/test_point1[2]
                v_resid1 = test_point1[1]/test_point1[2]
                dist += (u_resid1 - v_resid1)**2
            residual1[i][k_count] = math.sqrt(dist)
            #print("i: " + str(i))
            #print("k_count: " + str(k_count))
            #print("residual1: " + str(residual1[i][k_count]))
            if residual1[i][k_count] < low_residual:
                low_residual = residual1[i][k_count]
                M_low = M
    k_count += 1
print("Average residual: " + str(residual1))
print("Best M matrix: " + str(M_low))
print(M_low.shape)

#1c
C = -np.matmul(np.linalg.inv(M_low[:, 0:3]), M_low[:, 3])
print("C: " + str(C))

#2a
from least_squares2 import least_squares as least_sq
F_lst_sq = least_sq(pica_2d,picb_2d)
print("FINAL F: " + str(F_lst_sq))

#2b
#TODO: 
u, s, vh = np.linalg.svd(F_lst_sq)
s[s.argmin()] = 0
F = np.matmul(np.matmul(u, np.diag(s)), vh)
print("Fundamental matrix F: " + str(F))

def get_epipolar_lines(f, pts_b, img, t=None):
    pts_b_t = np.append(np.asarray(pts_b), np.ones((len(pts_b), 1)), axis=1).T
    l_l = np.cross([0, 0, 1], [img.shape[0], 0, 1])
    l_r = np.cross([0, img.shape[1], 1], [img.shape[0], img.shape[1], 1])
    l_a = np.matmul(f.T, pts_b_t)
    if t is not None:
        l_a = np.matmul(t.T, l_a)
    skew = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    p_l = np.matmul(skew(l_l).T, l_a).T
    p_r = np.matmul(skew(l_r).T, l_a).T
    p_l = np.asarray([(int(x[0] / x[2]), int(x[1] / x[2])) for x in p_l])
    p_r = np.asarray([(int(x[0] / x[2]), int(x[1] / x[2])) for x in p_r])
    for i in range(p_l.shape[0]):
        start_point = (p_l[i][0],p_l[i][1])
        end_point = (p_r[i][0],p_r[i][1])
        img = cv.line(img,start_point,end_point,color=(255,0,0),thickness=1)
    return img
pic_a = cv.imread("./input/pic_a.jpg")
lines_a = get_epipolar_lines(F_lst_sq,picb_2d,pic_a)
cv.imwrite("./output/ps4-2-c-1.png",lines_a)
pic_b = cv.imread("./input/pic_b.jpg")
lines_b = get_epipolar_lines(F_lst_sq,pica_2d,pic_b)
cv.imwrite("./output/ps4-2-c-2.png",lines_b)


