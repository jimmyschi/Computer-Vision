import cv2 as cv 
import numpy as np

from hough_lines_acc import hough_lines_acc
from hough_peaks import hough_peaks
from hough_lines_draw import hough_lines_draw
from hough_circles_acc import hough_circles_acc
from find_circles import find_circles

#1a
img = cv.imread("./input/ps2-input0.png")
img_edges = cv.Canny(img,100,200)
cv.imwrite("./output/ps2-1-a-1.png",img_edges)

#2a
BW = cv.imread("./output/ps2-1-a-1.png")
BW = cv.cvtColor(BW,cv.COLOR_BGR2GRAY)
H0, theta0, rho0 = hough_lines_acc(BW)
im_out0 = cv.convertScaleAbs(H0, alpha=(255.0/H0.max()))
cv.imwrite("./output/ps2-2-a-1.png",im_out0)

#2b
peaks = hough_peaks(H0,10)
H_out0 = cv.convertScaleAbs(H0, alpha=(255.0/H0.max()))
H_out0 = cv.cvtColor(H_out0,cv.COLOR_GRAY2RGB)
for i in range(peaks.shape[0]):
    r0 = int(peaks[i][0])
    t0 = int(peaks[i][1])
    im_out00 = cv.circle(H_out0,(t0,r0),radius=1,color=(0,0,255),thickness=-1)
cv.imwrite("./output/ps2-2-b-1.png",im_out00)

#2c
hough_lines_draw(img,"./output/ps2-2-c-1.png",peaks,rho0,theta0)

#3a
noisy = cv.imread("./input/ps2-input0-noise.png")
filtered = cv.GaussianBlur(noisy,(5,5),5)
cv.imwrite("./output/ps2-3-a-1.png",filtered)
filtered_edges = cv.Canny(filtered,100,200)
cv.imwrite("./output/ps2-3-a-2.png",filtered_edges)

#3b
noisy_edges = cv.Canny(noisy,100,200)
cv.imwrite("./output/ps2-3-b-1.png",noisy_edges)

#3c
H, theta, rho = hough_lines_acc(filtered_edges)
peaks = hough_peaks(H,10)
H_out = cv.convertScaleAbs(H, alpha=(255.0/H.max()))
H_out = cv.cvtColor(H_out,cv.COLOR_GRAY2RGB)
for i in range(peaks.shape[0]):
    r = int(peaks[i][0])
    t = int(peaks[i][1])
    im_out = cv.circle(H_out,(t,r),radius=1,color=(0,0,255),thickness=-1)
cv.imwrite("./output/ps2-3-c-1.png",im_out)
hough_lines_draw(noisy,"./output/ps2-3-c-2.png",peaks,rho,theta)

#4a
img_2 = cv.imread("./input/ps2-input1.png")
img_2 = cv.cvtColor(img_2,cv.COLOR_BGR2GRAY)
filtered_2 = cv.GaussianBlur(img_2,(5,5),5)
cv.imwrite("./output/ps2-4-a-1.png",filtered_2)

#4b
img_edges_2 = cv.Canny(filtered_2,100,200)
cv.imwrite("./output/ps2-4-b-1.png",img_edges_2)

#4c
H_2, theta_2, rho_2 = hough_lines_acc(img_edges_2)
peaks_2 = hough_peaks(H_2,10)
H_out = cv.convertScaleAbs(H_2, alpha=(255.0/H_2.max()))
H_out = cv.cvtColor(H_out,cv.COLOR_GRAY2RGB)
for i in range(peaks_2.shape[0]):
    r = int(peaks_2[i][0])
    t = int(peaks_2[i][1])
    im_out2 = cv.circle(H_out,(t,r),radius=1,color=(0,0,255),thickness=-1)
cv.imwrite("./output/ps2-4-c-1.png",im_out2)
print("hough draw")
img_2_orig = cv.imread("./input/ps2-input1.png")
hough_lines_draw(img_2_orig,"./output/ps2-4-c-2.png",peaks_2,rho_2,theta_2)

#5a
img_in_5a = cv.imread("./input/ps2-input1.png")
img_smooth_5a = cv.imread("./output/ps2-4-a-1.png")
cv.imwrite("./output/ps2-5-a-1.png",img_smooth_5a)
img_edges_5a = cv.imread("./output/ps2-4-b-1.png")
cv.imwrite("./output/ps2-5-a-2.png",img_edges_5a)
img_edges_5a = cv.cvtColor(img_edges_5a,cv.COLOR_BGR2GRAY)
H_5a = hough_circles_acc(img_edges_5a,20)
peaks_5a = hough_peaks(H_5a,130)
H_out_5a = cv.convertScaleAbs(H_5a, alpha=(255.0/H_5a.max()))
H_out_5a = cv.cvtColor(H_out_5a,cv.COLOR_GRAY2RGB)
for i in range(peaks_5a.shape[0]):
    r_5a = int(peaks_5a[i][0])
    t_5a = int(peaks_5a[i][1])
    im_out_5a = cv.circle(img_in_5a,((r_5a,t_5a)),radius=20,color=(0,0,255))
cv.imwrite("./output/ps2-5-a-3.png",im_out_5a)

#5b
img_3 = cv.imread("./input/ps2-input1.png")
filtered_3 = cv.GaussianBlur(img_3,(5,5),5)
img_edges_3 = cv.Canny(filtered_3,100,200)
centers, radii = find_circles(img_edges_3,130)
for i in range(centers.shape[0]):
    x = int(centers[i][0])
    y = int(centers[i][1])
    im_out3 = cv.circle(img_3,((x,y)),radius=radii[i],color=(0,0,255))
    cv.imwrite("./output/ps2-5-b-1.png",im_out3)



#6a
img_3 = cv.imread("./input/ps2-input2.png")
filtered_3 = cv.GaussianBlur(img_3,(5,5),5)
filtered_edges_3 = cv.Canny(filtered_3,100,200)
H3, theta3, rho3 = hough_lines_acc(filtered_edges_3)
peaks_3 = hough_peaks(H3,10)
hough_lines_draw(filtered_3,"./output/ps2-6-a-1.png",peaks_3,rho3,theta3)

#7a
centers2, radii2 = find_circles(filtered_edges_3,130)
for i in range(centers2.shape[0]):
    x2 = int(centers2[i][0])
    y2 = int(centers2[i][1])
    im_out = cv.circle(img_3,((x2,y2)),radius=radii2[i],color=(0,0,255))
    cv.imwrite("./output/ps2-7-a-1.png",im_out)

#8a
img_4 = cv.imread("./input/ps2-input3.png")
filtered_4 = cv.GaussianBlur(img_4,(5,5),5)
filtered_edges_4 = cv.Canny(filtered_4,100,200)
H4, theta4, rho4 = hough_lines_acc(filtered_edges_4)
peaks_4 = hough_peaks(H4,10)
hough_lines_draw(filtered_4,"./output/ps2-8-a-1.png",peaks_4,rho4,theta4)
centers3, radii3 = find_circles(filtered_edges_4,130)
for i in range(centers3.shape[0]):
    x3 = int(centers3[i][0])
    y3 = int(centers3[i][1])
    im_out = cv.circle(img_4,((x3,y3)),radius=radii3[i],color=(0,0,255))
    cv.imwrite("./output/ps2-8-a-1.png",im_out)
