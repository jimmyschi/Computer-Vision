from random import gauss
import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from lucas_kanade import lucas_kanade
from lucas_kanade import draw_flow
from gaussian_pyramid import gaussian_pyramid
from laplacian_pyramid import laplacian_pyramid
from hierarchical_lk import hierarchical_lk

#1a
Shift0 = cv.imread("./input/TestSeq/Shift0.png")
ShiftR2 = cv.imread("./input/TestSeq/ShiftR2.png")
ShiftR5U5 = cv.imread("./input/TestSeq/ShiftR5U5.png")
ShiftR10 = cv.imread("./input/TestSeq/ShiftR10.png")
ShiftR20 = cv.imread("./input/TestSeq/ShiftR20.png")
ShiftR40 = cv.imread("./input/TestSeq/ShiftR40.png")

flow_1 = lucas_kanade(Shift0,ShiftR2,(15,15))
print("flow_1 shape 1a lk: " + str(flow_1.shape))
Shift0_flow_1 = draw_flow(Shift0,flow_1)
cv.imwrite("./output/ps6-1-a-1.png",Shift0_flow_1)
flow_2 = lucas_kanade(Shift0,ShiftR5U5,(47,47))
Shift0_flow_2 = draw_flow(Shift0,flow_2)
cv.imwrite("./output/ps6-1-a-2.png",Shift0_flow_2)

#1b
flow_3 = lucas_kanade(Shift0,ShiftR10,(47,47))
Shift0_flow_3 = draw_flow(Shift0,flow_3)
cv.imwrite("./output/ps6-1-b-1.png",Shift0_flow_3)
flow_4 = lucas_kanade(Shift0,ShiftR20,(47,47))
Shift0_flow_4 = draw_flow(Shift0,flow_4)
cv.imwrite("./output/ps6-1-b-2.png",Shift0_flow_4)
flow_5 = lucas_kanade(Shift0,ShiftR40,(47,47))
Shift0_flow_5 = draw_flow(Shift0,flow_5)
cv.imwrite("./output/ps6-1-b-3.png",Shift0_flow_5)

#2a
yos_img_01 = cv.imread("./input/DataSeq1/yos_img_01.jpg")
gauss_pyramid = gaussian_pyramid(yos_img_01,4)
plt.subplot(1,4,1)
plt.imshow(gauss_pyramid[0], 'gray')
plt.subplot(1,4,2)
plt.imshow(gauss_pyramid[1], 'gray')
plt.subplot(1,4,3)
plt.imshow(gauss_pyramid[2], 'gray')
plt.subplot(1,4,4)
plt.imshow(gauss_pyramid[3], 'gray')
plt.savefig("./output/ps6-2-a-1.png")


#2b
lap_pyramid = laplacian_pyramid(yos_img_01,4)
plt.subplot(1,4,1)
plt.imshow(lap_pyramid[0], 'gray')
plt.subplot(1,4,2)
plt.imshow(lap_pyramid[1], 'gray')
plt.subplot(1,4,3)
plt.imshow(lap_pyramid[2], 'gray')
plt.subplot(1,4,4)
plt.imshow(lap_pyramid[3], 'gray')
plt.savefig("./output/ps6-2-b-1.png")

#3
yos_img_02 = cv.imread("./input/DataSeq1/yos_img_02.jpg")
yos_img_03 = cv.imread("./input/DataSeq1/yos_img_03.jpg")
img_0 = cv.imread("./input/DataSeq2/0.png")
img_1 = cv.imread("./input/DataSeq2/1.png")
img_2 = cv.imread("./input/DataSeq2/2.png")
data_seq1 = [yos_img_01,yos_img_02,yos_img_03]
for img in data_seq1:
    gauss_pyramid = gaussian_pyramid(img,4,True)
    flows = [lucas_kanade(gauss_pyramid[i], gauss_pyramid[i + 1][1], (27, 27)) for i in range(2)]
    flows = np.array(flows)
    print("flows shape 3a: " + str(flows.shape))
    arrows = [draw_flow(gauss_pyramid[i], flows[i]) for i in range(2)]
    h_0, w_0 = flows[0].shape[:2]
    h_1,w_1 = flows[1].shape[:2]
    flow_map_0 = -flows[0].copy()
    flow_map_1 = -flows[1].copy()
    flow_map_0[:, :, 0] += np.arange(w_0)
    flow_map_0[:, :, 1] += np.arange(h_0)[:, np.newaxis]
    flow_map_1[:, :, 0] += np.arange(w_1)
    flow_map_1[:, :, 1] += np.arange(h_1)[:, np.newaxis]
    remaps_0 = cv.remap(gauss_pyramid[0][1], flow_map_0.astype(np.float32), None, cv.INTER_LINEAR)
    remaps_1 = cv.remap(gauss_pyramid[1][1], flow_map_1.astype(np.float32), None, cv.INTER_LINEAR)
    plt.subplot(1,2,1)
    plt.imshow(arrows[0],'gray')
    plt.subplot(1,2,2)
    plt.imshow(arrows[1],'gray')
    plt.savefig("./output/ps6-3-a-1.png")
    diffs_0 = ((img - np.min(remaps_0)) / (np.max(img) - np.min(remaps_0)) * 255).astype(int)
    diffs_1 = ((img - np.min(remaps_1)) / (np.max(img) - np.min(remaps_1)) * 255).astype(int)
    plt.subplot(1,2,1)
    plt.imshow(diffs_0,'gray')
    plt.subplot(1,2,2)
    plt.imshow(diffs_1,'gray')
    plt.savefig("./output/ps6-3-a-2.png")

data_seq2 = [img_0,img_1,img_2]
for img in data_seq2:
    gauss_pyramid = gaussian_pyramid(img,4,True)
    flows = [lucas_kanade(gauss_pyramid[i], gauss_pyramid[i + 1][1], (27, 27)) for i in range(2)]
    flows = np.array(flows)
    print("flows shape 3b: " + str(flows.shape))
    arrows = [draw_flow(gauss_pyramid[i], flows[i]) for i in range(2)]
    h_0, w_0 = flows[0].shape[:2]
    h_1,w_1 = flows[1].shape[:2]
    flow_map_0 = -flows[0].copy()
    flow_map_1 = -flows[1].copy()
    flow_map_0[:, :, 0] += np.arange(w_0)
    flow_map_0[:, :, 1] += np.arange(h_0)[:, np.newaxis]
    flow_map_1[:, :, 0] += np.arange(w_1)
    flow_map_1[:, :, 1] += np.arange(h_1)[:, np.newaxis]
    remaps_0 = cv.remap(gauss_pyramid[1][1], flow_map_0.astype(np.float32), None, cv.INTER_LINEAR)
    remaps_1 = cv.remap(gauss_pyramid[2][1], flow_map_1.astype(np.float32), None, cv.INTER_LINEAR)
    plt.subplot(1,2,1)
    plt.imshow(arrows[0],'gray')
    plt.subplot(1,2,2)
    plt.imshow(arrows[1],'gray')
    plt.savefig("./output/ps6-3-a-3.png")
    diffs_0 = ((img - np.min(remaps_0)) / (np.max(img) - np.min(remaps_0)) * 255).astype(int)
    diffs_1 = ((img - np.min(remaps_1)) / (np.max(img) - np.min(remaps_1)) * 255).astype(int)
    plt.subplot(1,2,1)
    plt.imshow(diffs_0,'gray')
    plt.subplot(1,2,2)
    plt.imshow(diffs_1,'gray')
    plt.savefig("./output/ps6-3-a-4.png")

#4a
hier = []
hier_1 = hierarchical_lk(Shift0,ShiftR2)
hier_flow_1 = draw_flow(Shift0,hier_1)
hier.append(hier_flow_1)
print("hier_flow_1 4a: " + str(hier_flow_1.shape))
hier_2 = hierarchical_lk(Shift0,ShiftR5U5)
hier_flow_2 = draw_flow(Shift0,hier_2)
hier.append(hier_flow_2)
print("hier_flow_2 4a: " + str(hier_flow_2.shape))
hier_3 = hierarchical_lk(Shift0,ShiftR10)
hier_flow_3 = draw_flow(Shift0,hier_3)
hier.append(hier_flow_3)
print("hier_flow_3 4a: " + str(hier_flow_3.shape))
hier_4 = hierarchical_lk(Shift0,ShiftR20)
hier_flow_4 = draw_flow(Shift0,hier_4)
hier.append(hier_flow_4)
print("hier_flow_4 4a: " + str(hier_flow_4.shape))
hier_5 = hierarchical_lk(Shift0,ShiftR40)
hier_flow_5 = draw_flow(Shift0,hier_5)
hier.append(hier_flow_5)
print("hier_flow_5 4a: " + str(hier_5.shape))
numpy_horizontal = np.hstack((hier[0],hier[1],hier[2],hier[3],hier[4]))
cv.imwrite("./output/ps6-4-a-1.png",numpy_horizontal)

diffs_1 = ((Shift0 - np.min(hier_flow_1)) / (np.max(Shift0) - np.min(hier_flow_1)) * 255).astype(int)
plt.subplot(1,5,1)
plt.imshow(diffs_1, 'gray')
diffs_2 = ((Shift0 - np.min(hier_flow_2)) / (np.max(Shift0) - np.min(hier_flow_2)) * 255).astype(int)
plt.subplot(1,5,2)
plt.imshow(diffs_2, 'gray')
diffs_3 = ((Shift0 - np.min(hier_flow_3)) / (np.max(Shift0) - np.min(hier_flow_3)) * 255).astype(int)
plt.subplot(1,5,3)
plt.imshow(diffs_3, 'gray')
diffs_4 = ((Shift0 - np.min(hier_flow_4)) / (np.max(Shift0) - np.min(hier_flow_4)) * 255).astype(int)
plt.subplot(1,5,4)
plt.imshow(diffs_4, 'gray')
diffs_5 = ((Shift0 - np.min(hier_flow_5)) / (np.max(Shift0) - np.min(hier_flow_5)) * 255).astype(int)
plt.subplot(1,5,5)
plt.imshow(diffs_5, 'gray')
plt.savefig("./output/ps6-4-a-2.png")


#4b
img_count = 1
arr = []
diff = []
for i in range(2):
    gauss_pyramid = gaussian_pyramid(data_seq1[i],4,True)
    flows = hierarchical_lk(data_seq1[i], data_seq1[i + 1])
    flows = np.array(flows)
    print("new flows shape 4b: " + str(flows[0].shape))
    print("img shape 4b: " + str(data_seq1[i].shape))
    arrows = draw_flow(data_seq1[i], flows)
    arr.append(arrows)
    h_0, w_0 = flows.shape[:2]
    flow_map_0 = -flows.copy()
    flow_map_0[:, :, 0] += np.arange(w_0)
    flow_map_0[:, :, 1] += np.arange(h_0)[:, np.newaxis]
    remaps_0 = cv.remap(gauss_pyramid[0][1], flow_map_0.astype(np.float32), None, cv.INTER_LINEAR)
    diffs_0 = ((data_seq1[i] - np.min(remaps_0)) / (np.max(data_seq1[i]) - np.min(remaps_0)) * 255).astype(int)
    diff.append(diffs_0)
    img_count += 1
numpy_horizontal = np.hstack((arr[0], arr[1]))
cv.imwrite("./output/ps6-4-b-1.png",numpy_horizontal)
numpy_horizontal = np.hstack((diff[0], diff[1]))
cv.imwrite("./output/ps6-4-b-2.png",numpy_horizontal)

#4c
img_count = 1
arr = []
diff = []
for i in range(2):
    gauss_pyramid = gaussian_pyramid(data_seq2[i],4,True)
    flows = hierarchical_lk(data_seq2[i], data_seq2[i + 1])
    flows = np.array(flows)
    print("new flows shape 4b: " + str(flows[0].shape))
    print("img shape 4b: " + str(data_seq2[i].shape))
    arrows = draw_flow(data_seq2[i], flows)
    arr.append(arrows)
    h_0, w_0 = flows.shape[:2]
    flow_map_0 = -flows.copy()
    flow_map_0[:, :, 0] += np.arange(w_0)
    flow_map_0[:, :, 1] += np.arange(h_0)[:, np.newaxis]
    remaps_0 = cv.remap(gauss_pyramid[0][1], flow_map_0.astype(np.float32), None, cv.INTER_LINEAR)
    diffs_0 = ((data_seq2[i] - np.min(remaps_0)) / (np.max(data_seq2[i]) - np.min(remaps_0)) * 255).astype(int)
    diff.append(diffs_0)
numpy_horizontal = np.hstack((arr[0], arr[1]))
cv.imwrite("./output/ps6-4-c-1.png",numpy_horizontal)
numpy_horizontal = np.hstack((diff[0], diff[1]))
cv.imwrite("./output/ps6-4-c-2.png",numpy_horizontal)