import cv2 as cv 
import numpy as np 

from harris import harris
from ransac import ransac


red = (0,0,255)

#1a
print("CV2 VERSION: " + str(cv.__version__))
print("NUMPY VERSION: " + str(np.__version__))
transA = cv.imread("./input/transA.jpg")
simA = cv.imread("./input/simA.jpg")
trans_gx = cv.Sobel(transA, cv.CV_64F, 1, 0, ksize=3)
trans_gy = cv.Sobel(transA, cv.CV_64F, 0, 1, ksize=3)
trans_grad = np.concatenate([transA, ((trans_gx - np.min(trans_gx)) / (np.max(trans_gx) - np.min(trans_gx)) * 255).astype(int), ((trans_gy - np.min(trans_gy)) / (np.max(trans_gy) - np.min(trans_gy)) * 255).astype(int)], axis=1)
cv.imwrite("./output/ps5-1-a-1.png",trans_grad)
sim_gx = cv.Sobel(simA, cv.CV_64F, 1, 0, ksize=3)
sim_gy = cv.Sobel(simA, cv.CV_64F, 0, 1, ksize=3)
sim_grad = np.concatenate([simA, ((sim_gx - np.min(sim_gx)) / (np.max(sim_gx) - np.min(sim_gx)) * 255).astype(int), ((sim_gy - np.min(sim_gy)) / (np.max(sim_gy) - np.min(sim_gy)) * 255).astype(int)], axis=1)
cv.imwrite("./output/ps5-1-a-2.png",sim_grad)

#1b and 1c
transB = cv.imread("./input/transB.jpg")
simB = cv.imread("./input/simB.jpg")
harris_transA, corners_transA,points_transA = harris(transA,5,.004,.30)
print("harris_transA shape: " + str(harris_transA))
print("harris_transA: " + str(harris_transA))
cv.imwrite("./output/ps5-1-b-1.png",harris_transA)
cv.imwrite("./output/ps5-1-c-1.png",corners_transA)
harris_transB, corners_transB,points_transB = harris(transB,5,.004,.30)
cv.imwrite("./output/ps5-1-b-2.png",harris_transB)
cv.imwrite("./output/ps5-1-c-2.png",corners_transB)
harris_simA, corners_simA,points_simA = harris(simA,5,.004,.30)
cv.imwrite("./output/ps5-1-b-3.png",harris_simA)
cv.imwrite("./output/ps5-1-c-3.png",corners_simA)
harris_simB, corners_simB,points_simB = harris(simB,5,.004,.30)
cv.imwrite("./output/ps5-1-b-4.png",harris_simB)
cv.imwrite("./output/ps5-1-c-4.png",corners_simB)

#2a
from sift import sift
trans_pts_a, trans_desc_a = sift(transA,points_transA)
trans_pts_b, trans_desc_b = sift(transB,points_transB)
gray_transA = cv.cvtColor(transA,cv.COLOR_BGR2GRAY)
gray_transB = cv.cvtColor(transB,cv.COLOR_BGR2GRAY)
im_a = np.concatenate([gray_transA[:, :, np.newaxis]] * 3, axis=2)
im_b = np.concatenate([gray_transB[:, :, np.newaxis]] * 3, axis=2)
cv.drawKeypoints(im_a, trans_pts_a, im_a, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.drawKeypoints(im_b, trans_pts_b, im_b, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
trans_ab = np.concatenate([im_a, im_b], axis=1)
print("trans_ab shape: " + str(trans_ab.shape))
cv.imwrite("./output/ps5-2-a-1.png",trans_ab)

sim_pts_a, sim_desc_a = sift(simA,points_simA)
sim_pts_b, sim_desc_b = sift(simB,points_simB)
gray_simA = cv.cvtColor(simA,cv.COLOR_BGR2GRAY)
gray_simB = cv.cvtColor(simB,cv.COLOR_BGR2GRAY)
im_c = np.concatenate([gray_simA[:, :, np.newaxis]] * 3, axis=2)
im_d = np.concatenate([gray_simB[:, :, np.newaxis]] * 3, axis=2)
cv.drawKeypoints(im_c, sim_pts_a, im_c, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.drawKeypoints(im_d, sim_pts_b, im_d, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
sim_ab = np.concatenate([im_c, im_d], axis=1)
print("sim_ab shape: " + str(sim_ab.shape))
cv.imwrite("./output/ps5-2-a-2.png",sim_ab)

#2b
trans_pts_a = np.array(trans_pts_a)
trans_pts_b = np.array(trans_pts_b)
bfm = cv.BFMatcher()
print("trans_desc_a: " + str(trans_desc_a.shape))
print("trans_desc_a: " + str(trans_desc_a))
print("sim_desc_b shape: " + str(sim_desc_a.shape))
print("trans_desc_b: " + str(trans_desc_b))
matches_trans = bfm.match(trans_desc_a, trans_desc_b)
#matches_trans = sorted(matches_trans, key=lambda x: x.distance)
matches_trans = [(trans_pts_a[match.queryIdx].pt,trans_pts_b[match.trainIdx].pt) for match in matches_trans]
putative_trans = np.concatenate([gray_transA, gray_transB], axis=1)
putative_trans = np.concatenate([putative_trans[:, :, np.newaxis]] * 3, axis=2)
matches_trans = np.array(matches_trans,dtype=int)
print("matches shape: (" + str(matches_trans.shape))
print("matches: " + str(matches_trans))
for i in range(matches_trans.shape[0]):
    cv.line(putative_trans,(matches_trans[i][0][0],matches_trans[i][0][1]),(matches_trans[i][1][0] + gray_transA.shape[1],matches_trans[i][1][1]),color=red,thickness=1)
cv.imwrite("./output/ps5-2-b-1.png",putative_trans)

sim_pts_a = np.array(sim_pts_a)
sim_pts_b = np.array(sim_pts_b)
bfm = cv.BFMatcher()
print("sim_desc_a shape: " + str(sim_desc_a.shape))
print("sim_desc_a: " + str(sim_desc_a))
print("sim_desc_b shape: " + str(sim_desc_b.shape))
print("sim_desc_b: " + str(sim_desc_b))
matches_sim = bfm.match(sim_desc_a, sim_desc_b)
#matches_sim = sorted(matches_sim, key=lambda x: x.distance)
matches_sim = [(sim_pts_a[match.queryIdx].pt,sim_pts_b[match.trainIdx].pt) for match in matches_sim]
putative_sim = np.concatenate([gray_simA, gray_simB], axis=1)
putative_sim = np.concatenate([putative_sim[:, :, np.newaxis]] * 3, axis=2)
matches_sim = np.array(matches_sim,dtype=int)
for i in range(matches_sim.shape[0]):
    cv.line(putative_sim,(matches_sim[i][0][0],matches_sim[i][0][1]),(matches_sim[i][1][0] + gray_simA.shape[1],matches_sim[i][1][1]),color=red,thickness=1)
cv.imwrite("./output/ps5-2-b-2.png",putative_sim)


#3a
matches_index = ransac(matches=matches_trans,sample_size=1)
matches_trans = [matches_trans[i] for i in list(matches_index)]
consensus_trans = np.concatenate([gray_transA, gray_transB], axis=1)
consensus_trans = np.concatenate([consensus_trans[:, :, np.newaxis]] * 3, axis=2)
matches_trans = np.array(matches_trans,dtype=int)
for i in range(matches_trans.shape[0]):
    cv.line(consensus_trans,(matches_trans[i][0][0],matches_trans[i][0][1]),(matches_trans[i][1][0] +gray_transA.shape[1],matches_trans[i][1][1]),color=red,thickness=1)
cv.imwrite("./output/ps5-3-a-1.png",consensus_trans)

#3b
#TODO: 
matches_index = ransac(matches=matches_sim,sample_size=2)
matches_sim = [matches_sim[i] for i in list(matches_index)]
consensus_sim = np.concatenate([gray_simA, gray_simB], axis=1)
consensus_sim = np.concatenate([consensus_sim[:, :, np.newaxis]] * 3, axis=2)
matches_sim = np.array(matches_sim,dtype=int)
for i in range(matches_sim.shape[0]):
    cv.line(consensus_sim,(matches_sim[i][0][0],matches_sim[i][0][1]),(matches_sim[i][1][0] + gray_simA.shape[1],matches_sim[i][1][1]),color=red,thickness=1)
cv.imwrite("./output/ps5-3-b-1.png",consensus_sim)