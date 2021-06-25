#!/usr/bin/env python3

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import math

left_ = [cv2.imread(image, 0) for image in sorted(glob.glob("../../images/task_3_and_4/left_*.png"))]
right_ = [cv2.imread(image, 0) for image in sorted(glob.glob("../../images/task_3_and_4/right_*.png"))]

def plot_figures(figures, nrows=1, ncols=1):

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))
    if(nrows > 1 or ncols > 1):
        for ind,title in enumerate(figures):
            axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
            axeslist.ravel()[ind].set_title(title)
            axeslist.ravel()[ind].set_axis_off()
        plt.tight_layout() # optional
    else:
        for ind,title in enumerate(figures):
            axeslist.imshow(figures[title], cmap=plt.gray())
            axeslist.set_title(title)
            axeslist.set_axis_off()

s = cv2.FileStorage('../../parameters/left_camera_intrinsics.xml', cv2.FILE_STORAGE_READ)

mtx_left = s.getNode('Intrinsic').mat()
dist_left = s.getNode('Distortion').mat()
s.release()

s = cv2.FileStorage('../../parameters/right_camera_intrinsics.xml', cv2.FILE_STORAGE_READ)

mtx_right = s.getNode('Intrinsic').mat()
dist_right = s.getNode('Distortion').mat()
s.release()

s = cv2.FileStorage('../../parameters/stereo_calibration.xml', cv2.FILE_STORAGE_READ)
R = s.getNode('R').mat()
T = s.getNode('T').mat()
E = s.getNode('E').mat()
F = s.getNode('F').mat()


s = cv2.FileStorage('../../parameters/stereo_rectification.xml', cv2.FILE_STORAGE_READ)
R1 = s.getNode('R1').mat()
R2 = s.getNode('R2').mat()
P1 = s.getNode('P1').mat()
P2 = s.getNode('P2').mat()
Q = s.getNode('Q').mat()
roi1 = s.getNode('roi1').mat()
roi2 = s.getNode('roi2').mat()

img = left_[0].copy()
# num = 0
for i, img in enumerate(left_):
    h,  w = img.shape[:2]
    # print(h,w)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_left,dist_left,(w,h),0,(w,h))
    dst = cv2.undistort(img, mtx_left, dist_left, None, newcameramtx)
    # print(dst.shape)
    x,y,w,h = roi
    # print(roi)
    dst = dst[y:y+h, x:x+w]
    # cv2.imwrite('i'+str(num)+'.png', dst)
    left_[i] = dst
    # num+=1

img = right_[0].copy()
for i, img in enumerate(right_):
    h,  w = img.shape[:2]
    # print(h,w)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_right,dist_right,(w,h),0,(w,h))
    dst = cv2.undistort(img, mtx_right, dist_right, None, newcameramtx)
    # print(dst.shape)
    x,y,w,h = roi
    # print(roi)
    dst = dst[y:y+h, x:x+w]
    right_[i] = dst

def euclidean_dist(pt1, pt2):
    return math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)

def filter_kp(kp, distance=8):
    l = []

    for i in range(len(kp)-2):
        for j in range(i+1,len(kp)-2):
            if euclidean_dist(kp[i].pt,kp[j].pt) < distance:
                # keep the point with the higher response value
                if kp[i].response > kp[j].response:
                    l.append(j)
                else:
                    l.append(i)
                    j = len(kp)

    l = list(dict.fromkeys(l))
    l = sorted(l, reverse=True)
    for i in l:
        kp.pop(i)
    
    return kp


# kp_left_filtered = filter_kp(kp_left.copy())
# kp_right_filtered = filter_kp(kp_right.copy())


# img_left_pts = cv2.drawKeypoints(img_left.copy(), kp_left_filtered, outImage = None, color=(0,255,0))
# img_right_pts = cv2.drawKeypoints(img_right.copy(), kp_right_filtered, outImage = None, color=(255,0,0))

# plot_figures({'ORB features left':img_left_pts, 'ORB features right':img_right_pts},1,2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
for i in range(len(left_)):
    img_left = left_[i].copy()
    img_right = right_[i].copy()

    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp_left = orb.detect(img_left,None)
    kp_right = orb.detect(img_right,None)

     # compute the descriptors with ORB
    kp_left, des_left = orb.compute(img_left, kp_left)
    kp_right, des_right = orb.compute(img_right, kp_right)

    kp_left_filtered = filter_kp(kp_left.copy())
    kp_right_filtered = filter_kp(kp_right.copy())

    img_left_pts_u = cv2.drawKeypoints(img_left.copy(), kp_left, outImage = None, color=(0,255,0))
    img_right_pts_u = cv2.drawKeypoints(img_right.copy(), kp_right, outImage = None, color=(255,0,0))

    img_left_pts = cv2.drawKeypoints(img_left.copy(), kp_left_filtered, outImage = None, color=(0,255,0))
    img_right_pts = cv2.drawKeypoints(img_right.copy(), kp_right_filtered, outImage = None, color=(255,0,0))

    # plot_figures({'ORB features left':img_left_pts, 'ORB features right':img_right_pts},1,2)

    img2 = cv2.hconcat([img_left_pts,img_right_pts])
    img2i = cv2.hconcat([img_left_pts_u,img_left_pts])
    
    # Match descriptors.
    matches = bf.match(des_left,des_right)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)


    img3 = cv2.drawMatches(img_left,kp_left,img_right,kp_right,matches[:50], None,flags=2)
    # plot_figures({'matches' + str(i):img3})
    
    if i in [5,7,8,9]:
        cv2.imwrite('../../output/task_3/stereo_matches_example_'+str(i)+'.png', img3)
        cv2.imwrite('../../output/task_3/stereo_orbfeatures_example_'+str(i)+'.png', img2)
        cv2.imwrite('../../output/task_3/stereo_orbfeatures__unfiltered_example_'+str(i)+'.png', img2i)

featurePoints_left = []
featurePoints_right = []

for i in range(len(matches)):  
    featurePoints_left.append(kp_left[matches[i].queryIdx].pt)
    featurePoints_right.append(kp_right[matches[i].trainIdx].pt)

np.array(featurePoints_left).transpose()[0]

def createProjectionMatrix(m, R, T):
    pi0 = np.identity(3)
    pi0 = np.append(pi0,np.array([[0.],[0.],[0.]]),axis=1)
    temp = np.append(R,T,axis=1)
    temp = np.append(temp, [0.,0.,0.,1.]).reshape(4,4)
    PM1 = np.matmul(mtx_left,pi0)
    PM1 = np.matmul(PM1, temp)
    return PM1
rot = np.identity(3)
points4D = cv2.triangulatePoints(projMatr1=P1,
                                 projMatr2=P2,
                                 projPoints1=np.array(featurePoints_left).transpose(), 
                                 projPoints2=np.array(featurePoints_right).transpose(), 
                                 points4D=None)

x = points4D[0]/points4D[1]
y = points4D[1]/points4D[2]
z = points4D[2]/points4D[3]

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x,y,z)
plt.show()