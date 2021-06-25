#!/usr/bin/env python3

import cv2
import numpy as np
import os
import glob
import math as m
import matplotlib.pyplot as plt

rows = 6
columns = 9
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

left_ = [cv2.imread(image) for image in glob.glob("../../images/task_2/left_*.png")]
right_ = [cv2.imread(image) for image in glob.glob("../../images/task_2/right_*.png")]

def plot_figures(figures, nrows=1, ncols=1):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))
    if(nrows > 1 or ncols > 1):
        for ind,title in enumerate(figures):
            axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
            axeslist.ravel()[ind].set_title(title)
            axeslist.ravel()[ind].set_axis_off()
            plt.tight_layout()
    else:
        for ind,title in enumerate(figures):
            axeslist.imshow(figures[title], cmap=plt.gray())
            axeslist.set_title(title)
            axeslist.set_axis_off()

# plot_figures({'left_1': left_[0], 'right_1': right_[0]}, 1, 2)

objpoints = []
imgpoints_left = []
imgpoints_right = []

objp = np.zeros((rows*columns, 3), np.float32)
objp[:,:2] = np.mgrid[0:columns, 0:rows].T.reshape(-1,2)
prev_img_shape = None

cv_file_right = cv2.FileStorage("../../parameters/right_camera_intrinsics.xml",cv2.FILE_STORAGE_READ)
mtx_right = cv_file_right.getNode("Intrinsic").mat()
dist_right = cv_file_right.getNode("Distortion").mat()
cv_file_right.release()
cv_file_left = cv2.FileStorage("../../parameters/left_camera_intrinsics.xml",cv2.FILE_STORAGE_READ)
mtx_left = cv_file_left.getNode("Intrinsic").mat()
dist_left = cv_file_left.getNode("Distortion").mat()
cv_file_left.release()

# print(mtx_left[0][0])
images_left = sorted(glob.glob('../../images/task_2/left_*.png'))
for fname in images_left:
    img = cv2.imread(fname)
    gray_left = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (columns,rows), None)
    if ret_left == True:
        objpoints.append(objp)
        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
        imgpoints_left.append(corners2_left)
        img = cv2.drawChessboardCorners(img, (columns,rows), corners2_left, ret_left)

images_right = sorted(glob.glob('../../images/task_2/right_*.png'))
for fname in images_right:
    img = cv2.imread(fname)
    gray_right = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (columns,rows), None)
    if ret_right == True:
        #objpoints.append(objp)
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)
        imgpoints_right.append(corners2_right)
        img = cv2.drawChessboardCorners(img, (columns,rows), corners2_right, ret_right)



img_shape = gray_left.shape[::-1]

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objectPoints= objpoints, imagePoints1 = imgpoints_left,imagePoints2 = imgpoints_right, imageSize = tuple([640,480]), cameraMatrix1 =mtx_left,distCoeffs1 = dist_left, cameraMatrix2 = mtx_right,distCoeffs2 = dist_right,criteria = stereocalib_criteria,flags = flags)
# print(ret)
# print('Intrinsic_mtx_1\n', M1)
# print('dist_1\n', d1)
# print('Intrinsic_mtx_2\n', M2)
# print('dist_2\n', d2)
# print('R \n', R)
# print('T \n', T)
# print('E\n', E)
# print('F\n', F)

undistortedPoints_left = []
for imgpoints in imgpoints_left:
    undistortedPoints_left.append(cv2.undistortPoints(imgpoints, M1, d1, None, None, None))
undistortedPoints_right = []
for imgpoints in imgpoints_right:
    undistortedPoints_right.append(cv2.undistortPoints(imgpoints, M2, d2, None, None, None))

# undistortedPoints_right[0].shape
# undistortedPoints_right[0].reshape(2,54)
# undistortedPoints_right[0].transpose().reshape(2,54)

def createProjectionMatrix(m, R, T):
    p0 = np.identity(3)
    p0 = np.append(p0,np.array([[0.],[0.],[0.]]),axis=1)
    # print(p0)
    temp = np.append(R, T, axis=1)
    temp = np.append(temp, [0.,0.,0.,1.]).reshape(4,4)
    PM1 = np.matmul(m,p0)
    PM1 = np.matmul(PM1, temp)
    return PM1

# print(createProjectionMatrix(mtx_left, R, T))
R1 = np.identity(3)
points4D = cv2.triangulatePoints(projMatr1=createProjectionMatrix(mtx_left,R1,T*0),projMatr2=createProjectionMatrix(mtx_right,R,T),projPoints1=undistortedPoints_left[0], projPoints2=undistortedPoints_right[0], points4D=None)

# print(points4D.shape)
# print(points4D)
x = points4D[0]/points4D[3]
y = points4D[1]/points4D[3]
z = points4D[2]/points4D[3]

#Triangulation after Rectification

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
h, w = left_[0].shape[:2]
h = h*0.264 # 1 pixel = 0.264mm
w = w*0.264
fx = mtx_left[0][0]/1000
fy = mtx_left[1][1]/1000
fovy = 2*(m.atan(h/(2*fy)))
fovx = 2*(m.atan(w/(2*fx)))
tan_x = m.tan(w/2)
tan_y = m.tan(h/2)
# tan_x = fovx
# tan_y = fovy
R_prime = R
t_prime = T
cam_center_local1 = np.asarray([
    [0, 0, 0],      [tan_x, tan_y, 1],
       [tan_x, -tan_y, 1],     [0, 0, 0],      [tan_x, -tan_y, 1],
       [-tan_x, -tan_y, 1],    [0, 0, 0],      [-tan_x, -tan_y, 1],
       [-tan_x, tan_y, 1],     [0, 0, 0],      [-tan_x, tan_y, 1],
       [tan_x, tan_y, 1],      [0, 0, 0]
       ]).T
 
cam_center_local1 *= fx
cam_center1 = np.matmul(R_prime, cam_center_local1) + t_prime

fx = (mtx_right[0][0])/100
fy = (mtx_right[1][1])/100
h = h*0.264 # 1 pixel = 0.264mm
w = w*0.264
fovy = 2*(m.atan(h/(2*fy)))
fovx = 2*(m.atan(w/(2*fx)))
tan_x = m.tan(w/2)
tan_y = m.tan(h/2)
# tan_x = fovx
# tan_y = fovy
R_prime = R
t_prime = T
cam_center_local2 = np.asarray([
    [0, 0, 0],      [tan_x, tan_y, 1],
       [tan_x, -tan_y, 1],     [0, 0, 0],      [tan_x, -tan_y, 1],
       [-tan_x, -tan_y, 1],    [0, 0, 0],      [-tan_x, -tan_y, 1],
       [-tan_x, tan_y, 1],     [0, 0, 0],      [-tan_x, tan_y, 1],
       [tan_x, tan_y, 1],      [0, 0, 0]
       ]).T
print(t_prime) 
cam_center_local2 *= fy
cam_center2 = np.matmul(R_prime, cam_center_local2) + t_prime

ax.scatter(x,y,z)
# fig = plt.figure()
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')
ax.plot(cam_center1[0, :], cam_center1[1, :], cam_center1[2, :],color='k', linewidth=1)
ax.plot(cam_center2[0, :], cam_center2[1, :], cam_center2[2, :],color='k', linewidth=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax = fig.add_subplot(111, projection = '3d')
# ax.scatter(x,y,z)

plt.show()

R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(cameraMatrix1=mtx_left, cameraMatrix2=mtx_right,distCoeffs1=dist_left, distCoeffs2=dist_right,imageSize=(480,640),R=R,T=T)
# print(R1,R2,P1,P2,Q,roi1,roi2)
# print('R1',R1)
# print('R2',R2)
# print('P1',P1)
# print('P2',P2)
# print('Q',Q)
# print('roi1',roi1)
# print('roi2',roi2)

#Triangulation after Rectification
points4D = cv2.triangulatePoints(projMatr1=P1,projMatr2=P2,projPoints1=undistortedPoints_left[0], projPoints2=undistortedPoints_right[0], points4D=None)

# print(points4D.shape)
# print(points4D)
x = points4D[0]/points4D[3]
y = points4D[1]/points4D[3]
z = points4D[2]/points4D[3]


fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
h, w = left_[0].shape[:2]
h = h*0.264 # 1 pixel = 0.264mm
w = w*0.264
fx = mtx_left[0][0]/1000
fy = mtx_left[1][1]/1000
fovy = 2*(m.atan(h/(2*fy)))
fovx = 2*(m.atan(w/(2*fx)))
tan_x = m.tan(w/2)
tan_y = m.tan(h/2)
# tan_x = fovx
# tan_y = fovy
R_prime = R1
t_prime = T
cam_center_local1 = np.asarray([
    [0, 0, 0],      [tan_x, tan_y, 1],
       [tan_x, -tan_y, 1],     [0, 0, 0],      [tan_x, -tan_y, 1],
       [-tan_x, -tan_y, 1],    [0, 0, 0],      [-tan_x, -tan_y, 1],
       [-tan_x, tan_y, 1],     [0, 0, 0],      [-tan_x, tan_y, 1],
       [tan_x, tan_y, 1],      [0, 0, 0]
       ]).T
 
cam_center_local1 *= fx
cam_center1 = np.matmul(R_prime, cam_center_local1) + t_prime

fx = (mtx_right[0][0])/100
fy = (mtx_right[1][1])/100
h = h*0.264 # 1 pixel = 0.264mm
w = w*0.264
fovy = 2*(m.atan(h/(2*fy)))
fovx = 2*(m.atan(w/(2*fx)))
tan_x = m.tan(w/2)
tan_y = m.tan(h/2)
# tan_x = fovx
# tan_y = fovy
R_prime = R2
t_prime = T
cam_center_local2 = np.asarray([
    [0, 0, 0],      [tan_x, tan_y, 1],
       [tan_x, -tan_y, 1],     [0, 0, 0],      [tan_x, -tan_y, 1],
       [-tan_x, -tan_y, 1],    [0, 0, 0],      [-tan_x, -tan_y, 1],
       [-tan_x, tan_y, 1],     [0, 0, 0],      [-tan_x, tan_y, 1],
       [tan_x, tan_y, 1],      [0, 0, 0]
       ]).T
print(t_prime) 
cam_center_local2 *= fy
cam_center2 = np.matmul(R_prime, cam_center_local2) + t_prime

ax.scatter(x,y,z)
# fig = plt.figure()
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')
ax.plot(cam_center1[0, :], cam_center1[1, :], cam_center1[2, :],color='k', linewidth=1)
ax.plot(cam_center2[0, :], cam_center2[1, :], cam_center2[2, :],color='k', linewidth=1) 
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax = fig.add_subplot(111, projection = '3d')
# ax.scatter(x,y,z)

plt.show()



h, w = left_[0].shape[:2]
# print(R1)
gray_left = cv2.cvtColor(left_[0], cv2.COLOR_BGR2GRAY)
ret_left, corners_left = cv2.findChessboardCorners(gray_left, (columns,rows), None)
if ret_left == True:
    corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
    left_[0] = cv2.drawChessboardCorners(left_[0], (columns,rows), corners2_left, ret_left)
gray_left = cv2.cvtColor(left_[1], cv2.COLOR_BGR2GRAY)
ret_left, corners_left = cv2.findChessboardCorners(gray_left, (columns,rows), None)
if ret_left == True:
    corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
    left_[1] = cv2.drawChessboardCorners(left_[1], (columns,rows), corners2_left, ret_left)
ori = cv2.hconcat([left_[0], left_[1]])
cv2.imwrite('../../output/task_2/original_left0_left1.png', ori)
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_left, dist_left, (w,h), 1, (w,h))
# print(roi)
x,y,w,h = roi1
left_0_undistorted = cv2.undistort(left_[0].copy(), mtx_left, dist_left, None, None)#[y:y+h, x:x+w]
mapx,mapy = cv2.initUndistortRectifyMap(mtx_left,dist_left,R1,None,(w,h), cv2.CV_32FC1)
# print(mapx,mapy)
left_0_rectified = cv2.remap(left_[0].copy(),mapx,mapy,cv2.INTER_LINEAR)#[y:y+h, x:x+w]
# print(left_0_rectified)
# cv2.imshow('image',left_0_rectified)
# cv2.imshow('image',left_0_rectified)

# h, w = left_[1].shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_left, dist_left, (w,h), 1, (w,h))
x,y,w,h = roi1
left_1_undistorted = cv2.undistort(left_[1].copy(), mtx_left, dist_left, None, None)#[y:y+h, x:x+w]
mapx,mapy = cv2.initUndistortRectifyMap(mtx_left,dist_left,R1,None,(w,h), cv2.CV_32FC1)
left_1_rectified = cv2.remap(left_[1].copy(),mapx,mapy,cv2.INTER_LINEAR)#[y:y+h, x:x+w]
# print(left_0_rectified)
# print(left_0_rectified.shape,left_1_rectified.shape)
# print(left_0_rectified,left_1_rectified)
ori = cv2.hconcat([left_0_undistorted, left_1_undistorted])
cv2.imwrite('../../output/task_2/undistorted_left0_left1.png', ori)

out = cv2.hconcat([left_0_rectified, left_1_rectified])
cv2.imwrite('../../output/task_2/left_0_left_1_rectified.png', out)


#Saving Parameters

s = cv2.FileStorage('../../parameters/stereo_calibration.xml', cv2.FILE_STORAGE_WRITE)

# calibration
s.write('R', R)
s.write('T', T)
s.write('E', E)
s.write('F', F)

s.release()


s = cv2.FileStorage('../../parameters/stereo_rectification.xml', cv2.FileStorage_WRITE)

# rectification parameters
s.write('R1', R1)
s.write('R2', R2)
s.write('P1', P1)
s.write('P2', P2)
s.write('Q', Q)
s.write('roi1', roi1)
s.write('roi2', roi2)

s.release()





# plot_figures({'original_left_0':left_[1], 'original_left_1':left_[1],'left_0_undistorted':left_0_undistorted,'left_1_undistorted':left_1_undistorted,
# 'left_0_rectified':left_0_rectified, 'left_1_rectified':left_1_rectified},3,2)