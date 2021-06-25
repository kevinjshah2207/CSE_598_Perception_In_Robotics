#!/usr/bin/env python3

import cv2
import numpy as np
import os
import glob
import sys



rows = 6
columns = 9
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints_left = []
imgpoints_right = []

objp = np.zeros((rows*columns, 3), np.float32)
objp[:,:2] = np.mgrid[0:columns, 0:rows].T.reshape(-1,2)
prev_img_shape = None

# Left Camera
images_left = sorted(glob.glob('../../images/task_1/left_*.png'))
num = 0
for fname in images_left:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (columns,rows), None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints_left.append(corners2)
        img = cv2.drawChessboardCorners(img, (columns,rows), corners2, ret)
    if(fname == '../../images/task_1/left_10.png'):
        cv2.imwrite('../../output/task_1/drawChessboardCorners_left_10.png',img)
    else:
        cv2.imwrite('../../output/task_1/drawChessboardCorners_left_'+str(num)+'.png',img)
        num+=1

ret, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, (640,480), None, None)
images = sorted(glob.glob('../../images/task_1/left_*.png'))
num = 0
for fname in images:
    img = cv2.imread(fname)
    h,w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_left,dist_left,(w,h),0,(w,h))
    dst = cv2.undistort(img, mtx_left, dist_left, None, newcameramtx)

    x,y,w,h = roi
    dst_crop = dst[y:y+h, x:x+w]
    if(fname == '../../images/task_1/left_10.png'):
        cv2.imwrite('../../output/task_1/Final_Image_left_10.png',dst_crop)
    else:
        cv2.imwrite('../../output/task_1/Final_Image_left_'+str(num)+'.png',dst_crop)
        num+=1

# Right Camera
objpoints = []
images_right = sorted(glob.glob('../../images/task_1/right_*.png'))
num = 0
for fname in images_right:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (columns,rows), None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints_right.append(corners2)
        img = cv2.drawChessboardCorners(img, (columns,rows), corners2, ret)
    if(fname == '../../images/task_1/right_10.png'):
        cv2.imwrite('../../output/task_1/drawChessboardCorners_right_10.png',img)
    else:
        cv2.imwrite('../../output/task_1/drawChessboardCorners_right_'+str(num)+'.png',img)
        num+=1

ret, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, (640,480), None, None)

images = sorted(glob.glob('../../images/task_1/right_*.png'))
num = 0
for fname in images:
    img = cv2.imread(fname)
    h,w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_right,dist_right,(w,h),0,(w,h))
    dst = cv2.undistort(img, mtx_right, dist_right, None, newcameramtx)

    x,y,w,h = roi
    dst_crop = dst[y:y+h, x:x+w]
    if(fname == '../../images/task_1/right_10.png'):
        cv2.imwrite('../../output/task_1/Final_Image_right_10.png',dst_crop)
    else:
        cv2.imwrite('../../output/task_1/Final_Image_right_'+str(num)+'.png',dst_crop)
        num+=1

#Parameter Storage
c_left = cv2.FileStorage("../../parameters/left_camera_intrinsics.xml",cv2.FILE_STORAGE_WRITE)
c_left.write("Intrinsic", mtx_left)
c_left.write("Distortion", dist_left)
c_left.release()

c_right = cv2.FileStorage("../../parameters/right_camera_intrinsics.xml",cv2.FILE_STORAGE_WRITE)
c_right.write("Intrinsic", mtx_right)
c_right.write("Distortion", dist_right)
c_right.release()
