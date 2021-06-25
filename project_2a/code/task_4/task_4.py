#!/usr/bin/env python3

import cv2
import glob
import numpy as np

left_ = [cv2.imread(image, 0) for image in sorted(glob.glob("../../images/task_3_and_4/left_*.png"))]
right_ = [cv2.imread(image, 0) for image in sorted(glob.glob("../../images/task_3_and_4/right_*.png"))]

import matplotlib.pyplot as plt
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

h, w = left_[4].shape[:2]
# print(h,w)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_left, dist_left, (w,h), 0, (w,h))
print(roi)
# mapx,mapy = cv2.initUndistortRectifyMap(mtx_left,dist_left,None,newcameramtx,(w,h),cv2.CV_32FC1)
# left_6_rectified = cv2.remap(left_[6].copy(),mapx,mapy,cv2.INTER_LINEAR)[y:y+h, x:x+w]
left_5_rectified = cv2.undistort(left_[8].copy(), mtx_left, dist_left, None, newcameramtx)
x,y,w,h = roi
left_5_rectified = left_5_rectified[y:y+h, x:x+w]

# newcameramtx, roi5 = cv2.getOptimalNewCameraMatrix(mtx_left, dist_left, (w,h), 0, (w,h))
# print(roi5)
right_5_rectified = cv2.undistort(right_[8].copy(), mtx_right, dist_right, None, newcameramtx)
# x,y,w,h = roi
right_5_rectified = right_5_rectified[y:y+h, x:x+w]
# mapx,mapy = cv2.initUndistortRectifyMap(mtx_right,dist_right,None,newcameramtx,(w,h),cv2.CV_32FC1)
# right_6_rectified = cv2.remap(right_[6].copy(),mapx,mapy,cv2.INTER_LINEAR)[y:y+h, x:x+w]
# right_6_rectified = right_6_rectified[y:y+h, x:x+w+41]
# left_6_rectified = left_6_rectified[y:y+h+61, x:x+w]

print(left_5_rectified.shape, right_5_rectified.shape)
out = cv2.hconcat([left_5_rectified, right_5_rectified])
cv2.imwrite('../../output/task_4/con1.png', out)

stereo = cv2.StereoBM_create(numDisparities=32, blockSize=13)
disparity = stereo.compute(left_5_rectified,right_5_rectified)
# print(disparity)
plt.imshow(disparity,'gray')
plt.show()

_3dImage = cv2.reprojectImageTo3D(disparity=disparity, Q=Q)
_3dImage.shape

x = _3dImage.transpose()[0].flatten()
y = _3dImage.transpose()[1].flatten()
z = _3dImage.transpose()[2].flatten()

z = _3dImage.transpose()[2].copy()
z = z.transpose()

plt.imshow(z,'gray')
plt.show()

cv2.imwrite('../../output/task_4/left_right_8_disparity.png', disparity)
cv2.imwrite('../../output/task_4/left_right_8_3D.png', z)