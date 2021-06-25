#!/usr/bin/env python3

import cv2
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
     
f = 1
tan_x = 1
tan_y = 1
 
R_prime = np.identity(3)
t_prime = np.zeros((3, 1))
 
cam_center_local = np.asarray([
    [0, 0, 0],      [tan_x, tan_y, 1],
       [tan_x, -tan_y, 1],     [0, 0, 0],      [tan_x, -tan_y, 1],
       [-tan_x, -tan_y, 1],    [0, 0, 0],      [-tan_x, -tan_y, 1],
       [-tan_x, tan_y, 1],     [0, 0, 0],      [-tan_x, tan_y, 1],
       [tan_x, tan_y, 1],      [0, 0, 0]
       ]).T
 
cam_center_local *= f
cam_center = np.matmul(R_prime, cam_center_local) + t_prime
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
ax.plot(cam_center[0, :], cam_center[1, :], cam_center[2, :],color='k', linewidth=2)
 
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
    
plt.show()