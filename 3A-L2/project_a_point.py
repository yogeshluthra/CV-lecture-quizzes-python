import cv2
import numpy as np

# Project a point from 3D to 2D using a matrix operation

# Given: Point p in 3-space [x y z], and focal length f
# Return: Location of projected point on 2D image plane [u v]


def project_point(p, f):
    projection_matrix = np.array([[f,0,0,0],
                                [0,f,0,0],
                                [0,0,1,0]]).astype(float)
    p = np.hstack((p,[[1]])).T
    linear_trans = projection_matrix.dot(p)
    return linear_trans[:-1,0]/linear_trans[-1,0]

# Test: Given point and focal length (units: mm)
p = np.array([[200, 100, 120]])
f = 50

print project_point(p, f)