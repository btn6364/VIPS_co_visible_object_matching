#!/usr/bin/python

import numpy as np
from mmdet3d.structures import LiDARInstance3DBoxes
import torch
from co_visible_object_matching import findCorrespondence
from utils import extractCorners

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def findTransformation(correspondence, infra_data, vehicle_data): 
    # TODO: Add weight to each detection object. 
    infra_corners = None
    vehicle_corners = None
    for infra_idx, vehicle_idx in correspondence:
        if infra_corners is None: 
            infra_corners = np.matrix(extractCorners(infra_data)[infra_idx]).transpose()
        else: 
            infra_corners = np.hstack((infra_corners, np.matrix(extractCorners(infra_data)[infra_idx]).transpose()))
        if vehicle_corners is None:
            vehicle_corners = np.matrix(extractCorners(vehicle_data)[vehicle_idx]).transpose()
        else: 
            vehicle_corners = np.hstack((vehicle_corners, np.matrix(extractCorners(vehicle_data)[vehicle_idx]).transpose()))
    R, t = rigid_transform_3D(infra_corners, vehicle_corners)

    # Create a transformation matrix 
    # r1 r2 r3 t1
    # r4 r5 r6 t2
    # r7 r8 r9 t3
    # 0  0  0  1
    T = np.eye(4)
    T[:3, :3] = np.copy(R)
    T[:3, 3:4] = np.copy(t) 

    return T



if __name__=="__main__": 
    correspondence, infra_data, vehicle_data = findCorrespondence()
    T = findTransformation(correspondence, infra_data, vehicle_data)
    print(T) 