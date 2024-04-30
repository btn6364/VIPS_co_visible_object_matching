import numpy as np
from mmdet3d.structures import LiDARInstance3DBoxes
import torch
from co_visible_object_matching import findCorrespondence
from utils import extractCorners
import os 
import time

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

def findTransformationOneFrame(correspondence, infra_data, vehicle_data): 
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

def findTransformationAll(): 
    # Start the timer
    start_time = time.time()

    # Find the transformation matrix for each frame
    transformation_matrices = {}
    infra_dir = f"../mmdetection3d/outputs/test/infra/preds/"
    veh_dir = f"../mmdetection3d/outputs/test/vehicle/preds/"
    num_frames_to_process = min(len(os.listdir(infra_dir)), len(os.listdir(veh_dir)))
    print(f"Num frames to process = {num_frames_to_process}")
    infra_files = sorted(os.listdir(infra_dir), key=lambda x: int(x.split(".")[0])) 
    veh_files = sorted(os.listdir(veh_dir), key=lambda x: int(x.split(".")[0]))

    # Skip the last frame because we need at least 2 frames to calculate the velocity
    for i in range(num_frames_to_process - 1): 
        # TODO: These frames raise an Exception related to Scipy that needs to be fixed. 
        if i in [70, 90, 91, 93, 101, 102, 160]:
            continue

        print(f"Processing frame {i}th...")
        cur_infra_frame = os.path.join(infra_dir, infra_files[i])
        next_infra_frame = os.path.join(infra_dir, infra_files[i+1])
        cur_veh_frame = os.path.join(veh_dir, veh_files[i])
        next_veh_frame = os.path.join(veh_dir, veh_files[i+1])

        correspondence, infra_data, vehicle_data = findCorrespondence(
            cur_infra_frame, next_infra_frame, 
            cur_veh_frame, next_veh_frame, 
            i
        )
        T = findTransformationOneFrame(correspondence, infra_data, vehicle_data)

        # Store the transformation matrix for each frame ith. 
        transformation_matrices[i] = T 

    end_time = time.time() 
    elapsed_time = round(end_time - start_time, 2) 
    print(f"Total elapsed time for all transformation = {elapsed_time}s")

    return transformation_matrices

if __name__=="__main__": 
    # correspondence, infra_data, vehicle_data = findCorrespondence(
    #     "../mmdetection3d/outputs/test/infra/preds/0.json", 
    #     "../mmdetection3d/outputs/test/infra/preds/1.json",
    #     "../mmdetection3d/outputs/test/vehicle/preds/0.json", 
    #     "../mmdetection3d/outputs/test/vehicle/preds/1.json", 
    #     0
    # )
    # T = findTransformationOneFrame(correspondence, infra_data, vehicle_data)
    # print(T) 
    findTransformationAll()