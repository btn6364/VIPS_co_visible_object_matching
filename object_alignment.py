import numpy as np
from mmdet3d.structures import LiDARInstance3DBoxes
import torch
from co_visible_object_matching import findCorrespondence
from utils import extractCorners
import os 

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
    # TODO: Run the pipeline in all 5 datasets
    # datasets = ["Dataset_1", "Dataset_2", "Dataset_3"]
    # sub_datasets = ["D1", "D2", "D3", "D4", "D5"]
    datasets = ["Dataset_1"]
    sub_datasets = ["D1"]
    transformation_matrices = {}
    for dataset in datasets: 
        for sub_dataset in sub_datasets:
            infra_dir = f"../mmdetection3d/outputs/carla/{dataset}/{sub_dataset}/infra/preds/"
            veh_dir = f"../mmdetection3d/outputs/carla/{dataset}/{sub_dataset}/vehicle/preds/"
            num_frames_to_process = min(len(os.listdir(infra_dir)), len(os.listdir(veh_dir)))
            print(f"Num frames to process = {num_frames_to_process}")
            infra_files, veh_files = os.listdir(infra_dir), os.listdir(veh_dir)
            for i in range(num_frames_to_process - 1): 
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
                transformation_matrices[(dataset, sub_dataset, i)] = T 
    return transformation_matrices

if __name__=="__main__": 
    # correspondence, infra_data, vehicle_data = findCorrespondence(
    #     "../mmdetection3d/outputs/carla/Dataset_1/D1/infra/preds/1689811023.137300000.json", 
    #     "../mmdetection3d/outputs/carla/Dataset_1/D1/infra/preds/1689811023.215958000.json",
    #     "../mmdetection3d/outputs/carla/Dataset_1/D1/vehicle/preds/1689811023.097195000.json", 
    #     "../mmdetection3d/outputs/carla/Dataset_1/D1/vehicle/preds/1689811023.177662000.json",
    #     0
    # )
    # T = findTransformationOneFrame(correspondence, infra_data, vehicle_data)
    # print(T) 
    findTransformationAll()