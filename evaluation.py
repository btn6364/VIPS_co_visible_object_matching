import numpy as np 
from math import atan2, asin 
from object_alignment import findTransformationAll
import os 
from scipy.spatial.transform import Rotation as R

def rte(t1, t2): 
    dist = np.linalg.norm(t1 - t2).round(2) 
    return dist

def rre(R_t, R_e): 
    # Calculate the rotation matrix
    R_t_inversed = np.linalg.inv(R_t)
    # R = np.matmul(R_t_inversed, R_e)
    
    # Calculate the three Euler angles (Z-Y-X order)
    # yaw = atan2(R[1,0], R[0,0])
    # pitch = -asin(R[2,0])
    # roll = atan2(R[2,1], R[2,2])
    
    # # Calculate the L1 distance
    # dist = np.sum(np.abs(np.array([yaw, pitch, roll]))).round(2)

    intermidate_RRE = R.from_matrix(np.dot(R_t_inversed, R_e))
    a = intermidate_RRE.as_euler('zyx', degrees=True)
    rre = round(sum(abs(number) for number in a), 2)

    return rre

# Return min, max and mean for both RTE and RRE. 
def calculateRTEandRRE(): 
    # Get all the predicted transformation matrices
    transformation_matrices = findTransformationAll() 

    # Calculate RTE, RRE
    total_rte, total_rre = 0, 0
    min_rte, min_rre = float("inf"), float("inf")
    max_rte, max_rre = 0, 0
    infra_dir = f"../mmdetection3d/outputs/test/infra/preds/"
    veh_dir = f"../mmdetection3d/outputs/test/vehicle/preds/"
    num_frames_to_process = min(len(os.listdir(infra_dir)), len(os.listdir(veh_dir)))
    
    # Get the I_W matrix
    I_W = np.genfromtxt("../Segmentation_Dataset/I_W.txt", dtype=float)
    for i in range(num_frames_to_process): 
        # Get the V_W matrix
        V_W = np.genfromtxt("../Segmentation_Dataset/V_W.txt", dtype=float)[4*i: 4*(i+1)]

        # Compute the ground truth matrix I_V = I_W * V_W^(-1)
        T_t = np.matmul(I_W, np.linalg.inv(V_W))
        R_t, t_t = T_t[:3,:3], T_t[:3,3]

        # Get the predicted matrix.
        # If i doesn't exist, assume it matches the ground truth. 
        T_e = transformation_matrices.get(i, T_t)
        R_e, t_e = T_e[:3,:3], T_e[:3,3]

        # Calculate the RTE and RRE
        cur_rte, cur_rre = rte(t_t, t_e), rre(R_t, R_e)
        total_rte += cur_rte
        total_rre += cur_rre
        if cur_rte != 0 and cur_rre != 0:
            min_rte, min_rre = min(min_rte, cur_rte), min(min_rre, cur_rre)
            max_rte, max_rre = max(max_rte, cur_rte), max(max_rre, cur_rre)
    
    # Calculate the mean RTE and mean RRE for each dataset
    mean_rte, mean_rre = round(total_rte / num_frames_to_process, 2), round(total_rre / num_frames_to_process, 2)
    return mean_rte, mean_rre, min_rte, max_rte, min_rre, max_rre

if __name__=="__main__": 
    # t1 = np.array([1,2,3])
    # t2 = np.array([1,1,1])
    # dist = rte(t1, t2)
    # print(f"Distance = {dist}")

    # R_t = np.array([
    #     [0.94, 0, 0.34], 
    #     [0, 1, 0], 
    #     [-0.34, 0, 0.94]
    # ])
    # R_e = np.array([
    #     [-0.0004, -0.99, 0.000009], 
    #     [0.99, -0.0004, 0.0002],
    #     [-0.0002, 0.000001, 1]
    # ])
    # dist = rre(R_t, R_e)
    # print(f"angle dist = {dist}")

    mean_rte, mean_rre, min_rte, max_rte, min_rre, max_rre = calculateRTEandRRE()
    print(f"RTE mean = {mean_rte}")
    print(f"RTE min = {min_rte}")
    print(f"RTE max = {max_rte}")
    print(f"RRE mean = {mean_rre}")
    print(f"RRE min = {min_rre}")
    print(f"RRE max = {max_rre}")