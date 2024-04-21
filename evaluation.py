import numpy as np 
from math import atan2, asin 
from object_alignment import findTransformationAll
import os 

def rte(t1, t2): 
    dist = np.linalg.norm(t1 - t2).round(2) 
    return dist

def rre(R_t, R_e): 
    # Calculate the rotation matrix
    R_t_inversed = np.linalg.inv(R_t)
    R = np.matmul(R_t_inversed, R_e)
    
    # Calculate the three Euler angles (Z-Y-X order)
    yaw = atan2(R[1,0], R[0,0])
    pitch = -asin(R[2,0])
    roll = atan2(R[2,1], R[2,2])
    
    # Calculate the L1 distance
    dist = np.sum(np.abs(np.array([yaw, pitch, roll]))).round(2)
    return dist

# TODO: Figure out how to get the ground truth rotation matrix. 
# TODO: Implement latency 
# I_V = I_W * V_W^(-1)

def calculateMeanRTEandRRE(): 
    transformation_matrices = findTransformationAll() 
    # TODO: Get the actual ground truth transformation here.
    T_t = np.array([
        [-0.904152274, 0.412234515, -0.112122796, 57.1650085],
        [-0.406502962, -0.910887897, -0.0709831044, -11.8978882],
        [-0.131392971, -0.0186012909, 0.991155863, 0.473078251],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    ])
    R_t, t_t = T_t[:3,:3], T_t[:3,3]

    # TODO: Change this to Dataset_1 -> Dataset_3
    # TODO: Change this to D1 -> D5. 
    datasets = ["Dataset_1"]
    sub_datasets = ["D1"]
    rte_map, rre_map = {}, {}
    for dataset in datasets: 
        for sub_dataset in sub_datasets:
            total_rte, total_rre = 0, 0
            infra_dir = f"../mmdetection3d/outputs/carla/{dataset}/{sub_dataset}/infra/preds/"
            veh_dir = f"../mmdetection3d/outputs/carla/{dataset}/{sub_dataset}/vehicle/preds/"
            num_frames_to_process = min(len(os.listdir(infra_dir)), len(os.listdir(veh_dir)))
            for i in range(num_frames_to_process - 1): 
                T_e = transformation_matrices[(dataset, sub_dataset, i)]
                R_e, t_e = T_e[:3,:3], T_e[:3,3]
                total_rte += rte(t_t, t_e)
                total_rre += rre(R_t, R_e)
            
            mean_rte, mean_rre = round(total_rte / num_frames_to_process, 2), round(total_rre / num_frames_to_process, 2)
            rte_map[(dataset, sub_dataset)] = mean_rte
            rre_map[(dataset, sub_dataset)] = mean_rre
    return rte_map, rre_map

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

    rte_map, rre_map = calculateMeanRTEandRRE()
    print(f"RTE map = {rte_map}")
    print(f"RRE map = {rre_map}")