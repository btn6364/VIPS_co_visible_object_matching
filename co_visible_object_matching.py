from main import main as object_matching
from utils import extractCorners, angleBetween
import json 
import torch
import numpy as np
import open3d as o3d
import numpy as np

def findDist(p1, p2): 
    dist = torch.sqrt(torch.sum(torch.pow(torch.subtract(p1, p2), 2), dim=0)).item()
    return dist

def findBbox(corner): 
    width = findDist(corner[0], corner[3])
    length = findDist(corner[3], corner[7])
    height = findDist(corner[0], corner[1])
    return [length, width, height]

def createMap(filename, filename2, tag, infra_transform, vehicle_transform): 
    with open(filename) as cur_frame, open(filename2) as next_frame: 
        # Load the JSON file
        data = json.load(cur_frame)
        next_data = json.load(next_frame)

        # Get the 8 corners
        corners = extractCorners(data) 
        next_corners = extractCorners(next_data) 

        num_objects = len(data["labels_3d"])   
        num_objects_next = len(next_data["labels_3d"])
        # print(f"Type = {tag} Cur frame num objects = {num_objects}, Num objects in next frame = {num_objects_next}")
        map = {
            "category": [],
            "position": [],
            "bounding_box": [], 
            "world_position": [], 
            "heading": [],
        } 
        for i in range(num_objects): 
            category = data["labels_3d"][i]
            corner = corners[i]
            position = (corner[0] + corner[6]) / 2
            bbox = findBbox(corner) 
            map["category"].append(category)
            map["position"].append(position)
            map["bounding_box"].append(bbox)

            # Find the world position for each bounding box
            # Calculate the new position for matrix multiplication
            new_position = position.numpy()
            new_position = np.append(new_position, 1) # add 1 to the end of the numpy array to make it 1x4
            new_position = new_position[:, np.newaxis] # add a new axis
            transform = infra_transform if tag == "infra" else vehicle_transform
            world_position = np.matmul(transform, new_position)[:3,:].flatten()
            map["world_position"].append(world_position)

            # Add heading angle to increase the credibility of the edge-to-edge affinity.
            # Find the nearest position in the next frame 
            closest_next_position = None 
            min_dist = float("inf")
            for j in range(num_objects_next):
                next_corner = next_corners[j]
                next_position = (next_corner[0] + next_corner[6]) / 2
                # Calculate the distance between next_position and position 
                # print(f"next_position = {next_position}, type = {type(next_position)}")
                dist = ((next_position-position)**2).sum(axis=0).item()
                if dist < min_dist: 
                    closest_next_position = next_position
                    min_dist = dist

            # Compute the velocity vector of the vehicle
            if closest_next_position is not None:
                heading_vector = closest_next_position - position
                
                # Compute the angle between the heading vector and the x-axis in Radians
                angle = angleBetween(heading_vector.numpy(), np.array([1, 0, 0]))   
            else: 
                angle = 0.0
            
            map["heading"].append([angle])
        return map, data

def findCorrespondence(infra_cur_frame, infra_next_frame, veh_cur_frame, veh_next_frame, frame_idx):
    # Read the transformation matrices to world coordinate
    infra_transform = np.genfromtxt("../Segmentation_Dataset/I_W.txt", dtype=float)
    vehicle_transform = np.genfromtxt("../Segmentation_Dataset/V_W.txt", dtype=float)[4*frame_idx:4*frame_idx+4,:]

    infra_map, infra_data = createMap(
        infra_cur_frame, infra_next_frame, "infra", infra_transform, vehicle_transform
    )
    vehicle_map, vehicle_data = createMap(
        veh_cur_frame, veh_next_frame, "vehicle", infra_transform, vehicle_transform
    )
    correspondence = object_matching(infra_map, vehicle_map)
    return correspondence, infra_data, vehicle_data

if __name__=="__main__": 
    correspondence, _, _ = findCorrespondence(
        "../mmdetection3d/outputs/test/infra/preds/32.json", 
        "../mmdetection3d/outputs/test/infra/preds/33.json",
        "../mmdetection3d/outputs/test/vehicle/preds/32.json", 
        "../mmdetection3d/outputs/test/vehicle/preds/33.json", 
        32
    )
    print(correspondence)