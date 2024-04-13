from mmdet3d.structures import LiDARInstance3DBoxes
import torch
import numpy as np

def threshold_matching_results(matching_results, threshold):
    thresholded_results = []
    for node_pair in matching_results:
        if node_pair[2] >= threshold:
            thresholded_results.append(node_pair)
    return thresholded_results

def extractCorners(data): 
    # Get a list of bboxes. 
    bboxes = LiDARInstance3DBoxes(torch.tensor(data['bboxes_3d']))

    # Convert each bbox to 8 corners
    corners = bboxes.corners
    return corners

def unitVector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angleBetween(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unitVector(v1)
    v2_u = unitVector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

if __name__=="__main__": 
    v1 = np.array([1, 0, 0])
    v2 = np.array([-1, 0, 0])
    print(f"Angle = {angleBetween(v1, v2)}")
