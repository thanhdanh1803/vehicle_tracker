import json
import bb_polygon
import numpy as np

def load_zone_anno(json_file):
    """
    Load the json with ROI and MOI annotation
    """
    with open(json_file) as jsonfile:
        dd = json.load(jsonfile)
        polygon = [(int(x), int(y)) for x, y in dd['shapes'][0]['points']]
        paths = {}
        for it in dd['shapes'][1:]:
            kk = str(int(it['label'][-2:]))
            paths[kk] = [(int(x), int(y)) for x, y in it['points']]
    return polygon, paths

def check_bbox_intersect_polygon(polygon, bbox):
    """
  
    Args:
        polygon: List of points (x,y)
        bbox: A tuple (xmin, ymin, xmax, ymax)
    
    Returns:
        True if the bbox intersect the polygon
    """
    x1, y1, x2, y2 = bbox
    bb = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return bb_polygon.is_bounding_box_intersect(bb, polygon)


def cosin_similarity(a2d, b2d):
    a = np.array((a2d[1][0] - a2d[0][0], a2d[1][1 ]- a2d[0][1]))
    b = np.array((b2d[1][0] - b2d[0][1], b2d[1][1] - b2d[1][0]))
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def counting_moi(paths, obj_vector_list):
    """
    Args:
    paths: List of MOI - (first_point, last_point)
    moto_vector_list: List of tuples (first_point, last_point, last_frame_id) 

    Returns:
    A list of tuples (frame_id, movement_id, vehicle_class_id)
    """
    moi_detection_list = []
    for obj_vector in obj_vector_list:
        max_cosin = -2.0
        movement_id = ''
        last_frame = 0
        for movement_label, movement_vector in paths.items():
            cosin = cosin_similarity(movement_vector, obj_vector)
            if cosin > max_cosin:
                max_cosin = cosin
                movement_id = movement_label
                last_frame = obj_vector[2]
        if movement_id != '':
            moi_detection_list.append((last_frame, movement_id, obj_vector[3]))
    return moi_detection_list

def save_result(result_path, video_name, object_moi_detections):
    with open(result_path, 'w') as f:
        for frame_id, movement_id, vehicle_class_id in object_moi_detections:
            f.write('{} {} {} {}\n'.format(video_name, frame_id, movement_id, vehicle_class_id))
    print('[INFO] Save result susscess!!!') 

def get_vehicle_type(name):
    if name == "person":
        return 1
    elif name == "car":
        return 2
    elif name == "bus":
        return 3
    elif name == "truck":
        return 4