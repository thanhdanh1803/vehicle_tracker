import numpy as np
import time
import imutils
import cv2
import os
import yolo_config as config
import vehicle_detector

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from utils import *

list_of_things = ['person','bus', 'truck','car']   
#list_of_things = ['motorbike','truck'] 
                 #type 1: bicycle, motobike
                #type 2: car
                #type 3: bus
                #type 4: truck
demo_video = os.path.join(config.DATA_PATH, 'sample_01.mp4')
print(demo_video)
lablesPath = os.path.join(config.MODEl_PATH, 'coco.names')
LABELS = open(lablesPath).read().strip().split('\n')
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weigthsPath = os.path.join(config.MODEl_PATH,'yolov4.weights')
configPath = os.path.join(config.MODEl_PATH, 'yolov4.cfg')

NUM_CLASS = vehicle_detector.read_class_names(LABELS)
key_list = list(NUM_CLASS.keys()) 
#print(key_list)
val_list = list(NUM_CLASS.values())
#print(val_list)
#config params for deep sort
max_cosine_distance = 1.5
max_euclidean_distance = 20.0
nn_budget = None
mode_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(mode_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('euclidean', max_euclidean_distance, nn_budget)
#metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)
#load our YOLO object detector trained on COCO dataset
print('[INFO] Loading YOLO model...')
net = cv2.dnn.readNetFromDarknet(configPath, weigthsPath)

#determine only the output layer names that we need from the YOLO
ln = net.getLayerNames() #get name of all layers (254 components with 106 conv layers)
#print(len(ln))
ln = [ ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] #get detection layers name: 82, 94, 106
#print(ln)
#load path vector and polygon
polygon, paths = load_zone_anno('sample_data/videos/sample_01.json')
print('[INFO] Accessing video stream...')
fps = 10
vs = cv2.VideoCapture(demo_video)
vs.set(cv2.CAP_PROP_FPS, fps)
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = None
#loop over the frames from the video stream
#create a dict to save tracked item wiht id
track_dict = {}
frameid = 0
while True:
    #read the next frame from file
    (grabbed, frame) = vs.read()
    if not grabbed:
        print('[INFO] Compeleted!!!')
        break
    try:
        frameid += 1
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    except:
        break
    #test with 50 first frames
    if frameid > 100:
        break
    #resize the frame
    frame = original_frame#imutils.resize(original_frame, width = width)
    results = vehicle_detector.dectect(frame, net, ln, \
        list_of_things=list_of_things, labels = LABELS, set_confidence= 0.4)
    (bboxes, confidences, classIDs, classnames) = results
    #print(classIDs)
    idxs = cv2.dnn.NMSBoxes(bboxes, confidences, 0.4, 0.1)
    #get boxes, scores and names
    boxes, scores, names = [], [], []
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box coordinates
            if classnames[i] in list_of_things:
                boxes.append(bboxes[i])
                scores.append(confidences[i])
                names.append(classnames[i])
                # draw a bounding box rectangle and label on the image
                # (x, y) = (bboxes[i][0], bboxes[i][1])
                # (w, h) = (bboxes[i][2], bboxes[i][3])
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                # text = "{}: {:.1f}".format(LABELS[classIDs[i]], confidences[i])
                # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5, color, 2)
    boxes = np.array(boxes)
    names = np.array(names)
    scores = np.array(scores)
    features = np.array(encoder(frame, boxes))
    #print(boxes)
    detections = [Detection(bbox, score, class_name, feature)\
         for bbox, score, class_name, feature in zip(boxes, scores, names, features)]
    tracker.predict()
    tracker.update(detections)
    #obtain info from the tracks
    tracked_bboxes = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 3:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        #save tracked item id
        if check_bbox_intersect_polygon(polygon,bbox):
            tracking_id = int(track.track_id)
            if tracking_id not in track_dict.keys():
                track_dict[tracking_id] = [(bbox[0], bbox[1], bbox[2], bbox[3], class_name, frameid)]
            else:
                track_dict[tracking_id].append((bbox[0], bbox[1], bbox[2], bbox[3], class_name, frameid))
            index = key_list[val_list.index(class_name)]
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index])
            #print temp result
            tracker_list = track_dict[tracking_id]
            if(len(tracker_list) > 1):
                first = tracker_list[0]
                last = tracker_list[-1]
                first_point = ((first[2] - first[0])/2, (first[3] - first[1])/2)
                last_point = ((last[2] - last[0])/2, (last[3] - last[1])/2)
                vehicle_type = get_vehicle_type(last[4])
                last_frame = last[5]
                object_moi_detections = counting_moi(paths, [[first_point, last_point, last_frame, vehicle_type]])
                print(object_moi_detections)
    
                frame = vehicle_detector.draw_bbox(frame, tracked_bboxes, labels= LABELS, tracking=True)
    if 1 > 0:
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        #check q key
        if key == ord('q'):
            break
#count vehicle from track_dict
object_vector_list =[]
for tracker_id, tracker_list in track_dict.items():
    if(len(tracker_list) > 1):
        first = tracker_list[0]
        last = tracker_list[-1]
        first_point = ((first[2] - first[0])/2, (first[3] - first[1])/2)
        last_point = ((last[2] - last[0])/2, (last[3] - last[1])/2)
        vehicle_type = get_vehicle_type(last[4])
        last_frame = last[5]
        object_vector_list.append([first_point, last_point, last_frame, vehicle_type])

object_moi_detections = counting_moi(paths, object_vector_list)
print(object_moi_detections)
    #check output
    # if args['output'] != '' and writer is None:
    #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #     writer = cv2.VideoWriter(args['output'], fourcc, 25, (frame.shape[1], frame.shape[0]), True)
    # if writer is not None:
    #     writer.write(frame)           