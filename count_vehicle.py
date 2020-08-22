import numpy as np
import cv2

def get_vehicle_count(boxes, class_names, list_of_vehicle = []):
    total_vehicle_count = 0
    dict_vehicel_count = {}
    for i in range(len(boxes)):
        class_name = class_names[i]
        if class_name in list_of_vehicle:
            total_vehicle_count += 1
            dict_vehicel_count[class_name] = dict_vehicel_count.get(class_name, 0) + 1 #return 0 if class_name not existed in dict
    return total_vehicle_count, dict_vehicel_count

def count(frame, net, layers, labels, list_of_things=[],  set_confidence = 0.5):
    boxes = []
    confidences = []
    classIDs = []
    classnames = []
    (H, W) = frame.shape[:2]
    #print('{}x{}'.format(H, W))
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), \
        swapRB=True, crop=False)
    net.setInput(blob)
    # the `.forward` method is used to forward-propagate
    #  our image and obtain the actual classification.
    layerOutputs = net.forward(layers)
    #scan each output in all layeroutput
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:] # get all probability of each object
            classID = np.argmax(scores) #
            confidence = scores[classID]
            if confidence > set_confidence:
                #scale the bouding box back to relative to the 
                #size of the image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')
                #get the top left of the bounding box
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                classnames.append(labels[classID])
    total_vehicle, each_vehicle = get_vehicle_count(boxes, classnames, list_of_things)
    return (boxes, confidences, classIDs, classnames, total_vehicle, each_vehicle)