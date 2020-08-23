import numpy as np
import cv2
import random
import colorsys

def read_class_names(labels):
    names = {}
    for idx, name in enumerate(labels):
        names[idx] = name
    return names
def draw_bbox(image, bboxes, labels, show_label = True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=False):
    NUM_CLASS = read_class_names(labels)
    #print(NUM_CLASS)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    #print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " "+str(score)

            label = "{}".format(NUM_CLASS[class_ind]) + score_str

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image

def get_vehicle_count(boxes, class_names, list_of_vehicle = []):
    total_vehicle_count = 0
    dict_vehicel_count = {}
    for i in range(len(boxes)):
        class_name = class_names[i]
        if class_name in list_of_vehicle:
            total_vehicle_count += 1
            dict_vehicel_count[class_name] = dict_vehicel_count.get(class_name, 0) + 1 #return 0 if class_name not existed in dict
    return total_vehicle_count, dict_vehicel_count

def dectect(frame, net, layers, labels, list_of_things=[],  set_confidence = 0.5):
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
    return (boxes, confidences, classIDs, classnames)