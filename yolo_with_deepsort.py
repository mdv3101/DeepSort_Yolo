from __future__ import division, print_function, absolute_import

import numpy as np
from src_code.utils import nms, NNDistanceMetric
from src_code.tracker import  Detection, Tracker
from src_code import detection_process as gdet
import os
import cv2
import sys
import darknet as darknet

encoder = gdet.box_encoder_fetch('./model_data/mars-small128.pb',sz_bh=1)
metric = NNDistanceMetric(0.2, 100)
tracker = Tracker(metric)

def create_detections(bbox,confidence_list,features, min_height=0):
    detection_list = []
    for i in range(0,len(bbox)):
        bbox2, confidence, feature = bbox[i], confidence_list[i], features[i]
        if bbox2[3] < min_height:
            continue
        detection_list.append(Detection(bbox2, confidence, feature))
    return detection_list

def track_fun(tracker,im,bbox,confidence_list,min_detection_height,nms_max_overlap):
         bbox_trk_cord = []
         bbox_trk_id = []
         features = encoder(im,bbox)
         detections = create_detections(bbox,confidence_list,features,min_detection_height)
         boxes = np.array([d.bbox for d in detections])
         scores = np.array([d.score for d in detections])
         indices = nms(boxes, nms_max_overlap, scores)
         detections = [detections[i] for i in indices]
         bb = [d.bbox for d in detections]
         tracker.predict_tracker()
        
         tracker.update_tracker(detections)
         for track in tracker.track_list:
              if not track.state==2 or track.last_update > 1:
                   continue
              bbox_trk_cord.append(track.to_bbox2())
              bbox_trk_id.append(track.id)
         return bbox_trk_cord,bbox_trk_id

def run(min_confidence=0.25, nms_max_overlap=1.0, min_detection_height =0.0, max_cosine_distance = 0.2,nn_budget=100):
    global metric
    global tracker
    metric = NNDistanceMetric(max_cosine_distance, nn_budget)
    tracker = Tracker(metric)


imagePath = './videos/yolo_person.mp4'
showImage = "False"
show_box = "True"
show_confidence = "False"

personDetectionThreshold = 0.25
network, class_names, class_colors = darknet.load_network('./model_data/yolov4.cfg','./model_data/coco.data','./yolov4.weights',batch_size=1)

run(min_confidence=0.25, nms_max_overlap=1.0, min_detection_height =0.0, max_cosine_distance = 0.2,nn_budget=100)

bbox = []
cap = cv2.VideoCapture(imagePath)
ret,im = cap.read()
height,width = im.shape[:2]
print(width,height)
out = cv2.VideoWriter(os.getcwd()+'/Deep_sort_output.mp4',cv2.VideoWriter_fourcc(*'DIVX'),25,(width,height))
cap.release()

cap = cv2.VideoCapture(imagePath)
if showImage == 'True':
    cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", (720, 480))

while (1):
    ret,im2 = cap.read()
    if ret == False:
        break
    confidence_list= []
    bbox = []
    frame_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb,(width, height),interpolation=cv2.INTER_LINEAR)
    img_for_detect = darknet.make_image(width,height,3)
    darknet.copy_image_from_bytes(img_for_detect,frame_resized.tobytes())
    im = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    detections_total = darknet.detect_image(network, class_names, img_for_detect, thresh=personDetectionThreshold)
    for detection in detections_total:
        label = detection[0]
        if label != 'person':
              continue
        confidence = float(detection[1])
        pstring = label+": "+str(np.rint(100 * confidence))+"%"
        bounds = detection[2]
        h = int(bounds[3])
        w = int(bounds[2])
        xCoord = max(int(bounds[0] - bounds[2]/2),0)
        yCoord = max(int(bounds[1] - bounds[3]/2),0)
        bbox_temp = [xCoord , yCoord , (w) ,(h)]
        if w*h>(width*height/4):
          continue
        bbox.append(bbox_temp)
        confidence_list.append(confidence)
        if show_box == "True":
              cv2.rectangle(im, (int(xCoord), int(yCoord)), (int(xCoord+w), int(yCoord+h)),(0,0,255), 2)
        if show_confidence == 'True':
              cv2.putText(im,str(pstring),(int(xCoord), int(yCoord)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)

    bbox_trk_cord,bbox_trk_id = track_fun(tracker,im,bbox,confidence_list,min_detection_height= 0.0,nms_max_overlap= 1.0)
    for j  in range(0,len(bbox_trk_id)):
        bbox = bbox_trk_cord[j]
        bbox_id = bbox_trk_id[j]
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        cv2.putText(im, str(bbox_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
    out.write(im)
    if showImage == 'True':
        cv2.imshow("Video",im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              out.release()
              break
print("--Complete--")
out.release()