import os
import cv2
import argparse
import numpy as np
from app_support import visualize
from src.utils import nms, NNDistanceMetric
from src.tracker import  Detection, Tracker

def start(sequence_path, detection_path, output_path, min_conf,
        nms_thresh, min_detect_height_thresh, cosine_thresh,nn_budget, disp):
    image_dir = os.path.join(sequence_path, "img1")
    images = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    gt_file = os.path.join(sequence_path, "gt/gt.txt")
    detections = None
    if detection_path is not None:
        detections = np.load(detection_path)
    gt = None
    if os.path.exists(gt_file):
        gt = np.loadtxt(gt_file, delimiter=',')
    if len(images) > 0:
        image = cv2.imread(next(iter(images.values())), cv2.IMREAD_GRAYSCALE)
        image_shape = image.shape
    else:
        image_shape = None
    if len(images) > 0:
        min_idx = min(images.keys())
        max_idx = max(images.keys())
    else:
        min_idx = int(detections[:, 0].min())
        max_idx = int(detections[:, 0].max())
    info_filename = os.path.join(sequence_path, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)
        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None
    print(len(detections))		
    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    info = {
        "sequence_name": os.path.basename(sequence_path),
        "images": images,
        "detections": detections,
        "gt": gt,
        "image_shape": image_shape,
        "min_idx": min_idx,
        "max_idx": max_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    metric = NNDistanceMetric(cosine_thresh, nn_budget) 
    trker = Tracker(metric)   
    outputs = []
    def process_frames(visualizer, index_frame):
        detection_list = []
        for row in info["detections"][info["detections"][:, 0].astype(np.int) == index_frame]:
            bbox, confidence, feature = row[2:6], row[6], row[10:]
            if bbox[3] < min_detect_height_thresh:
                continue
            detection_list.append(Detection(bbox, confidence, feature))
        detections = [d for d in detection_list if d.score >= min_conf]
        indices = nms(
            np.array([d.bbox for d in detections]), nms_thresh, 
            np.array([d.score for d in detections]))
        detections = [detections[i] for i in indices]
        trker.predict_tracker()
        trker.update_tracker(detections)
        if disp=="True":
            img = cv2.imread(info["images"][index_frame], cv2.IMREAD_COLOR)
            visualizer.image = img
            visualizer.detections(detections)
            visualizer.trackers(trker.track_list)
        for trk in trker.track_list:
            if not trk.state==2 or trk.last_update > 1: 
                continue
            bbox = trk.to_bbox()
            outputs.append([
                index_frame, trk.id, bbox[0], bbox[1], bbox[2], bbox[3]])

    if disp =="False":
        while info['min_idx'] <= info['max_idx']:
            process_frames(None,info['min_idx'])
            info['min_idx'] += 1
    else:
        vis = visualize.Visualization(info, time_to_update_in_ms=15)
		vis.start_viewer_(process_frames)

    f = open(output_path, 'w')
    for row in outputs:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)

if __name__ == "__main__":
    # TODO: Update default 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_path", help="Path to sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_path", help="Path to detections", default=None,
        required=True)
    parser.add_argument(
        "--output_path", help="Path to the tracking output file",
        default="./output_detections_MOT16.txt")
    parser.add_argument(
        "--min_conf", help="Detection confidence threshold", default=0.8,
        type=float)
    parser.add_argument(
        "--min_detect_height_thresh", help="Detection bounding box height"
        " threshold", default=0, type=int)
    parser.add_argument(
        "--nms_thresh",  help="Non-maxima suppression threshold", default=1.0,
        type=float)
    parser.add_argument(
        "--cosine_thresh", help="Cosine distance gating threshold",
        type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--disp", help='Visualize the results in between',
        dest='disp', action='store_true',default="False",type=str
    )

    args = parser.parse_args()
    start(
        args.sequence_path, args.detection_path, args.output_path,
        args.min_conf, args.nms_thresh, args.min_detect_height_thresh,
        args.cosine_thresh, args.nn_budget, args.disp)
