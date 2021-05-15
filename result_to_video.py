import os
import argparse
import sys
import cv2
import numpy as np


parser = argparse.ArgumentParser(description="Video Generation")
parser.add_argument("--mot_dir", help="Path to MOTChallenge directory (train or test)",
	required=True)
parser.add_argument("--result_file", help="Path to tracking output",required=True)
parser.add_argument("--output_path", help="Path to store output video",required=True)
args= parser.parse_args()
out_file = args.output_path + '/mot_output.avi'

image_dir = os.path.join(args.mot_dir, "img1")
if not os.path.exists(image_dir):
	print("Unable to locate images")
	sys.exit()
images = {int(os.path.splitext(f)[0]): os.path.join(image_dir, f) for f in os.listdir(image_dir)}
min_idx = min(images.keys())
max_idx = max(images.keys())
info_filename = os.path.join(args.mot_dir, "seqinfo.ini")
if os.path.exists(info_filename):
	with open(info_filename, "r") as f:
		line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
		info_dict = dict(s for s in line_splits if isinstance(s, list) and len(s) == 2)
	update_ms = 1000 / int(info_dict["frameRate"])

results = np.loadtxt(args.result_file, delimiter=',')
im = cv2.imread(images[min_idx], cv2.IMREAD_COLOR)
height,width = im.shape[:2]
out = cv2.VideoWriter(out_file,cv2.VideoWriter_fourcc(*'MPEG'),update_ms,(width,height))
trackid_color = {}
for i in images.keys():
	img = cv2.imread(images[i], cv2.IMREAD_COLOR)
	mask = results[:, 0].astype(np.int) == i
	track_ids = results[mask, 1].astype(np.int)
	boxes = results[mask, 2:6]
	for j  in range(0,len(track_ids)):
		bbox = boxes[j]
		bbox_id = track_ids[j]
		if bbox_id in trackid_color.keys():
			color = trackid_color[bbox_id]
		else:
			color = np.random.randint(0, 255, 3)
			trackid_color[bbox_id] = color
			#print(color)
		cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),(int(color[0]),int(color[1]),int(color[2])), 2)
		cv2.putText(img, str(bbox_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (int(color[0]),int(color[1]),int(color[2])),2)
	out.write(img)
	cv2.imshow("Video",img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		out.release()
		break
out.release()