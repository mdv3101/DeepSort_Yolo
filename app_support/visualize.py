import cv2
import time
import numpy as np


class Visualization(object):
    def __init__(self, info, time_to_update_in_ms):
        self.info = info
        image_shape = self.info["image_shape"][::-1]
        self.image_shape = 1024, int(float(image_shape[1]) / image_shape[0] * 1024)
        self.time_to_update_in_ms = time_to_update_in_ms

    def start_viewer(self, update_fun=None):
        user_function = lambda: None
        end_it = False
        image = np.zeros(self.image_shape + (3, ), dtype=np.uint8)
        if update_fun is not None:
            user_function = update_fun
        end_it, pause = False, False
        while not end_it:
            t0 = time.time()
            if not pause:
                end_it = not user_function()
            t1 = time.time()
            remaining_time = max(1, int(self.time_to_update_in_ms - 1e3*(t1-t0)))
            cv2.imshow(
                "Figure", cv2.resize(image, self.image_shape[:2]))
            key = cv2.waitKey(remaining_time)
            if key & 255 == 27:  # ESC
                end_it = True
            elif key & 255 == 32:  # ' '
                pause = not pause
            elif key & 255 == 115:  # 's'
                end_it = not user_function()
                pause = True
        image[:] = 0
        cv2.destroyWindow("Figure")
        cv2.waitKey(1)
        cv2.imshow("Figure", image)

    def start_viewer_(self, frame_callback):
        self.start_viewer(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.info["min_idx"] > self.info["max_idx"]:
            return False
        frame_callback(self, self.info["min_idx"])
        self.info["min_idx"] += 1
        return True

    def detections(self, detections):
        for detection in detections:
            x, y, w, h = detection.bbox 
            cv2.rectangle(self.image, (int(x), int(y)), (int(x + w), int(y + h)), (125, 125, 125), 1)

    def trackers(self, tracks):
        for track in tracks:
            if not track.state==2 or track.last_update > 0: 
                continue
            x, y, w, h = track.to_bbox().astype(np.int)
            cv2.rectangle(self.image, (int(x), int(y)), (int(x + w), int(y + h)), (125, 125, 125), 1)