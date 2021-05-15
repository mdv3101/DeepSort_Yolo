import numpy as np
import cv2

def nms(boxes, max_bbox_overlap, scores=None):
    """Code referenced and updated from 
    http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float)
    pick = []
    x1, x2, y1, y2 = boxes[:, 0], boxes[:, 2] + boxes[:, 0], boxes[:, 1], boxes[:, 3] + boxes[:, 1]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is None:
        idxs = np.argsort(y2)
    else: 
        idxs = np.argsort(scores)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))
    return pick


class NNDistanceMetric(object):
	def __init__(self, matching_threshold, budget=None):
		self.matching_threshold = matching_threshold
		self.budget = budget
		self.samples = {}

	def cosine_distance(self, x, y):
		x = np.asarray(x) / np.linalg.norm(x, axis=1, keepdims=True)
		y = np.asarray(y) / np.linalg.norm(y, axis=1, keepdims=True)
		distances = 1. - np.dot(x, y.T)
		return distances.min(axis=0)

	def partial_fit(self, features, targets, active_targets):
		for feature, target in zip(features, targets):
			self.samples.setdefault(target, []).append(feature)
			if self.budget is not None:
				self.samples[target] = self.samples[target][-self.budget:]
		self.samples = {k: self.samples[k] for k in active_targets}

	def distance(self, features, targets):
		cost_matrix = np.zeros((len(targets), len(features)))
		for i, target in enumerate(targets):
			cost_matrix[i, :] = self.cosine_distance(self.samples[target], features)
		return cost_matrix
