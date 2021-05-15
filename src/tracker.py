import numpy as np
from . import kalman_filter
from . import assignment

class Track:
	def __init__(self,mean,cov,id,min_det,max_age,feature=None):
		self.mean = mean
		self.cov = cov
		self.min_det = min_det
		self.max_age = max_age
		self.id = id
		self.features = []
		self.age = 1
		#Tentative = 1, Confirmed = 2, Deleted = 3
		self.state = 1 
		self.last_update = 0
		self.hits = 1
		if feature is not None:
			self.features.append(feature)

	def to_bbox(self):
		##c_x,c_y,a,h => x,y,w,h
		ret = self.mean[:4].copy()
		c_x,c_y,a,h = ret[0],ret[1],ret[2],ret[3]
		w = a*h
		x = c_x -(w/2)
		y = c_y - (h/2)
		new_cord = [x,y,w,h]
		return np.array(new_cord)

	def to_bbox2(self):
		## minx,miny,maxx,maxy
		coord = self.to_bbox()
		coord[2:] = coord[:2] + coord[2:]
		return coord    		
	
	def update_tracker(self,kf,trk_det):
		self.mean,self.cov = kf.update_tracker(self.mean,self.cov,trk_det.convert_bbox())
		self.features.append(trk_det.feature)
		self.last_update = 0
		self.hits+=1
		if self.state == 1 and self.hits>=self.min_det:
			self.state = 2

	def predict_tracker(self,kf):
		self.mean,self.cov = kf.predict_tracker(self.mean,self.cov)
		self.last_update+=1
		self.age+=1	


class Detection(object):
	def __init__(self,bbox,score,feature):
		##bbox = x,y,w,h
		self.score = float(score)
		self.bbox = np.asarray(bbox, dtype=np.float)
		self.feature = np.asarray(feature, dtype=np.float32)
	
	def convert_bbox(self):
		##x,y,w,h => c_x,c_y,a,h
		ret = self.bbox.copy()
		x,y,w,h = ret[:4]
		c_x = x+(w/2)
		c_y = y+(h/2)
		a = w/h
		new_cord = [c_x,c_y,a,h]
		return np.array(new_cord)
		
class Tracker:
	def __init__(self,NN_metric,iou_threshold=0.7,max_age=30,min_det = 3):
		self.NN_metric = NN_metric
		self.kf = kalman_filter.KalmanFilter()	
		self.track_list = []		
		self.iou_threshold = iou_threshold
		self.track_id = 1
		self.max_age = max_age
		self.min_det = min_det

			
	def create_tracker(self,trk_det):
		mean, cov = self.kf.create_tracker(trk_det.convert_bbox())
		new_trk = Track(mean, cov, self.track_id, self.min_det, self.max_age,trk_det.feature)
		self.track_list.append(new_trk)
		self.track_id += 1

	def predict_tracker(self):
		for track in self.track_list:
			track.predict_tracker(self.kf)

	def match_tracker(self,trk_det):
		cnf_trk = []
		u_trk = []
		for i,trk in enumerate(self.track_list):
			if trk.state == 2:
				cnf_trk.append(i)
			else:
				u_trk.append(i)           
		
		##Compute gate matrix B
		def gate_metric(track_list, dets, track_idx, det_idx):
			features = np.array([dets[i].feature for i in det_idx])
			trk_ids = np.array([track_list[i].id for i in track_idx])
			cost_matrix = self.NN_metric.distance(features, trk_ids)
			return assignment.gate_cost_matrix(self.kf, cost_matrix, track_list, dets, track_idx, det_idx)	
			
		match_a, u_track_a, u_det_a = assignment.matching_cascade(gate_metric, self.NN_metric.matching_threshold, self.max_age,self.track_list, trk_det, cnf_trk)
		i_trk_u = []
		for i in u_track_a:
			if self.track_list[i].last_update == 1:
				i_trk_u.append(i)
		iou_matching_u = u_trk + i_trk_u
		
		## Filter unmatched trackers having last update
		u_track_a = []
		for i in u_track_a:
			if self.track_list[i].last_update != 1:
				u_track_a.append(i)
		match_b, u_track_b, u_det_a = assignment.min_cost_matching(assignment.iou_cost, self.iou_threshold, self.track_list,trk_det, iou_matching_u, u_det_a)
		
		matches = match_a + match_b
		return matches, list(set(u_track_a + u_track_b)),u_det_a


	def update_tracker(self,trk_det):
		match, u_track,u_det = self.match_tracker(trk_det)
		for t_idx,d_idx in match:
			self.track_list[t_idx].update_tracker(self.kf,trk_det[d_idx])
		for t_idx in u_track:
			if self.track_list[t_idx].state == 1:
				self.track_list[t_idx].state = 3
			if self.track_list[t_idx].last_update >self.track_list[t_idx].max_age:
				self.track_list[t_idx].state = 3
		for d_idx in u_det:
			self.create_tracker(trk_det[d_idx])
		self.track_list = [t for t in self.track_list if not t.state ==3]
		features = []
		t_id = []
		for track in self.track_list:
			if track.state == 2:
				features+=track.features
				t_id +=[track.id for _ in track.features]
				track.features = []
		features = np.asarray(features)
		t_id  = np.asarray(t_id)
		id_conf = [t.id for t in self.track_list if t.state==2]
		self.NN_metric.partial_fit(features,t_id,id_conf)