import numpy as np
import scipy

class KalmanFilter(object):
	def __init__(self):
		self.H = np.eye(8,8)
		for i in range(4):
			self.H[i,4+i] = 1.
		self.update_ = np.eye(4,8)

	def get_projection(self,mean,cov):
		h_m =  float(mean[3])
		std_pos = [h_m/20,h_m/20,1e-1,h_m/20]
		mean_pro = np.dot(self.update_, mean)
		cov_pro = np.linalg.multi_dot((self.update_, cov, self.update_.T)) + np.diag(np.square(std_pos))
		#print(cov,"\n")
		return mean_pro,cov_pro

	def gating_distance(self,mean,cov,trk_pos,only_position=False):
		#print(cov)    
		mean_pro, cov_pro = self.get_projection(mean,cov)
		if only_position:
				mean_pro, cov_pro = mean_pro[:2], cov_pro[:2, :2]
				trk_pos = trk_pos[:, :2]
		chol_factor = np.linalg.cholesky(cov_pro)
		#print(np.linalg.eigvalsh(cov_pro),chol_factor)
		dis = trk_pos - mean_pro
		dis_cal = scipy.linalg.solve_triangular(chol_factor, dis.T, lower=True, check_finite=False,overwrite_b=True)
		return np.sum(dis_cal * dis_cal, axis=0)

	def update_tracker(self,mean,cov,trk_pos):
		mean_pro,cov_pro = self.get_projection(mean,cov)
		trk_change = trk_pos - mean_pro
		chol_factor, lower = scipy.linalg.cho_factor(cov_pro, lower=True, check_finite=False)
		kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(cov, self.update_.T).T,check_finite=False).T
		## x = x + K(y-H.x)
		new_mean = mean + np.dot(trk_change, kalman_gain.T)
		new_cov = cov - np.linalg.multi_dot((kalman_gain, cov_pro, kalman_gain.T))
		return new_mean,new_cov	

	def create_tracker(self,trk_pos):
		h =  float(trk_pos[3])
		mean = np.r_[trk_pos,np.zeros_like(trk_pos)]
		std_deviation = [h/10,h/10,1e-2,h/10,h/16,h/16,1e-5,h/16]
		return mean, np.diag(np.square(std_deviation))

	def predict_tracker(self,mean_prev,cov_prev):
		## K = (P.H)/(H.P.H^T + R)
		mean_pred = np.dot(self.H,mean_prev)		
		h_m  = float(mean_prev[3])
		std_dev = [h_m/20,h_m/20,1e-2,h_m/20,h_m/160,h_m/160,1e-5,h_m/160]
		R = np.diag(np.square(std_dev))
		cov_pred = np.linalg.multi_dot((self.H,cov_prev,self.H.T))+R
		return mean_pred,cov_pred
		
