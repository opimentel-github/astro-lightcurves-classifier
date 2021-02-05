from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer

###################################################################################################################################################

class LogStandardScaler():
	def __init__(self,
		eps=C_.EPS,
		):
		self.eps = eps
		self.scaler = StandardScaler()

	def fit(self, x):
		assert np.all(x>=0)
		assert len(x.shape)==2
		new_x = np.log(x+self.eps)
		self.scaler.fit(new_x)
		
	def transform(self, x):
		new_x = np.log(x+self.eps)
		y = self.scaler.transform(new_x)
		return y
	
	def inverse_transform(self, y):
		new_y = self.scaler.inverse_transform(y)
		x = np.exp(new_y)-self.eps
		return x

class LogQuantileTransformer():
	def __init__(self,
		n_quantiles=5000,
		random_state=0,
		eps=C_.EPS,
		):
		self.eps = eps
		self.scaler = QuantileTransformer(n_quantiles=n_quantiles, random_state=random_state, output_distribution='normal')

	def fit(self, x):
		assert np.all(x>=0)
		assert len(x.shape)==2
		new_x = np.log(x+self.eps)
		self.scaler.fit(new_x)
		
	def transform(self, x):
		new_x = np.log(x+self.eps)
		y = self.scaler.transform(new_x)
		return y
	
	def inverse_transform(self, y):
		new_y = self.scaler.inverse_transform(y)
		x = np.exp(new_y)-self.eps
		return x