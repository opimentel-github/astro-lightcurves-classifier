from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer

###################################################################################################################################################
ESC_EPS = 1. # ***

class CustomStandardScaler():
	def __init__(self,
		eps=1e-5,
		):
		self.eps = eps
		self.reset()
		
	def reset(self):
		pass

	def fit(self, x):
		assert len(x.shape)==2 # (b,f)
		self.m = x.mean(axis=0)[None].astype(x.dtype)
		self.s = x.std(axis=0)[None].astype(x.dtype)

	def transform(self, x):
		assert len(x.shape)==2 # (b,f)
		z = (x-self.m)/(self.s+self.eps)
		return z

	def inverse_transform(self, z):
		assert len(z.shape)==2 # (b,f)
		x = z*(self.s+self.eps)+self.m
		return x

###################################################################################################################################################

class LogStandardScaler():
	def __init__(self,
		eps=ESC_EPS,
		):
		self.eps = eps
		self.reset()

	def reset(self):
		self.scaler = CustomStandardScaler()

	def fit(self, x):
		assert np.all(x>=0)
		log_x = np.log(x+self.eps)
		self.scaler.fit(log_x)
		
	def transform(self, x):
		assert np.all(x>=0)
		log_x = np.log(x+self.eps)
		z = self.scaler.transform(log_x)
		return z
	
	def inverse_transform(self, z):
		log_x = self.scaler.inverse_transform(z)
		x = np.exp(log_x)-self.eps
		return x

###################################################################################################################################################

'''
class LogQuantileTransformer():
	def __init__(self,
		n_quantiles=5000,
		random_state=0,
		eps=ESC_EPS,
		):
		self.eps = eps
		self.reset()

	def reset(self):
		self.scaler = QuantileTransformer(n_quantiles=n_quantiles, random_state=random_state, output_distribution='normal')

	def fit(self, x):
		assert np.all(x>=0)
		assert len(x.shape)==2
		log_x = fcnumba.log(x+self.eps)
		self.scaler.fit(log_x)
		
	def transform(self, x):
		assert np.all(x>=0)
		assert len(x.shape)==2 # (b,f)
		log_x = fcnumba.log(x+self.eps)
		z = self.scaler.transform(log_x)
		return z
	
	def inverse_transform(self, z):
		assert len(z.shape)==2 # (b,f)
		log_x = self.scaler.inverse_transform(z)
		x = np.exp(log_x)-self.eps
		return x'''