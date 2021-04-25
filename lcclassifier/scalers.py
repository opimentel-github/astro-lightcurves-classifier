from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from numba import jit
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import flamingchoripan.numba as fcnumba

###################################################################################################################################################
ESC_EPS = 1. # ***
class CustomStandardScaler(): # faster than numba implementation :(
	def __init__(self,
		):
		self.reset()
		
	def reset(self):
		self.scaler = StandardScaler()

	def fit(self, x):
		assert len(x.shape)==2 # (b,f)
		self.scaler.fit(x)
		
	def transform(self, x):
		assert len(x.shape)==2 # (b,f)
		z = self.scaler.transform(x)
		return z
	
	def inverse_transform(self, z):
		assert len(z.shape)==2 # (b,f)
		x = self.scaler.inverse_transform(z)
		return x

###################################################################################################################################################

class LogStandardScaler(): # faster than numba implementation :(
	def __init__(self,
		eps=ESC_EPS,
		):
		self.eps = eps
		self.reset()
		
	def reset(self):
		self.scaler = StandardScaler()

	def fit(self, x):
		assert np.all(x>=0)
		assert len(x.shape)==2 # (b,f)
		log_x = fcnumba.log(x, self.eps)
		self.scaler.fit(log_x)
		
	def transform(self, x):
		assert np.all(x>=0)
		assert len(x.shape)==2 # (b,f)
		log_x = fcnumba.log(x, self.eps)
		z = self.scaler.transform(log_x)
		return z
	
	def inverse_transform(self, z):
		assert len(z.shape)==2 # (b,f)
		log_x = self.scaler.inverse_transform(z)
		x = np.exp(log_x)-self.eps
		return x

###################################################################################################################################################

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
		return x