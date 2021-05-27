from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from fuzzytorch.utils import get_numpy_dtype, tensor_to_numpy
import torch

###################################################################################################################################################
ESC_EPS = 1. # ***

class CustomStandardScaler(): # faster than numba implementation :(
	def __init__(self,
		):
		self.reset()
		
	def reset(self):
		#self.scaler = StandardScaler()
		pass

	def aaaa(self, x):
		assert len(x.shape)==2 # (b,f)
		self.dtype = get_numpy_dtype(x.dtype)
		self.device = x.device
		self.scaler.fit(tensor_to_numpy(x))

	def xxx(self, x):
		assert len(x.shape)==2 # (b,f)
		z = self.scaler.transform(tensor_to_numpy(x)).astype(self.dtype)
		return torch.as_tensor(z).to(self.device)

	def fit(self, x):
		assert len(x.shape)==2 # (b,f)
		self.m = x.mean(0, keepdim=True)
		self.s = x.std(0, unbiased=False, keepdim=True)

	def transform(self, x):
		z = (x-m)/s
		return z

	def inverse_transform(self, z):
		assert len(z.shape)==2 # (b,f)
		x = self.scaler.inverse_transform(tensor_to_numpy(z)).astype(self.dtype)
		return torch.as_tensor(x).to(self.device)

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
		assert torch.all(x>=0)
		assert len(x.shape)==2 # (b,f)
		self.dtype = get_numpy_dtype(x.dtype)
		self.device = x.device
		log_x = torch.log(x+self.eps)
		self.scaler.fit(tensor_to_numpy(log_x))
		
	def transform(self, x):
		assert torch.all(x>=0)
		assert len(x.shape)==2 # (b,f)
		log_x = torch.log(x+self.eps)
		z = self.scaler.transform(tensor_to_numpy(log_x)).astype(self.dtype)
		return torch.as_tensor(z).to(self.device)
	
	def inverse_transform(self, z):
		assert len(z.shape)==2 # (b,f)
		log_x = self.scaler.inverse_transform(tensor_to_numpy(z)).astype(self.dtype)
		x = np.exp(log_x)-self.eps
		return torch.as_tensor(x).to(self.device)

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