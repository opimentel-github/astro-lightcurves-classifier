from __future__ import print_function
from __future__ import division
from . import C_

import torch
import numpy as np
from fuzzytorch.metrics import FTMetric, MetricResult
import fuzzytorch.models.seq_utils as seq_utils
from .losses import LCXEntropy

MSE_K = C_.MSE_K
REC_LOSS_EPS = C_.REC_LOSS_EPS
REC_LOSS_K = C_.REC_LOSS_K
XENTROPY_K = C_.XENTROPY_K

###################################################################################################################################################

class LCWMSE(FTMetric):
	def __init__(self, name, band_names,
		balanced=True,
		k=MSE_K,
		**kwargs):
		self.name = name
		self.band_names = band_names
		self.balanced = balanced
		self.k = k

	def __call__(self, tdict:dict, **kwargs):
		mse_loss_bdict = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = tdict[f'input/onehot.{b}'][...,0] # (b,t)
			#p_rtime = tdict[f'input/rtime.{b}'][...,0] # (b,t)
			#p_dtime = tdict[f'input/dtime.{b}'][...,0] # (b,t)
			#p_x = tdict[f'input/x.{b}'] # (b,t,f)
			p_rerror = tdict[f'target/rerror.{b}'] # (b,t,1)
			p_rx = tdict[f'target/recx.{b}'] # (b,t,1)

			p_rx_pred = tdict[f'model/decx.{b}'] # (b,t,1)
			mse_loss_b = (p_rx-p_rx_pred)**2/(REC_LOSS_EPS+REC_LOSS_K*(p_rerror**2)) # (b,t,1)
			mse_loss_b = seq_utils.seq_avg_pooling(mse_loss_b, p_onehot)[...,0] # (b,t,1) > (b,t) > (b)
			mse_loss_bdict[b] = mse_loss_b

		mse_loss = torch.cat([mse_loss_bdict[b][...,None] for b in self.band_names], axis=-1).mean(dim=-1)*self.k # (b,d) > (b)
		if self.balanced:
			balanced_w = tdict[f'target/balanced_w'] # (b)
			mse_loss = mse_loss*balanced_w
			return MetricResult(mse_loss, reduction_mode='sum')
		else:
			return MetricResult(mse_loss)

class LCXEntropyMetric(FTMetric):
	def __init__(self, name,
		classifier_key='y.last',
		model_out_uses_softmax:bool=False,
		target_is_onehot:bool=False,
		balanced=True,
		k=XENTROPY_K,
		**kwargs):
		self.name = name
		self.xentropy = LCXEntropy('',
			model_out_uses_softmax,
			target_is_onehot,
			False, # uses_poblation_weights
			classifier_key,
			)
		self.balanced = balanced
		self.k = k

	def __call__(self, tdict:dict, **kwargs):
		epoch = kwargs['_epoch']
		xentropy_loss = self.xentropy(tdict, **kwargs)._batch_loss*self.k # (b)
		if self.balanced:
			balanced_w = tdict[f'target/balanced_w'] # (b)
			xentropy_loss = xentropy_loss*balanced_w
			return MetricResult(xentropy_loss, reduction_mode='sum')
		else:
			return MetricResult(xentropy_loss)

class LCAccuracy(FTMetric):
	def __init__(self, name,
		classifier_key='y.last',
		target_is_onehot:bool=False,
		balanced=True,
		**kwargs):
		self.name = name
		self.classifier_key = classifier_key
		self.target_is_onehot = target_is_onehot
		self.balanced = balanced

	def __call__(self, tdict:dict, **kwargs):
		epoch = kwargs['_epoch']
		y_target = tdict[f'target/y']
		y_pred = tdict[f'model/{self.classifier_key}']

		assert y_target.dtype==torch.long

		if self.target_is_onehot:
			assert y_pred.shape==y_target.shape
			y_target = y_target.argmax(dim=-1)
		
		y_pred = y_pred.argmax(dim=-1)
		assert y_pred.shape==y_target.shape
		assert len(y_pred.shape)==1

		accuracies = (y_pred==y_target).float()*100  # (b)
		if self.balanced:
			balanced_w = tdict[f'target/balanced_w'] # (b)
			accuracies = accuracies*balanced_w
			return MetricResult(accuracies, reduction_mode='sum')
		else:
			return MetricResult(accuracies)

class LCBinXEntropyMetric(FTMetric):
	def __init__(self, name,
		classifier_key='y.last',
		model_out_uses_softmax:bool=False,
		target_is_onehot:bool=False,
		balanced=True,
		k=XENTROPY_K,
		**kwargs):
		self.name = name
		self.xentropy = LCXEntropy('',
			model_out_uses_softmax,
			target_is_onehot,
			False, # uses_poblation_weights
			classifier_key,
			)
		self.balanced = balanced
		self.k = k

	def __call__(self, tdict:dict, **kwargs):
		epoch = kwargs['_epoch']
		xentropy_loss = self.xentropy(tdict, **kwargs)._batch_loss*self.k # (b)
		if self.balanced:
			balanced_w = tdict[f'target/balanced_w'] # (b)
			xentropy_loss = xentropy_loss*balanced_w
			return MetricResult(xentropy_loss, reduction_mode='sum')
		else:
			return MetricResult(xentropy_loss)