from __future__ import print_function
from __future__ import division
from . import C_

import torch
import numpy as np
from fuzzytorch.metrics import FTMetric, MetricResult
import fuzzytorch.models.seq_utils as seq_utils
from .losses import LCXEntropy

###################################################################################################################################################

class LCWMSE(FTMetric):
	def __init__(self, name, band_names,
		balanced=True,
		k=C_.MSE_K,
		**kwargs):
		self.name = name
		self.band_names = band_names
		self.balanced = balanced
		self.k = k

	def __call__(self, tdict, **kwargs):
		onehot = tdict['input']['onehot']
		time = tdict['input']['time']
		error = tdict['target']['error']
		assert torch.all(error>=0)

		t = onehot.shape[1]
		mse_loss_bdict = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = seq_utils.serial_to_parallel(onehot, onehot[...,kb])[...,kb] # (b,t)
			p_time = seq_utils.serial_to_parallel(time, onehot[...,kb]) # (b,t,1)
			p_error = seq_utils.serial_to_parallel(error, onehot[...,kb]) # (b,t,1)
			p_rx = seq_utils.serial_to_parallel(tdict['target']['rec_x'], onehot[...,kb]) # (b,t,1)
			p_rx_pred = tdict['model'][f'rec_x.{b}'] # (b,t,1)

			mse_loss_b = (p_rx-p_rx_pred)**2/(p_error**2+C_.REC_LOSS_EPS) # (b,t,1)
			mse_loss_b = seq_utils.seq_avg_pooling(mse_loss_b, p_onehot)[...,0] # (b,t,1) > (b,t) > (b)
			mse_loss_bdict[b] = mse_loss_b

		mse_loss = torch.cat([mse_loss_bdict[b][...,None] for b in self.band_names], axis=-1).mean(dim=-1)*self.k # (b,d) > (b)
		if self.balanced:
			balanced_w = tdict['target']['balanced_w']
			mse_loss = mse_loss*balanced_w[...,0]
			return MetricResult(mse_loss, reduction_mode='sum')
		else:
			return MetricResult(mse_loss)

class LCXEntropyMetric(FTMetric):
	def __init__(self, name,
		classifier_key='y.last',
		model_out_uses_softmax:bool=False,
		target_is_onehot:bool=False,
		balanced=True,
		k=C_.XENTROPY_K,
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

	def __call__(self, tdict, **kwargs):
		epoch = kwargs['_epoch']
		input_tdict = tdict['input']
		target_tdict = tdict['target']
		model_tdict = tdict['model']

		xentropy_loss = self.xentropy(tdict, **kwargs)._batch_loss*self.k # (b)
		if self.balanced:
			balanced_w = tdict['target']['balanced_w']
			xentropy_loss = xentropy_loss*balanced_w[...,0]
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

	def __call__(self, tdict, **kwargs):
		epoch = kwargs['_epoch']
		y_target = tdict['target']['y']
		y_pred = tdict['model'][self.classifier_key]

		assert y_target.dtype==torch.long

		if self.target_is_onehot:
			assert y_pred.shape==y_target.shape
			y_target = y_target.argmax(dim=-1)
		
		y_pred = y_pred.argmax(dim=-1)
		assert y_pred.shape==y_target.shape
		assert len(y_pred.shape)==1

		accuracies = (y_pred==y_target).float()*100  # (b)
		if self.balanced:
			balanced_w = tdict['target']['balanced_w']
			accuracies = accuracies*balanced_w[...,0]
			return MetricResult(accuracies, reduction_mode='sum')
		else:
			return MetricResult(accuracies)

class LCBinXEntropyMetric(FTMetric):
	def __init__(self, name,
		classifier_key='y.last',
		model_out_uses_softmax:bool=False,
		target_is_onehot:bool=False,
		balanced=True,
		k=C_.XENTROPY_K,
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

	def __call__(self, tdict, **kwargs):
		epoch = kwargs['_epoch']
		input_tdict = tdict['input']
		target_tdict = tdict['target']
		model_tdict = tdict['model']

		xentropy_loss = self.xentropy(tdict, **kwargs)._batch_loss*self.k # (b)
		if self.balanced:
			balanced_w = tdict['target']['balanced_w']
			xentropy_loss = xentropy_loss*balanced_w[...,0]
			return MetricResult(xentropy_loss, reduction_mode='sum')
		else:
			return MetricResult(xentropy_loss)