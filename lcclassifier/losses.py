from __future__ import print_function
from __future__ import division
from . import C_

import math
import torch
import torch.nn.functional as F
from fuzzytorch.losses import FTLoss, LossResult, batch_xentropy
import fuzzytorch.models.seq_utils as seq_utils
import math

###################################################################################################################################################

class LCMSEReconstruction(FTLoss):
	def __init__(self, name, band_names,
		**kwargs):
		self.name = name
		self.band_names = band_names

	def __call__(self, tdict, **kwargs):
		mse_loss_bdict = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = tdict['input'][f'onehot.{b}'][...,0] # (b,t)
			#p_rtime = tdict['input'][f'rtime.{b}'][...,0] # (b,t)
			#p_dtime = tdict['input'][f'dtime.{b}'][...,0] # (b,t)
			#p_x = tdict['input'][f'x.{b}'] # (b,t,f)
			p_rerror = tdict['target'][f'rerror.{b}'] # (b,t,1)
			p_rx = tdict['target'][f'recx.{b}'] # (b,t,1)

			p_rx_pred = tdict['model'][f'decx.{b}'] # (b,t,1)
			mse_loss_b = (p_rx-p_rx_pred)**2/(C_.REC_LOSS_EPS+C_.REC_LOSS_K*(p_rerror**2)) # (b,t,1)
			mse_loss_b = seq_utils.seq_avg_pooling(mse_loss_b, p_onehot)[...,0] # (b,t,1) > (b,t) > (b)
			mse_loss_bdict[b] = mse_loss_b

		mse_loss = torch.cat([mse_loss_bdict[b][...,None] for b in self.band_names], axis=-1).mean(dim=-1) # (b,d) > (b)
		loss_res = LossResult(mse_loss)
		return loss_res

class LCXEntropy(FTLoss):
	def __init__(self, name,
		model_out_uses_softmax:bool=False,
		target_is_onehot:bool=False,
		uses_poblation_weights:bool=False,
		classifier_key='y_last_pt',
		**kwargs):
		self.name = name
		self.model_out_uses_softmax = model_out_uses_softmax
		self.target_is_onehot = target_is_onehot
		self.uses_poblation_weights = uses_poblation_weights
		self.classifier_key = classifier_key

	def __call__(self, tdict, **kwargs):
		input_tdict = tdict['input']
		target_tdict = tdict['target']
		model_tdict = tdict['model']

		y_target = target_tdict['y'].long()
		y_pred = model_tdict[self.classifier_key]
		poblation_weights = target_tdict['poblation_weights'][0] if self.uses_poblation_weights else None
		xentropy_loss = batch_xentropy(y_pred, y_target, self.model_out_uses_softmax, self.target_is_onehot, poblation_weights) # (b)
		#print(xentropy_loss.shape)

		loss_res = LossResult(xentropy_loss)
		return loss_res

class LCCompleteLoss(FTLoss):
	def __init__(self, name, band_names,
		model_out_uses_softmax:bool=False,
		target_is_onehot:bool=False,
		uses_poblation_weights:bool=False,
		classifier_key='y_last_pt',
		xentropy_k=C_.XENTROPY_K,
		mse_k=C_.MSE_K,
		**kwargs):
		self.name = name
		self.xentropy = LCXEntropy('',
			model_out_uses_softmax,
			target_is_onehot,
			uses_poblation_weights,
			classifier_key,
			)
		self.mse = LCMSEReconstruction('', band_names)
		self.xentropy_k = xentropy_k
		self.mse_k = mse_k

	def __call__(self, tdict, **kwargs):
		epoch = kwargs['_epoch']
		input_tdict = tdict['input']
		target_tdict = tdict['target']
		model_tdict = tdict['model']

		xentropy_loss = self.xentropy(tdict, **kwargs)._batch_loss*self.xentropy_k
		mse_loss = self.mse(tdict, **kwargs)._batch_loss*self.mse_k
		loss_res = LossResult(xentropy_loss+mse_loss)
		loss_res.add_subloss('xentropy', xentropy_loss)
		loss_res.add_subloss('mse', mse_loss)
		return loss_res

###################################################################################################################################################

def get_onehot(y,
	class_count=None,
	):
	class_count = torch.max(y)[0] if class_count is None else class_count
	return torch.eye(class_count, device=y.device)[y,:]

class LCBinXEntropy(FTLoss):
	def __init__(self, name,
		class_names=None,
		model_out_uses_sigmoid:bool=False,
		target_is_onehot:bool=False,
		classifier_key='y_last_ft',
		**kwargs):
		self.name = name
		self.class_names = class_names
		self.model_out_uses_sigmoid = model_out_uses_sigmoid
		self.target_is_onehot = target_is_onehot
		self.classifier_key = classifier_key
		self.loss = torch.nn.BCELoss(reduction='none')

	def __call__(self, tdict, **kwargs):
		input_tdict = tdict['input']
		target_tdict = tdict['target']
		model_tdict = tdict['model']

		y_target = target_tdict['y'].long()
		#print(self.classifier_key)
		y_pred = model_tdict[self.classifier_key]
		#print('y_pred',y_pred[:10])
		#print(y_target.shape, y_target[0])
		y_target = y_target if self.target_is_onehot else get_onehot(y_target, None if self.class_names is None else len(self.class_names))
		#print(y_target.shape, y_target[0])
		y_pred = y_pred if self.model_out_uses_sigmoid else torch.sigmoid(y_pred)
		#print(y_pred.shape, y_pred[0])
		#assert 0
		xentropy_loss = self.loss(y_pred, y_target)
		#xentropy_loss = batch_binxentropy(y_pred, y_target, self.model_out_uses_softmax, self.target_is_onehot) # (b)
		
		xentropy_loss = xentropy_loss.mean(dim=-1)
		#print(xentropy_loss.shape,xentropy_loss[:10])

		loss_res = LossResult(xentropy_loss)
		return loss_res