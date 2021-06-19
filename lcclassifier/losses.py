from __future__ import print_function
from __future__ import division
from . import C_

import math
import torch
import torch.nn.functional as F
from fuzzytorch.losses import FTLoss
import fuzzytorch.models.seq_utils as seq_utils
import math

XENTROPY_K = C_.XENTROPY_K
MSE_K = C_.MSE_K

###################################################################################################################################################

# class LCXEntropy(FTLoss):
# 	def __init__(self, name,
# 		class_names=None,
# 		target_is_onehot:bool=False,
# 		uses_poblation_weights:bool=False,
# 		classifier_key='y_last_pt',
# 		):
# 		self.name = name
# 		self.class_names = class_names
# 		self.target_is_onehot = target_is_onehot
# 		self.uses_poblation_weights = uses_poblation_weights
# 		self.classifier_key = classifier_key

# 	def __call__(self, tdict:dict,
# 		**kwargs):
# 		y_target = tdict[f'target/y'].long()
# 		y_pred = tdict[f'model/{self.classifier_key}']
# 		poblation_weights = tdict[f'target/poblation_weights'][0] if self.uses_poblation_weights else None
# 		xentropy_loss = batch_xentropy(y_pred, y_target, False, self.target_is_onehot, poblation_weights) # (b)
# 		#print(xentropy_loss.shape)

# 		loss_res = LossResult(xentropy_loss)
# 		return loss_res

###################################################################################################################################################

class LCMSEReconstruction(FTLoss):
	def __init__(self, name, weight_key,
		band_names=None,
		**kwargs):
		super().__init__(name, weight_key)
		self.band_names = band_names

	def compute_loss(self, tdict,
		**kwargs):
		mse_loss_bdict = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = tdict[f'input/onehot.{b}'][...,0] # (b,t)
			#p_rtime = tdict[f'input/rtime.{b}'][...,0] # (b,t)
			#p_dtime = tdict[f'input/dtime.{b}'][...,0] # (b,t)
			#p_x = tdict[f'input/x.{b}'] # (b,t,f)
			p_rerror = tdict[f'target/rerror.{b}'] # (b,t,1)
			p_rx = tdict[f'target/recx.{b}'] # (b,t,1)

			p_rx_pred = tdict[f'model/decx.{b}'] # (b,t,1)
			mse_loss_b = (p_rx-p_rx_pred)**2/(C_.REC_LOSS_EPS+C_.REC_LOSS_K*(p_rerror**2)) # (b,t,1)
			mse_loss_b = seq_utils.seq_avg_pooling(mse_loss_b, p_onehot)[...,0] # (b,t,1) > (b,t) > (b)
			mse_loss_bdict[b] = mse_loss_b

		mse_loss = torch.cat([mse_loss_bdict[b][...,None] for b in self.band_names], axis=-1).mean(dim=-1) # (b,d) > (b)
		return mse_loss

###################################################################################################################################################

class LCBinXEntropy(FTLoss):
	def __init__(self, name, weight_key,
		class_names=None,
		target_is_onehot:bool=False,
		target_y_key='target/y',
		pred_y_key='model/y',
		**kwargs):
		super().__init__(name, weight_key)
		self.class_names = class_names
		self.target_is_onehot = target_is_onehot
		self.target_y_key = target_y_key
		self.pred_y_key = pred_y_key
		self.reset()

	def reset(self):
		self.bin_loss = torch.nn.BCELoss(reduction='none')

	def get_onehot(self, y):
		class_count = torch.max(y)[0] if self.class_names is None else len(self.class_names)
		return torch.eye(class_count, device=y.device)[y,:]

	def compute_loss(self, tdict,
		**kwargs):
		y_target = tdict[self.target_y_key].long() # (b)
		y_pred = tdict[self.pred_y_key] # (b,c)

		y_target = y_target if self.target_is_onehot else self.get_onehot(y_target) # (b,c)
		binxentropy_loss = self.bin_loss(torch.sigmoid(y_pred), y_target) # (b,c)
		binxentropy_loss = binxentropy_loss.mean(dim=-1) # (b,c) > (b)
		return binxentropy_loss

###################################################################################################################################################

class LCCompleteLoss(FTLoss):
	def __init__(self, name, weight_key,
		band_names=None,
		class_names=None,
		target_is_onehot:bool=False,
		target_y_key='target/y',
		pred_y_key='model/y',
		binxentropy_k=XENTROPY_K,
		mse_k=MSE_K,
		**kwargs):
		super().__init__(name, weight_key)
		self.band_names = band_names
		self.class_names = class_names
		self.target_is_onehot = target_is_onehot
		self.target_y_key = target_y_key
		self.pred_y_key = pred_y_key
		self.binxentropy_k = binxentropy_k
		self.mse_k = mse_k
		self.reset()

	def reset(self):
		self.binxentropy = LCBinXEntropy('', None,
			self.class_names,
			self.target_is_onehot,
			self.target_y_key,
			self.pred_y_key,
			)
		self.mse = LCMSEReconstruction('', None,
			self.band_names,
			)

	def compute_loss(self, tdict,
		**kwargs):
		binxentropy_loss = self.binxentropy.compute_loss(tdict, **kwargs)*self.binxentropy_k # (b)
		mse_loss = self.mse.compute_loss(tdict, **kwargs)*self.mse_k # (b)
		d = {
			'_loss':binxentropy_loss+mse_loss,
			'binxentropy':binxentropy_loss,
			'mse':mse_loss,
			}
		return d
