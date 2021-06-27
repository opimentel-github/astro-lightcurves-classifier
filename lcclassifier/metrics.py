from __future__ import print_function
from __future__ import division
from . import C_

import torch
import numpy as np
from fuzzytorch.metrics import FTMetric
import fuzzytorch.models.seq_utils as seq_utils

###################################################################################################################################################

class LCAccuracy(FTMetric):
	def __init__(self, name, weight_key,
		target_is_onehot:bool=False,
		target_y_key='target/y',
		pred_y_key='model/y',
		**kwargs):
		super().__init__(name, weight_key)
		self.target_is_onehot = target_is_onehot
		self.target_y_key = target_y_key
		self.pred_y_key = pred_y_key

	def compute_metric(self, tdict,
		**kwargs):
		y_target = tdict[self.target_y_key] # (b)
		y_pred = tdict[self.pred_y_key] # (b,c)

		if self.target_is_onehot:
			assert y_pred.shape==y_target.shape
			y_target = y_target.argmax(dim=-1)
		
		y_pred = y_pred.argmax(dim=-1)
		assert y_pred.shape==y_target.shape
		assert len(y_pred.shape)==1

		accuracies = (y_pred==y_target).float()*100  # (b)
		return accuracies
