from __future__ import print_function
from __future__ import division
from . import C_

import torch
import numpy as np
from fuzzytorch.metrics import FTMetric, MetricResult, Accuracy
import fuzzytorch.models.seq_utils as seq_utils
from .losses import LCXEntropy

###################################################################################################################################################

class LCAccuracy(FTMetric):
	def __init__(self, name,
		target_is_onehot:bool=False,
		balanced=False,
		**kwargs):
		self.name = name
		self.target_is_onehot = target_is_onehot
		self.accuracy = Accuracy('',
			target_is_onehot,
			balanced,
			)

	def __call__(self, tdict, **kwargs):
		epoch = kwargs['__epoch__']
		input_tdict = tdict['input']
		target_tdict = tdict['target']
		model_tdict = tdict['model']

		new_tdict = {
			'model':{'y':model_tdict['y.last']},
			'target':{'y':target_tdict['y']},
		}
		return self.accuracy(new_tdict, **kwargs)

class LCXEntropyMetric(FTMetric):
	def __init__(self, name,
		model_out_uses_softmax:bool=False,
		target_is_onehot:bool=False,
		uses_poblation_weights:bool=True,
		k=C_.XENTROPY_K,
		**kwargs):
		self.name = name
		self.xentropy = LCXEntropy('',
			model_out_uses_softmax,
			target_is_onehot,
			uses_poblation_weights,
			)
		self.k = k

	def __call__(self, tdict, **kwargs):
		epoch = kwargs['__epoch__']
		input_tdict = tdict['input']
		target_tdict = tdict['target']
		model_tdict = tdict['model']

		xentropy_loss = self.xentropy(tdict, **kwargs).batch_loss_*self.k
		return MetricResult(xentropy_loss)