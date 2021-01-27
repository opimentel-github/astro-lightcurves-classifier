from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
from fuzzytorch.models.basics import MLP

###################################################################################################################################################

class SimpleClassifier(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()

		### ATTRIBUTES
		for name, val in kwargs.items():
			setattr(self, name, val)

		### MLP
		self.embd_dims_list = [self.input_dims]*self.embd_layers
		mlp_kwargs = {
			'activation':'relu',
			'last_activation':'linear',
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
		}
		self.mlp = MLP(self.input_dims, self.output_dims, self.embd_dims_list, **mlp_kwargs)
		print('classifier_mlp:', self.mlp)

	def get_output_dims(self):
		return self.mlp.get_output_dims()

	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		tdict['model']['y.last'] = self.mlp(tdict['model']['z.last'])
		return tdict