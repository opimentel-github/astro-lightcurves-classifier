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

		mlp_kwargs = {
			'activation':'relu',
			'last_activation':'linear',
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
		}
		self.classifier_mlp = MLP(self.input_dims, self.output_dims, [], **mlp_kwargs)
		print('classifier_mlp:', self.classifier_mlp)

		### MLP
		self.embd_dims_list = [self.input_dims*2]*self.embd_layers
		self.classifier_mlp_ft = MLP(self.input_dims, self.output_dims, self.embd_dims_list, **mlp_kwargs)
		print('classifier_mlp_ft:', self.classifier_mlp_ft)

	def get_output_dims(self):
		return self.classifier_mlp.get_output_dims()

	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		tdict['model']['y.last'] = self.classifier_mlp(tdict['model']['z.last'])
		tdict['model']['y.last-ft'] = self.classifier_mlp_ft(tdict['model']['z.last'])
		return tdict