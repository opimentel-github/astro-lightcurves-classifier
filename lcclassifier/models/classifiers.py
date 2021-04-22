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
		layers = 2
		self.embd_dims_list = [self.input_dims*(1+layers)]*1
		self.classifier_mlp_ft = MLP(self.embd_dims_list[0], self.output_dims, self.embd_dims_list, **mlp_kwargs)
		print('classifier_mlp_ft:', self.classifier_mlp_ft)

	def get_output_dims(self):
		return self.classifier_mlp.get_output_dims()

	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']

		z_last = tdict['model']['z_last']
		tdict['model']['y_last_pt'] = self.classifier_mlp(z_last)

		layers = 2
		complete_z = [z_last]+[tdict['model'][f'z-{layer}'] for layer in range(0, layers)]
		complete_z = torch.cat(complete_z, dim=-1)
		tdict['model']['y_last_ft'] = self.classifier_mlp_ft(complete_z)
		return tdict