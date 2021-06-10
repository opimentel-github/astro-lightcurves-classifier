from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
from fuzzytorch.models.basics import MLP, Linear

###################################################################################################################################################

class SimpleClassifier(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		### ATTRIBUTES
		for name, val in kwargs.items():
			setattr(self, name, val)
		self.reset()

	def reset(self):
		### MLP
		mlp_kwargs = {
			'activation':'relu',
			'last_activation':'linear',
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			}
		# self.layers = 2
		self.k = 1
		#self.classifiers_mlp_ft = nn.ModuleList([MLP(self.input_dims, 1, [self.input_dims*self.k]*self.layers, **mlp_kwargs) for _ in range(0, self.output_dims)])
		#print('classifiers_mlp_ft:', self.classifiers_mlp_ft)
		self.classifier_mlp_ft = MLP(self.input_dims, self.output_dims, [self.input_dims*self.k]*self.layers, **mlp_kwargs)
		print('classifier_mlp_ft:', self.classifier_mlp_ft)
		self.reset_parameters()

	def reset_parameters(self):
		self.classifier_mlp_ft.reset_parameters()

	def get_output_dims(self):
		return self.output_dims

	def forward(self, tdict:dict, **kwargs):
		encz_last = tdict['model']['encz_last']
		#encz_last = torch.cat([classifier_mlp_ft(encz_last) for classifier_mlp_ft in self.classifiers_mlp_ft], dim=-1)
		encz_last = self.classifier_mlp_ft(encz_last)
		#print(encz_last.shape)
		tdict['model']['y_last_ft'] = encz_last
		return tdict