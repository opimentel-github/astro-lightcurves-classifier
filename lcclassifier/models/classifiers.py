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
		self.layers = 1
		self.k = 2
		#self.classifiers_mlp_ft = nn.ModuleList([MLP(self.input_dims, 1, [self.input_dims*self.k]*self.layers, **mlp_kwargs) for _ in range(0, self.output_dims)])
		#print('classifiers_mlp_ft:', self.classifiers_mlp_ft)
		self.classifier_mlp_ft = MLP(self.input_dims, self.output_dims, [self.input_dims*self.k]*self.layers, **mlp_kwargs)
		print('classifier_mlp_ft:', self.classifier_mlp_ft)

	def get_output_dims(self):
		return self.output_dims

	def forward(self, tdict:dict, **kwargs):
		z_last = tdict['model']['z_last']
		#z_last = torch.cat([classifier_mlp_ft(z_last) for classifier_mlp_ft in self.classifiers_mlp_ft], dim=-1)
		z_last = self.classifier_mlp_ft(z_last)
		#print(z_last.shape)
		tdict['model']['y_last_ft'] = z_last
		return tdict

###################################################################################################################################################

'''class SimpleClassifier2(nn.Module):
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
		#self.encoder_layers = 2 # fixme
		#self.embd_dims_list = [self.input_dims*(1+self.encoder_layers)]*1
		#self.classifier_mlp_ft = MLP(self.embd_dims_list[0], self.output_dims, self.embd_dims_list, **mlp_kwargs)
		self.layers = 2
		self.k = 2
		self.classifier_mlp_ft = MLP(self.input_dims, self.output_dims, [self.input_dims*self.k]*self.layers, **mlp_kwargs)
		print('classifier_mlp_ft:', self.classifier_mlp_ft)

	def get_output_dims(self):
		return self.classifier_mlp_ft.get_output_dims()

	def forward(self, tdict:dict, **kwargs):
		z_last = tdict['model']['z_last']
		#complete_z = [z_last]+[tdict['model'][f'z-{layer}'] for layer in range(0, self.encoder_layers)]
		#complete_z = torch.cat(complete_z, dim=-1)
		#tdict['model']['y_last_ft'] = self.classifier_mlp_ft(complete_z)
		z_last = self.classifier_mlp_ft(z_last)
		z_last = torch.sigmoid(z_last) if 1 else z_last
		tdict['model']['y_last_ft'] = z_last
		return tdict'''