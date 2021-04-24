from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
import torch.nn.functional as F 
import fuzzytorch.models.rnn.basics as ft_rnn
from fuzzytorch.models.basics import MLP, Linear
from fuzzytorch.models.others import FILM
import fuzzytorch.models.seq_utils as seq_utils

###################################################################################################################################################

class RNNEncoderP(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		### ATTRIBUTES
		setattr(self, 'uses_batchnorm', False)
		setattr(self, 'bidirectional', False)
		for name, val in kwargs.items():
			setattr(self, name, val)
		self.reset()

	def reset(self):
		### PRE-INPUT
		linear_kwargs = {
			'activation':'linear',
		}
		extra_dims = 0
		self.x_projection = nn.ModuleDict({b:Linear(self.input_dims+extra_dims, self.rnn_embd_dims, **linear_kwargs) for b in self.band_names})
		print('x_projection:', self.x_projection)

		### RNN STACK
		rnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'bidirectional':self.bidirectional,
			'uses_batchnorm':self.uses_batchnorm,
			'activation':'relu',
			'last_activation':'relu',
		}
		self.ml_rnn = nn.ModuleDict({b:getattr(ft_rnn, f'ML{self.rnn_cell_name}')(self.rnn_embd_dims, self.rnn_embd_dims, [self.rnn_embd_dims]*(self.rnn_layers-1), **rnn_kwargs) for b in self.band_names})
		print('ml_rnn:', self.ml_rnn)

		### POST-PROJECTION
		linear_kwargs = {
			'activation':'linear',
		}
		self.z_projection = Linear(self.rnn_embd_dims*len(self.band_names), self.rnn_embd_dims, **linear_kwargs)
		print('z_projection:', self.z_projection)

	def get_output_dims(self):
		return self.rnn_embd_dims*len(self.band_names)
	
	def get_embd_dims_list(self):
		return {b:self.ml_rnn[b].get_embd_dims_list() for b in self.band_names}

	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		model_input = tdict['input']
		x = model_input['x']
		onehot = model_input['onehot']

		z_bdict = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = seq_utils.serial_to_parallel(onehot, onehot[...,kb])[...,kb] # (b,t)
			p_x = seq_utils.serial_to_parallel(x, onehot[...,kb])
			p_z = self.x_projection[b](p_x)
			p_zs, _ = self.ml_rnn[b](p_z, p_onehot, **kwargs) # out, (ht, ct)

			### representative element
			for layer in range(0, self.rnn_layers):
				z_bdict[f'z-{layer}.{b}'] = seq_utils.seq_last_element(p_zs[layer], p_onehot) # last element

		### BUILD OUT
		z_last = torch.cat([z_bdict[f'z-{self.rnn_layers-1}.{b}'] for b in self.band_names], dim=-1)
		tdict['model']['z_last'] = self.z_projection(z_last)
		for layer in range(0, self.rnn_layers):
			tdict['model'][f'z-{layer}'] = torch.mean(torch.cat([z_bdict[f'z-{layer}.{b}'][...,None] for b in self.band_names], dim=-1), dim=-1)
		return tdict

###################################################################################################################################################

class RNNEncoderS(nn.Module):
	def __init__(self,
		**kwargs):
		super().__init__()

		### ATTRIBUTES
		setattr(self, 'uses_batchnorm', False)
		setattr(self, 'bidirectional', False)
		for name, val in kwargs.items():
			setattr(self, name, val)

		### PRE-INPUT
		linear_kwargs = {
			'activation':'linear',
		}
		extra_dims = len(self.band_names)
		self.x_projection = Linear(self.input_dims+extra_dims, self.rnn_embd_dims, **linear_kwargs)
		print('x_projection:', self.x_projection)

		### RNN STACK
		rnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'bidirectional':self.bidirectional,
			'uses_batchnorm':self.uses_batchnorm,
			'activation':'relu',
			'last_activation':'relu',
		}
		self.ml_rnn = getattr(ft_rnn, f'ML{self.rnn_cell_name}')(self.rnn_embd_dims, self.rnn_embd_dims, [self.rnn_embd_dims]*(self.rnn_layers-1), **rnn_kwargs)
		print('ml_rnn:', self.ml_rnn)

		### POST-PROJECTION
		linear_kwargs = {
			'activation':'linear',
		}
		self.z_projection = Linear(self.rnn_embd_dims, self.rnn_embd_dims, **linear_kwargs)
		print('z_projection:', self.z_projection)

	def get_output_dims(self):
		return self.rnn_embd_dims
	
	def get_embd_dims_list(self):
		return self.ml_rnn.get_embd_dims_list()

	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		model_input = tdict['input']
		x = model_input['x']
		onehot = model_input['onehot']
		s_onehot = onehot.sum(dim=-1).bool()

		z = self.x_projection(torch.cat([x, onehot.float()], dim=-1))
		zs, _ = self.ml_rnn(z, s_onehot, **kwargs) # out, (ht, ct)

		z_bdict = {}
		### representative element
		for layer in range(0, self.rnn_layers):
			z_bdict[f'z-{layer}'] = seq_utils.seq_last_element(zs[layer], s_onehot) # last element

		### BUILD OUT
		tdict['model']['z_last'] = self.z_projection(z_bdict[f'z-{self.rnn_layers-1}'])
		for layer in range(0, self.rnn_layers):
			tdict['model'][f'z-{layer}'] = z_bdict[f'z-{layer}']
		return tdict