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

		### PRE-INPUT
		linear_kwargs = {
			'activation':'linear',
		}
		extra_dims = 0
		self.x_projection = nn.ModuleDict({b:Linear(self.input_dims+extra_dims, self.rnn_embd_dims, **linear_kwargs) for b in self.band_names})
		print('x_projection:', self.x_projection)

		### TE
		self.te_film = FILM(self.te_features, self.rnn_embd_dims)
		print('te_film:', self.te_film)

		### RNN STACK
		rnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'bidirectional':self.bidirectional,
			'uses_batchnorm':self.uses_batchnorm,
		}
		self.ml_rnn = nn.ModuleDict({b:getattr(ft_rnn, f'ML{self.rnn_cell_name}')(self.rnn_embd_dims, self.rnn_embd_dims, [self.rnn_embd_dims]*(self.rnn_layers-1), **rnn_kwargs) for b in self.band_names})
		print('ml_rnn:', self.ml_rnn)

		### PARALLEL PATCH
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

		last_z_dic = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = seq_utils.serial_to_parallel(onehot, onehot[...,kb])[...,kb] # (b,t)
			p_x = seq_utils.serial_to_parallel(x, onehot[...,kb])
			p_z = self.x_projection[b](p_x)
			p_z = self.te_film(p_z, seq_utils.serial_to_parallel(model_input['te'], onehot[...,kb])) if self.te_features>0 else p_z

			p_z, p_extra_info_rnn = self.ml_rnn[b](p_z, p_onehot, **kwargs) # out, (ht, ct)
			tdict['model'].update({f'z.{b}.last':p_z})

			### get last element
			last_z_dic[b] = seq_utils.seq_last_element(p_z, p_onehot) # get last value of sequence according to onehot

		last_z = torch.cat([last_z_dic[b] for b in self.band_names], dim=-1)
		last_z = self.z_projection(last_z)
		tdict['model'].update({
			#'z':z, # not used
			'z.last':last_z,
		})
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

		### TE
		self.te_film = FILM(self.te_features, self.rnn_embd_dims)
		print('te_film:', self.te_film)

		### RNN STACK
		rnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'bidirectional':self.bidirectional,
			'uses_batchnorm':self.uses_batchnorm,
		}
		self.ml_rnn = getattr(ft_rnn, f'ML{self.rnn_cell_name}')(self.rnn_embd_dims, self.rnn_embd_dims, [self.rnn_embd_dims]*(self.rnn_layers-1), **rnn_kwargs)
		print('ml_rnn:', self.ml_rnn)

	def get_output_dims(self):
		return self.rnn_embd_dims
	
	def get_embd_dims_list(self):
		return self.ml_rnn.get_embd_dims_list()

	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		model_input = tdict['input']
		x = model_input['x']
		onehot = model_input['onehot']

		z = self.x_projection(torch.cat([x, onehot.float()], dim=-1))

		if self.te_features>0:
			te = model_input['te']
			z = self.te_film(z, te)
		else:
			pass

		s_onehot = onehot.sum(dim=-1).bool()
		z, extra_info_rnn = self.ml_rnn(z, s_onehot, **kwargs) # out, (ht, ct)
		last_z = seq_utils.seq_last_element(z, s_onehot) # get last value of sequence according to onehot

		tdict['model'].update({
			#'z':z, # not used
			'z.last':last_z,
		})
		return tdict