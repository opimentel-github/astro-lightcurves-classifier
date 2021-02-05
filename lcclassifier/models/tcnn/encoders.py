from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
import torch.nn.functional as F 
import fuzzytorch.models.cnn.basics as ft_cnn
from fuzzytorch.models.basics import MLP, Linear
from fuzzytorch.models.others import FILM
import fuzzytorch.models.seq_utils as seq_utils

###################################################################################################################################################

class TCNNEncoderP(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()

		### ATRIBUTES
		#setattr(self, 'uses_batchnorm', False)
		#setattr(self, 'activation_name', 'relu') # tanh, relu
		#setattr(self, 'split_out_channels', 1) # no split
		#setattr(self, 'h_conditional_input_dims', 0) # no h
		for name, val in kwargs.items():
			setattr(self, name, val)

		### PRE-INPUT
		linear_kwargs = {
			'activation':'linear',
		}
		extra_dims = 0
		self.x_projection = nn.ModuleDict({b:Linear(self.input_dims+extra_dims, self.tcnn_embd_dims, **linear_kwargs) for b in self.band_names})
		print('x_projection:', self.x_projection)

		### TE
		self.te_film = FILM(self.te_features, self.tcnn_embd_dims)
		print('te_film:', self.te_film)

		### CNN STACK
		cnn_args = [self.tcnn_embd_dims, [self.curvelength_max], self.tcnn_embd_dims, [self.tcnn_embd_dims]*(self.tcnn_layers-1)] # input_dims:int, input_space:list, output_dims:int, embd_dims_list
		cnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'cnn_kwargs':{
				'kernel_size':5,
				'stride':1,
				'dilation':1,
			},
			'pool_kwargs':{
				'kernel_size':1,
				'stride':1,
				'dilation':1,
			},
			'padding_mode':'causal',
		}
		self.ml_cnn = nn.ModuleDict({b:ft_cnn.MLConv1D(*cnn_args, **cnn_kwargs) for b in self.band_names})
		print('ml_cnn:', self.ml_cnn)

		### PARALLEL PATCH
		linear_kwargs = {
			'activation':'linear',
		}
		self.z_projection = Linear(self.tcnn_embd_dims*len(self.band_names), self.tcnn_embd_dims, **linear_kwargs)
		print('z_projection:', self.z_projection)

	def get_output_dims(self):
		return self.tcnn_embd_dims#*len(self.band_names)
	
	def get_embd_dims_list(self):
		return {b:self.ml_cnn[b].get_embd_dims_list() for b in self.band_names}

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

			p_z = self.ml_cnn[b](p_z.permute(0,2,1)).permute(0,2,1)
			tdict['model'].update({f'z.{b}.last':p_z})

			### representative element
			last_z_dic[b] = seq_utils.seq_max_pooling(p_z, p_onehot)

		last_z = torch.cat([last_z_dic[b] for b in self.band_names], dim=-1)
		last_z = self.z_projection(last_z)
		tdict['model'].update({
			#'z':z, # not used
			'z.last':last_z,
		})
		return tdict

class TCNNEncoderS(nn.Module):
	def __init__(self,
		**kwargs):
		super().__init__()

		### ATTRIBUTES
		for name, val in kwargs.items():
			setattr(self, name, val)

		### PRE-INPUT
		linear_kwargs = {
			'activation':'linear',
		}
		extra_dims = len(self.band_names)
		self.x_projection = Linear(self.input_dims+extra_dims, self.tcnn_embd_dims, **linear_kwargs)
		print('x_projection:', self.x_projection)

		### TE
		self.te_film = FILM(self.te_features, self.tcnn_embd_dims)
		print('te_film:', self.te_film)

		### CNN STACK
		cnn_args = [self.tcnn_embd_dims, [self.curvelength_max], self.tcnn_embd_dims, [self.tcnn_embd_dims]*(self.tcnn_layers-1)] # input_dims:int, input_space:list, output_dims:int, embd_dims_list
		cnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'cnn_kwargs':{
				'kernel_size':5,
				'stride':1,
				'dilation':1,
			},
			'pool_kwargs':{
				'kernel_size':1,
				'stride':1,
				'dilation':1,
			},
			'padding_mode':'causal',
		}
		self.ml_cnn = ft_cnn.MLConv1D(*cnn_args, **cnn_kwargs)
		print('ml_cnn:', self.ml_cnn)

	def get_output_dims(self):
		return self.tcnn_embd_dims
	
	def get_embd_dims_list(self):
		return self.ml_cnn.get_embd_dims_list()

	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		model_input = tdict['input']
		x = model_input['x']
		onehot = model_input['onehot']

		z = self.x_projection(torch.cat([x, onehot.float()], dim=-1))
		z = self.te_film(z, model_input['te']) if self.te_features>0  else z

		s_onehot = onehot.sum(dim=-1).bool()
		z = self.ml_cnn(z.permute(0,2,1)).permute(0,2,1)

		### representative element
		last_z = seq_utils.seq_max_pooling(z, s_onehot) # get last value of sequence according to onehot

		tdict['model'].update({
			#'z':z, # not used
			'z.last':last_z,
		})
		return tdict