from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F 
import fuzzytorch.models.cnn.basics as ft_cnn
import fuzzytorch.models.attn.basics as ft_attn
from fuzzytorch.models.basics import MLP, Linear
from fuzzytorch.models.others import FILM
import fuzzytorch.models.seq_utils as seq_utils

###################################################################################################################################################

class AttnTCNNEncoderP(nn.Module):
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
		extra_dims = 0
		self.x_projection = nn.ModuleDict({b:Linear(self.input_dims+extra_dims, self.tcnn_embd_dims, **linear_kwargs) for b in self.band_names})
		print('x_projection:', self.x_projection)

		### TE
		assert self.te_features>0
		#self.te_film = FILM(self.te_features, self.tcnn_embd_dims)
		#print('te_film:', self.te_film)

		### CNN STACK
		cnn_args = [self.tcnn_embd_dims, [None], self.tcnn_embd_dims, [self.tcnn_embd_dims]*(self.tcnn_layers-1)] # input_dims:int, input_space:list, output_dims:int, embd_dims_list
		cnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'cnn_kwargs':{
				'kernel_size':5,
				'stride':1,
				'dilation':1,
			},
			'pool_kwargs':{
				'kernel_size':2,
				'stride':1,
				'dilation':1,
			},
			'padding_mode':'causal',
		}
		#self.ml_cnn = nn.ModuleDict({b:ft_cnn.MLConv1D(*cnn_args, **cnn_kwargs) for b in self.band_names})
		#print('ml_cnn:', self.ml_cnn)

		### ATTN
		attn_kwargs = {
			'num_heads':2,
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
		}
		self.ml_attn = nn.ModuleDict({b:ft_attn.MLTimeSelfAttn(self.tcnn_embd_dims, self.tcnn_embd_dims, [self.tcnn_embd_dims]*(self.tcnn_layers-1), self.te_features, self.max_te_period, **attn_kwargs) for b in self.band_names})
		self.return_scores = False
		print('ml_attn:', self.ml_attn)

		#self.attn_te_film = FILM(self.te_features, self.tcnn_embd_dims)
		#print('attn_te_film:', self.attn_te_film)

		### PARALLEL PATCH
		linear_kwargs = {
			'activation':'linear',
		}
		self.z_projection = Linear(self.tcnn_embd_dims*len(self.band_names), self.tcnn_embd_dims, **linear_kwargs)
		print('z_projection:', self.z_projection)

	def get_output_dims(self):
		return self.tcnn_embd_dims#*len(self.band_names)
	
	def get_embd_dims_list(self):
		return {b:self.ml_attn[b].get_embd_dims_list() for b in self.band_names}

	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		model_input = tdict['input']
		x = model_input['x']
		onehot = model_input['onehot']

		last_z_dic = {}
		layer_scores = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = seq_utils.serial_to_parallel(onehot, onehot[...,kb])[...,kb] # (b,t)
			p_x = seq_utils.serial_to_parallel(x, onehot[...,kb])
			p_z = self.x_projection[b](p_x)
			#p_te = seq_utils.serial_to_parallel(model_input['te'], onehot[...,kb])
			p_time = seq_utils.serial_to_parallel(model_input['time'], onehot[...,kb])

			#p_z = self.te_film(p_z, p_te) if self.te_features>0 else p_z
			#p_z = self.ml_cnn[b](p_z.permute(0,2,1)).permute(0,2,1)

			#p_z = self.attn_te_film(p_z, p_te) if self.te_features>0 else p_z
			p_z, p_layer_scores = self.ml_attn[b](p_z, p_onehot, p_time[...,0])

			### representative element
			last_z_dic[b] = seq_utils.seq_last_element(p_z, p_onehot) # last element
			#last_z_dic[b] = seq_utils.seq_max_pooling(p_z, p_onehot) # max pooling
			#last_z_dic[b] = seq_utils.seq_avg_pooling(p_z, p_onehot) # avg pooling
			#tdict['model'].update({f'z.{b}.last':p_z})
			layer_scores[b] = p_layer_scores
		
		last_z = torch.cat([last_z_dic[b] for b in self.band_names], dim=-1)
		last_z = self.z_projection(last_z)
		tdict['model'].update({
			#'z':z, # not used
			'z.last':last_z,
		})
		if self.return_scores:
			tdict['model'].update({'layer_scores':layer_scores})
		return tdict

###################################################################################################################################################

class AttnTCNNEncoderS(nn.Module):
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
		assert self.te_features>0
		#self.te_film = FILM(self.te_features, self.tcnn_embd_dims)
		#print('te_film:', self.te_film)

		### CNN STACK
		cnn_args = [self.tcnn_embd_dims, [None], self.tcnn_embd_dims, [self.tcnn_embd_dims]*(self.tcnn_layers-1)] # input_dims:int, input_space:list, output_dims:int, embd_dims_list
		cnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'cnn_kwargs':{
				'kernel_size':5,
				'stride':1,
				'dilation':1,
			},
			'pool_kwargs':{
				'kernel_size':2,
				'stride':1,
				'dilation':1,
			},
			'padding_mode':'causal',
		}
		#self.ml_cnn = ft_cnn.MLConv1D(*cnn_args, **cnn_kwargs)
		#print('ml_cnn:', self.ml_cnn)

		### ATTN
		attn_kwargs = {
			'num_heads':2,
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
		}
		self.ml_attn = ft_attn.MLTimeSelfAttn(self.tcnn_embd_dims, self.tcnn_embd_dims, [self.tcnn_embd_dims]*(self.tcnn_layers-1), self.te_features, self.max_te_period, **attn_kwargs)
		self.return_scores = False
		print('ml_attn:', self.ml_attn)

		#self.attn_te_film = FILM(self.te_features, self.tcnn_embd_dims)
		#print('attn_te_film:', self.attn_te_film)

	def get_output_dims(self):
		return self.tcnn_embd_dims
	
	def get_embd_dims_list(self):
		return self.ml_attn.get_embd_dims_list()

	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		model_input = tdict['input']
		x = model_input['x']
		onehot = model_input['onehot']
		s_onehot = onehot.sum(dim=-1).bool()
		z = self.x_projection(torch.cat([x, onehot.float()], dim=-1))

		#z = self.te_film(z, model_input['te']) if self.te_features>0  else z
		#z = self.ml_cnn(z.permute(0,2,1)).permute(0,2,1)

		#z = self.attn_te_film(z, model_input['te']) if self.te_features>0 else z
		z, layer_scores = self.ml_attn(z, s_onehot, model_input['time'][...,0])

		### representative element
		last_z = seq_utils.seq_last_element(z, s_onehot) # last element
		#last_z = seq_utils.seq_max_pooling(z, s_onehot) # max pooling
		#last_z = seq_utils.seq_avg_pooling(z, s_onehot) # avg pooling
		tdict['model'].update({
			#'z':z, # not used
			'z.last':last_z,
		})
		if self.return_scores:
			tdict['model'].update({'layer_scores':layer_scores})
		return tdict