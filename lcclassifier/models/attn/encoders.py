from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F 
import fuzzytorch.models.attn.basics as ft_attn
from fuzzytorch.models.basics import MLP, Linear
from fuzzytorch.models.others import FILM
import fuzzytorch.models.seq_utils as seq_utils

###################################################################################################################################################

class TimeSelfAttnEncoderP(nn.Module):
	def __init__(self,
		**kwargs):
		super().__init__()
		### ATTRIBUTES
		self.add_extra_return = False
		for name, val in kwargs.items():
			setattr(self, name, val)
		assert self.te_features>0, 'attn needs to work with temporal encoding'
		self.reset()

	def reset(self):
		### PRE-INPUT
		linear_kwargs = {
			'activation':'linear',
		}
		extra_dims = 0
		self.x_projection = nn.ModuleDict({b:Linear(self.input_dims+extra_dims, self.attn_embd_dims, **linear_kwargs) for b in self.band_names})
		print('x_projection:', self.x_projection)

		### ATTN
		attn_kwargs = {
			'num_heads':4,
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
		}
		self.ml_attn = nn.ModuleDict({b:ft_attn.MLTimeSelfAttn(self.attn_embd_dims, self.attn_embd_dims, [self.attn_embd_dims]*(self.attn_layers-1), self.te_features, self.max_period, **attn_kwargs) for b in self.band_names})
		print('ml_attn:', self.ml_attn)

		### POST-PROJECTION
		linear_kwargs = {
			'activation':'linear',
		}
		self.z_projection = Linear(self.attn_embd_dims*len(self.band_names), self.attn_embd_dims, **linear_kwargs)
		print('z_projection:', self.z_projection)

	def get_info(self):
		d = {}
		for kb,b in enumerate(self.band_names):
			d[f'ml_attn.{b}'] = self.ml_attn[b].get_info()
		return d

	def get_output_dims(self):
		return self.attn_embd_dims#*len(self.band_names)
	
	def get_embd_dims_list(self):
		return {b:self.ml_attn[b].get_embd_dims_list() for b in self.band_names}

	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		model_input = tdict['input']
		x = model_input['x']
		onehot = model_input['onehot']

		z_bdict = {}
		attn_scores = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = seq_utils.serial_to_parallel(onehot, onehot[...,kb])[...,kb] # (b,t)
			p_x = seq_utils.serial_to_parallel(x, onehot[...,kb])
			p_z = self.x_projection[b](p_x)
			p_time = seq_utils.serial_to_parallel(model_input['time'], onehot[...,kb])
			p_zs, p_scores = self.ml_attn[b](p_z, p_onehot, p_time[...,0], return_only_actual_scores=True)

			### representative element
			for layer in range(0, self.attn_layers):
				z_bdict[f'z-{layer}.{b}'] = seq_utils.seq_last_element(p_zs[layer], p_onehot) # last element
				attn_scores[f'z-{layer}.{b}'] = p_scores[layer]
		
		### BUILD OUT
		z_last = torch.cat([z_bdict[f'z-{self.attn_layers-1}.{b}'] for b in self.band_names], dim=-1)
		tdict['model']['z_last'] = self.z_projection(z_last)
		for layer in range(0, self.attn_layers):
			tdict['model'][f'z-{layer}'] = torch.max(torch.cat([z_bdict[f'z-{layer}.{b}'][...,None] for b in self.band_names], dim=-1), dim=-1)[0]

		if self.add_extra_return:
			tdict['model'].update({
				'attn_scores':attn_scores,
				})
		return tdict

###################################################################################################################################################

class TimeSelfAttnEncoderS(nn.Module):
	def __init__(self,
		**kwargs):
		super().__init__()
		### ATTRIBUTES
		self.add_extra_return = False
		for name, val in kwargs.items():
			setattr(self, name, val)
		self.reset()

	def reset(self):
		### PRE-INPUT
		linear_kwargs = {
			'activation':'linear',
		}
		extra_dims = len(self.band_names)
		self.x_projection = Linear(self.input_dims+extra_dims, self.attn_embd_dims, **linear_kwargs)
		print('x_projection:', self.x_projection)

		### TE
		assert self.te_features>0
		#self.te_film = FILM(self.te_features, self.attn_embd_dims)
		#print('te_film:', self.te_film)

		### CNN STACK
		cnn_args = [self.attn_embd_dims, [None], self.attn_embd_dims, [self.attn_embd_dims]*(self.attn_layers-1)] # input_dims:int, input_space:list, output_dims:int, embd_dims_list
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
			'num_heads':4,
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
		}
		self.ml_attn = ft_attn.MLTimeSelfAttn(self.attn_embd_dims, self.attn_embd_dims, [self.attn_embd_dims]*(self.attn_layers-1), self.te_features, self.max_period, **attn_kwargs)
		print('ml_attn:', self.ml_attn)
		
		### POST-PROJECTION
		linear_kwargs = {
			'activation':'linear',
		}
		self.z_projection = Linear(self.attn_embd_dims, self.attn_embd_dims, **linear_kwargs)
		print('z_projection:', self.z_projection)

	def get_info(self):
		d = {
			'ml_attn':self.ml_attn.get_info(),
			}
		return d

	def get_output_dims(self):
		return self.attn_embd_dims
	
	def get_embd_dims_list(self):
		return self.ml_attn.get_embd_dims_list()

	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		model_input = tdict['input']
		x = model_input['x']
		onehot = model_input['onehot']
		s_onehot = onehot.sum(dim=-1).bool()
		time = model_input['time']

		z = self.x_projection(torch.cat([x, onehot.float()], dim=-1))
		zs, scores = self.ml_attn(z, s_onehot, time[...,0], return_only_actual_scores=True)

		z_bdict = {}
		attn_scores = {}
		### representative element
		for layer in range(0, self.attn_layers):
			z_bdict[f'z-{layer}'] = seq_utils.seq_last_element(zs[layer], s_onehot) # last element
			attn_scores[f'z-{layer}'] = scores[layer]

		### BUILD OUT
		tdict['model']['z_last'] = self.z_projection(z_bdict[f'z-{self.attn_layers-1}'])
		for layer in range(0, self.attn_layers):
			tdict['model'][f'z-{layer}'] = z_bdict[f'z-{layer}']
		if self.add_extra_return:
			tdict['model'].update({
				'attn_scores':attn_scores,
				})
		return tdict