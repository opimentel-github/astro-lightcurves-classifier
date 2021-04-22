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

class TimeSeltAttnDecoderP(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		### ATTRIBUTES
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
		print('x_projection:',self.x_projection)

		### ATTN
		attn_kwargs = {
			'num_heads':4,
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
		}
		self.ml_attn = nn.ModuleDict({b:ft_attn.MLTimeSelfAttn(self.attn_embd_dims, self.attn_embd_dims, [self.attn_embd_dims]*(self.attn_layers-1), self.te_features, self.max_period, **attn_kwargs) for b in self.band_names})
		print('ml_attn:', self.ml_attn)

		### DEC MLP
		mlp_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'activation':'linear',
		}
		self.dz_projection = nn.ModuleDict({b:MLP(self.attn_embd_dims, 1, [self.attn_embd_dims]*1, **mlp_kwargs) for b in self.band_names})
		print('dz_projection:', self.dz_projection)

	def get_output_dims(self):
		return self.dz_projection.get_output_dims()

	def get_embd_dims_list(self):
		return {b:self.ml_attn[b].get_embd_dims_list() for b in self.band_names}
		
	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		model_input = tdict['input']
		x = model_input['x']
		onehot = model_input['onehot']

		extra_info = {}
		b,t,_ = onehot.size()
		dz = tdict['model']['z_last'][:,None,:].repeat(1, t, 1) # dz: decoder z
		for kb,b in enumerate(self.band_names):
			p_onehot = seq_utils.serial_to_parallel(onehot, onehot[...,kb])[...,kb] # (b,t)
			p_dz = seq_utils.serial_to_parallel(dz, onehot[...,kb])
			p_time = seq_utils.serial_to_parallel(model_input['time'], onehot[...,kb])

			p_rx = self.x_projection[b](p_dz)
			p_rx, _ = self.ml_attn[b](p_rx, p_onehot, p_time[...,0])
			p_rx = p_rx[-1]
			p_rx = self.dz_projection[b](p_rx)
			tdict['model'].update({f'rec_x.{b}':p_rx})

		return tdict

###################################################################################################################################################

class TimeSelfAttnDecoderS(nn.Module):
	def __init__(self,
		**kwargs):
		super().__init__()
		### ATTRIBUTES
		for name, val in kwargs.items():
			setattr(self, name, val)
		assert self.te_features>0, 'attn needs to work with temporal encoding'
		self.reset()

	def reset(self):
		### PRE-INPUT
		linear_kwargs = {
			'activation':'linear',
		}
		extra_dims = len(self.band_names)+0
		self.x_projection = Linear(self.input_dims+extra_dims, self.attn_embd_dims, **linear_kwargs)
		print('x_projection:', self.x_projection)


		### ATTN
		attn_kwargs = {
			'num_heads':4,
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
		}
		self.ml_attn = ft_attn.MLTimeSelfAttn(self.attn_embd_dims, self.attn_embd_dims, [self.attn_embd_dims]*(self.attn_layers-1), self.te_features, self.max_period, **attn_kwargs)
		print('ml_attn:', self.ml_attn)

		### DEC MLP
		mlp_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'activation':'linear',
		}
		self.dz_projection = MLP(self.attn_embd_dims, 1, [self.attn_embd_dims]*1, **mlp_kwargs)
		print('dz_projection:', self.dz_projection)

	def get_output_dims(self):
		return self.dz_projection.get_output_dims()

	def get_embd_dims_list(self):
		return self.ml_attn.get_embd_dims_list()
		
	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		model_input = tdict['input']
		x = model_input['x']
		onehot = model_input['onehot']

		b,t,_ = onehot.size()
		dz = tdict['model']['z_last'][:,None,:].repeat(1,t,1) # dz: decoder z
		dz = torch.cat([dz, onehot.float()], dim=-1) # cat bands
		time = model_input['time']
		s_onehot = onehot.sum(dim=-1).bool()
		
		rx = self.x_projection(dz)
		rx, _ = self.ml_attn(rx, s_onehot, time[...,0], **kwargs)
		rx = rx[-1]
		rx = self.dz_projection(rx)
			
		for kb,b in enumerate(self.band_names):
			p_rx = seq_utils.serial_to_parallel(rx, onehot[...,kb])
			tdict['model'].update({f'rec_x.{b}':p_rx})

		return tdict