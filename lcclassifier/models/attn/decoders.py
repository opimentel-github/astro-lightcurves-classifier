from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
import torch.nn.functional as F 
import fuzzytorch.models.attn.basics as ft_attn
from fuzzytorch.models.basics import MLP, Linear
import fuzzytorch.models.seq_utils as seq_utils

###################################################################################################################################################
NUM_HEADS = C_.NUM_HEADS
class TimeSelfAttnDecoderP(nn.Module):
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
		len_bands = len(self.band_names)
		extra_dims = 0
		band_embedding_dims = self.attn_embd_dims//len_bands
		self.x_projection = nn.ModuleDict({b:Linear(self.input_dims+extra_dims, band_embedding_dims, **linear_kwargs) for b in self.band_names})
		print('x_projection:',self.x_projection)

		### ATTN
		attn_kwargs = {
			'num_heads':NUM_HEADS,
			'scale_mode':self.scale_mode,
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'activation':'relu',
			'last_activation':'relu',
			}
		self.ml_attn = nn.ModuleDict({b:ft_attn.MLTimeSelfAttn(band_embedding_dims, band_embedding_dims, [band_embedding_dims]*(self.attn_layers-1), self.te_features, self.max_period, **attn_kwargs) for b in self.band_names})
		print('ml_attn:', self.ml_attn)

		### DEC MLP
		mlp_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'activation':'relu',
			}
		self.dz_projection = nn.ModuleDict({b:MLP(band_embedding_dims, 1, [band_embedding_dims]*C_.DECODER_MLP_LAYERS, **mlp_kwargs) for b in self.band_names})
		print('dz_projection:', self.dz_projection)

	def get_output_dims(self):
		return self.dz_projection.get_output_dims()

	def get_embd_dims_list(self):
		return {b:self.ml_attn[b].get_embd_dims_list() for b in self.band_names}
		
	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		for kb,b in enumerate(self.band_names):
			p_onehot = tdict['input'][f'onehot.{b}'][...,0] # (b,t)
			p_rtime = tdict['input'][f'rtime.{b}'][...,0] # (b,t)
			#p_dtime = tdict['input'][f'dtime.{b}'][...,0] # (b,t)
			#p_x = tdict['input'][f'x.{b}'] # (b,t,f)
			#p_rerror = tdict['target'][f'rerror.{b}'] # (b,t,1)
			#p_rx = tdict['target'][f'rec_x.{b}'] # (b,t,1)

			p_decz = tdict['model']['encz_last'][:,None,:].repeat(1, p_onehot.shape[1], 1) # decoder z
			p_decz = self.x_projection[b](p_decz)
			p_decz, p_scores = self.ml_attn[b](p_decz, p_onehot, p_rtime)
			p_decx = self.dz_projection[b](p_decz)
			tdict['model'].update({
				f'decx.{b}':p_decx,
				})

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
		len_bands = len(self.band_names)
		extra_dims = len_bands+0
		self.x_projection = Linear(self.input_dims+extra_dims, self.attn_embd_dims, **linear_kwargs)
		print('x_projection:', self.x_projection)

		### ATTN
		attn_kwargs = {
			'num_heads':NUM_HEADS,
			'scale_mode':self.scale_mode,
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'activation':'relu',
			'last_activation':'relu',
			}
		self.ml_attn = ft_attn.MLTimeSelfAttn(self.attn_embd_dims, self.attn_embd_dims, [self.attn_embd_dims]*(self.attn_layers-1), self.te_features, self.max_period, **attn_kwargs)
		print('ml_attn:', self.ml_attn)

		### DEC MLP
		mlp_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'activation':'relu',
			}
		self.dz_projection = MLP(self.attn_embd_dims, 1, [self.attn_embd_dims]*C_.DECODER_MLP_LAYERS, **mlp_kwargs)
		print('dz_projection:', self.dz_projection)

	def get_output_dims(self):
		return self.dz_projection.get_output_dims()

	def get_embd_dims_list(self):
		return self.ml_attn.get_embd_dims_list()
		
	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		s_onehot = tdict['input']['s_onehot'] # (b,t,d)
		onehot = tdict['input']['onehot.*'][...,0] # (b,t)
		rtime = tdict['input']['rtime.*'][...,0] # (b,t)
		#dtime = tdict['input'][f'dtime.*'][...,0] # (b,t)
		#x = tdict['input'][f'x.*'] # (b,t,f)
		#rerror = tdict['target'][f'rerror.*'] # (b,t,1)
		#rx = tdict['target'][f'rec_x.*'] # (b,t,1)

		decz = tdict['model']['encz_last'][:,None,:].repeat(1,onehot.shape[1],1) # dz: decoder z
		decz = torch.cat([decz, s_onehot.float()], dim=-1) # (b,t,f+d)
		decz = self.x_projection(decz)
		decz, scores = self.ml_attn(decz, onehot, rtime, **kwargs)
		decx = self.dz_projection(decz)
		for kb,b in enumerate(self.band_names):
			p_decx = seq_utils.serial_to_parallel(decx, s_onehot[...,kb])
			tdict['model'].update({
				f'decx.{b}':p_decx,
				})

		return tdict