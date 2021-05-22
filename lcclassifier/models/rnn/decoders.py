from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
import torch.nn.functional as F 
import fuzzytorch.models.rnn.basics as ft_rnn
from fuzzytorch.models.basics import MLP, Linear
import fuzzytorch.models.seq_utils as seq_utils

###################################################################################################################################################

class RNNDecoderP(nn.Module):
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

		len_bands = len(self.band_names)
		extra_dims = 1
		band_embedding_dims = self.rnn_embd_dims//len_bands
		self.x_projection = nn.ModuleDict({b:Linear(self.input_dims+extra_dims, band_embedding_dims, **linear_kwargs) for b in self.band_names})
		print('x_projection:',self.x_projection)

		### RNN
		rnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'bidirectional':self.bidirectional,
			}
		self.ml_rnn = nn.ModuleDict({b:getattr(ft_rnn, f'ML{self.rnn_cell_name}')(band_embedding_dims, band_embedding_dims, [band_embedding_dims]*(self.rnn_layers-1), **rnn_kwargs) for b in self.band_names})
		print('ml_rnn:', self.ml_rnn)

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
		return {b:self.ml_rnn[b].get_embd_dims_list() for b in self.band_names}
		
	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		for kb,b in enumerate(self.band_names):
			p_onehot = tdict['input'][f'onehot.{b}'][...,0] # (b,t)
			#p_time = tdict['input'][f'time.{b}'][...,0] # (b,t)
			p_dtime = tdict['input'][f'dtime.{b}'][...,0] # (b,t)
			#p_x = tdict['input'][f'x.{b}'] # (b,t,f)
			#p_error = tdict['target'][f'error.{b}'] # (b,t,1)
			#p_rx = tdict['target'][f'rec_x.{b}'] # (b,t,1)

			p_decz = tdict['model']['encz_last'][:,None,:].repeat(1, p_onehot.shape[1], 1) # decoder z
			p_decz = torch.cat([p_decz, p_dtime[...,None]], dim=-1) # cat dtime
			p_decz = self.x_projection[b](p_decz)
			p_decz, _ = self.ml_rnn[b](p_decz, p_onehot, **kwargs) # out, (ht, ct)
			p_decx = self.dz_projection[b](p_decz)
			tdict['model'].update({
				f'decx.{b}':p_decx,
				})

		return tdict

###################################################################################################################################################

class RNNDecoderS(nn.Module):
	def __init__(self,
		**kwargs):
		super().__init__()
		### ATTRIBUTES
		setattr(self, 'bidirectional', False)
		for name, val in kwargs.items():
			setattr(self, name, val)
		self.reset()
		
	def reset(self):
		### PRE-INPUT
		linear_kwargs = {
			'activation':'linear',
		}
		len_bands = len(self.band_names)
		extra_dims = len_bands+1
		self.x_projection = Linear(self.input_dims+extra_dims, self.rnn_embd_dims, **linear_kwargs)
		print('x_projection:', self.x_projection)

		### RNN
		rnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'bidirectional':self.bidirectional,
			}
		self.ml_rnn = getattr(ft_rnn, f'ML{self.rnn_cell_name}')(self.rnn_embd_dims, self.rnn_embd_dims, [self.rnn_embd_dims]*(self.rnn_layers-1), **rnn_kwargs)
		print('ml_rnn:', self.ml_rnn)

		### DEC MLP
		mlp_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'activation':'relu',
		}
		self.dz_projection = MLP(self.rnn_embd_dims, 1, [self.rnn_embd_dims]*C_.DECODER_MLP_LAYERS, **mlp_kwargs)
		print('dz_projection:', self.dz_projection)

	def get_output_dims(self):
		return self.dz_projection.get_output_dims()

	def get_embd_dims_list(self):
		return self.ml_rnn.get_embd_dims_list()
		
	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		s_onehot = tdict['input']['s_onehot'] # (b,t,d)
		onehot = tdict['input']['onehot.*'][...,0] # (b,t)
		#time = tdict['input']['time.*'][...,0] # (b,t)
		dtime = tdict['input'][f'dtime.*'][...,0] # (b,t)
		#x = tdict['input'][f'x.*'] # (b,t,f)
		#error = tdict['target'][f'error.*'] # (b,t,1)
		#rx = tdict['target'][f'rec_x.*'] # (b,t,1)

		decz = tdict['model']['encz_last'][:,None,:].repeat(1,onehot.shape[1],1) # dz: decoder z
		decz = torch.cat([decz, s_onehot.float(), dtime[...,None]], dim=-1) # cat bands & dtime # (b,t,f+d+1)
		
		decz = self.x_projection(decz)
		decz, _ = self.ml_rnn(decz, onehot, **kwargs) # out, (ht, ct)
		decx = self.dz_projection(decz)
		for kb,b in enumerate(self.band_names):
			p_decx = seq_utils.serial_to_parallel(decx, s_onehot[...,kb])
			tdict['model'].update({
				f'decx.{b}':p_decx,
				})

		return tdict