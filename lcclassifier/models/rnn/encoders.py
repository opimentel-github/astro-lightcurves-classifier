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
		len_bands = len(self.band_names)
		extra_dims = 0
		band_embedding_dims = self.rnn_embd_dims//len_bands
		self.x_projection = nn.ModuleDict({b:Linear(self.input_dims+extra_dims, band_embedding_dims, **linear_kwargs) for b in self.band_names})
		print('x_projection:', self.x_projection)

		### RNN STACK
		rnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'bidirectional':self.bidirectional,
			'uses_batchnorm':self.uses_batchnorm,
			}
		self.ml_rnn = nn.ModuleDict({b:getattr(ft_rnn, f'ML{self.rnn_cell_name}')(band_embedding_dims, band_embedding_dims, [band_embedding_dims]*(self.rnn_layers-1), **rnn_kwargs) for b in self.band_names})
		print('ml_rnn:', self.ml_rnn)

		### POST-PROJECTION
		linear_kwargs = {
			'in_dropout':self.dropout['p'],
			'activation':'linear',
			}
		self.mb_projection = Linear(band_embedding_dims*len_bands, band_embedding_dims*len_bands, **linear_kwargs)
		print('mb_projection:', self.mb_projection)

		### XENTROPY REG
		linear_kwargs = {
			'in_dropout':self.dropout['p'],
			'activation':'linear',
			}
		self.xentropy_projection = Linear(band_embedding_dims*len_bands, self.output_dims, **linear_kwargs)
		print('xentropy_projection:', self.xentropy_projection)

	def get_output_dims(self):
		return self.rnn_embd_dims
	
	def get_embd_dims_list(self):
		return {b:self.ml_rnn[b].get_embd_dims_list() for b in self.band_names}

	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		encz_bdict = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = tdict['input'][f'onehot.{b}'][...,0] # (b,t)
			#p_rtime = tdict['input'][f'rtime.{b}'][...,0] # (b,t)
			#p_dtime = tdict['input'][f'dtime.{b}'][...,0] # (b,t)
			p_x = tdict['input'][f'x.{b}'] # (b,t,f)
			#p_rerror = tdict['target'][f'rerror.{b}'] # (b,t,1)
			#p_rx = tdict['target'][f'rec_x.{b}'] # (b,t,1)

			p_encz = self.x_projection[b](p_x)
			p_encz, _ = self.ml_rnn[b](p_encz, p_onehot, **kwargs) # out, (ht, ct)
			### representative element
			encz_bdict[f'encz.{b}'] = seq_utils.seq_last_element(p_encz, p_onehot) # last element

		### BUILD OUT
		encz_last = self.mb_projection(torch.cat([encz_bdict[f'encz.{b}'] for b in self.band_names], dim=-1))
		tdict['model']['encz_last'] = encz_last
		tdict['model']['y_last_pt'] = self.xentropy_projection(encz_last)
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
		len_bands = len(self.band_names)
		extra_dims = len_bands
		self.x_projection = Linear(self.input_dims+extra_dims, self.rnn_embd_dims, **linear_kwargs)
		print('x_projection:', self.x_projection)

		### RNN STACK
		rnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'bidirectional':self.bidirectional,
			'uses_batchnorm':self.uses_batchnorm,
			}
		self.ml_rnn = getattr(ft_rnn, f'ML{self.rnn_cell_name}')(self.rnn_embd_dims, self.rnn_embd_dims, [self.rnn_embd_dims]*(self.rnn_layers-1), **rnn_kwargs)
		print('ml_rnn:', self.ml_rnn)

		### XENTROPY REG
		linear_kwargs = {
			'in_dropout':self.dropout['p'],
			'activation':'linear',
			}
		self.xentropy_projection = Linear(self.rnn_embd_dims, self.output_dims, **linear_kwargs)
		print('xentropy_projection:', self.xentropy_projection)

	def get_output_dims(self):
		return self.rnn_embd_dims
	
	def get_embd_dims_list(self):
		return self.ml_rnn.get_embd_dims_list()

	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		encz_bdict = {}

		s_onehot = tdict['input']['s_onehot'] # (b,t,d)
		onehot = tdict['input']['onehot.*'][...,0] # (b,t)
		#rtime = tdict['input']['rtime.*'][...,0] # (b,t)
		#dtime = tdict['input'][f'dtime.*'][...,0] # (b,t)
		x = tdict['input'][f'x.*'] # (b,t,f)
		#rerror = tdict['target'][f'rerror.*'] # (b,t,1)
		#rx = tdict['target'][f'rec_x.*'] # (b,t,1)

		encz = self.x_projection(torch.cat([x, s_onehot.float()], dim=-1)) # (b,t,f+d)
		encz, _ = self.ml_rnn(encz, onehot, **kwargs) # out, (ht, ct)
		### representative element
		encz_bdict[f'encz'] = seq_utils.seq_last_element(encz, onehot) # last element

		### BUILD OUT
		encz_last = encz_bdict[f'encz']
		tdict['model']['encz_last'] = encz_last
		tdict['model']['y_last_pt'] = self.xentropy_projection(encz_last)
		return tdict