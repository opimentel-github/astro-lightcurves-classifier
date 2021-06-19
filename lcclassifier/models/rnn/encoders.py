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
		len_bands = len(self.band_names)
		extra_dims = 1 # dtime
		band_embedding_dims = self.embd_dims//len_bands
		self.x_projection = nn.ModuleDict({b:Linear(self.input_dims+extra_dims, band_embedding_dims,
			activation='linear',
			bias=False,
			) for b in self.band_names})
		print('x_projection:', self.x_projection)

		### RNN STACK
		self.ml_rnn = nn.ModuleDict({b:getattr(ft_rnn, f'ML{self.rnn_cell_name}')(band_embedding_dims, band_embedding_dims, [band_embedding_dims]*(self.layers-1),
			in_dropout=self.dropout['p'],
			dropout=self.dropout['p'],
			bidirectional=self.bidirectional,
			uses_batchnorm=self.uses_batchnorm,
			) for b in self.band_names})
		print('ml_rnn:', self.ml_rnn)

		self.seft = nn.ModuleDict({b:seq_utils.LinearSEFT(band_embedding_dims,
			in_dropout=self.dropout['p'],
			) for b in self.band_names})
		print('seft:', self.seft)

		self.mb_projection = Linear(band_embedding_dims*len_bands, band_embedding_dims*len_bands,
			in_dropout=self.dropout['p'],
			activation='linear',
			bias=False,
			)
		print('mb_projection:', self.mb_projection)

	def get_info(self):
		pass

	def get_finetuning_parameters(self):
		return [self.seft, self.mb_projection]

	def init_fine_tuning(self):
		pass

	def get_output_dims(self):
		return self.embd_dims
	
	def get_embd_dims_list(self):
		return {b:self.ml_rnn[b].get_embd_dims_list() for b in self.band_names}

	def forward(self, tdict:dict, **kwargs):
		encz_bdict = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = tdict[f'input/onehot.{b}'][...,0] # (b,t)
			#p_rtime = tdict[f'input/rtime.{b}'][...,0] # (b,t)
			p_dtime = tdict[f'input/dtime.{b}'][...,0] # (b,t)
			p_x = tdict[f'input/x.{b}'] # (b,t,f)
			#p_rerror = tdict[f'target/rerror.{b}'] # (b,t,1)
			#p_rx = tdict[f'target/rec_x.{b}'] # (b,t,1)

			p_encz = self.x_projection[b](torch.cat([p_x, p_dtime[...,None]], dim=-1))
			p_encz, _ = self.ml_rnn[b](p_encz, p_onehot, **kwargs) # out, (ht, ct)
			encz_bdict[f'encz.{b}'] = self.seft[b](p_encz, p_onehot) # (b,t,f) > (b,f)

		### return
		encz_last = self.mb_projection(torch.cat([encz_bdict[f'encz.{b}'] for b in self.band_names], dim=-1))
		tdict[f'model/encz_last'] = encz_last
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
		self.reset()

	def reset(self):
		### PRE-INPUT
		len_bands = len(self.band_names)
		extra_dims = 1+len_bands # dtime + bands
		self.x_projection = Linear(self.input_dims+extra_dims, self.embd_dims,
			activation='linear',
			bias=False,
			)
		print('x_projection:', self.x_projection)

		### RNN STACK
		self.ml_rnn = getattr(ft_rnn, f'ML{self.rnn_cell_name}')(self.embd_dims, self.embd_dims, [self.embd_dims]*(self.layers-1),
			in_dropout=self.dropout['p'],
			dropout=self.dropout['p'],
			bidirectional=self.bidirectional,
			uses_batchnorm=self.uses_batchnorm,
			)
		print('ml_rnn:', self.ml_rnn)

		self.seft = seq_utils.LinearSEFT(self.embd_dims,
			in_dropout=self.dropout['p'],
			)
		print('seft:', self.seft)

	def get_info(self):
		pass

	def get_finetuning_parameters(self):
		return [self.seft]

	def init_finetuning(self):
		pass

	def get_output_dims(self):
		return self.embd_dims
	
	def get_embd_dims_list(self):
		return self.ml_rnn.get_embd_dims_list()

	def forward(self, tdict:dict, **kwargs):
		encz_bdict = {}
		s_onehot = tdict[f'input/s_onehot'] # (b,t,d)
		onehot = tdict[f'input/onehot.*'][...,0] # (b,t)
		#rtime = tdict[f'input/rtime.*'][...,0] # (b,t)
		dtime = tdict[f'input/dtime.*'][...,0] # (b,t)
		x = tdict[f'input/x.*'] # (b,t,f)
		#rerror = tdict[f'target/rerror.*'] # (b,t,1)
		#rx = tdict[f'target/rec_x.*'] # (b,t,1)

		encz = self.x_projection(torch.cat([x, dtime[...,None], s_onehot.float()], dim=-1)) # (b,t,f+d)
		encz, _ = self.ml_rnn(encz, onehot, **kwargs) # out, (ht, ct)
		encz_bdict[f'encz'] = self.seft(encz, onehot) # (b,t,f) > (b,f)

		### return
		encz_last = encz_bdict[f'encz']
		tdict[f'model/encz_last'] = encz_last
		return tdict