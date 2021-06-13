from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
import torch.nn.functional as F
import fuzzytorch.models.rnn.basics as ft_rnn
from fuzzytorch.models.basics import MLP, Linear
import fuzzytorch.models.seq_utils as seq_utils

DECODER_MLP_LAYERS = 1
DECODER_LAYERS = 1

###################################################################################################################################################

class LatentGRUDecoderP(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		### ATTRIBUTES
		for name, val in kwargs.items():
			setattr(self, name, val)
		self.reset()
	
	def reset(self):
		### RNN
		dec_input_dims = 1 # dt
		self.ml_rnn = nn.ModuleDict({b:getattr(ft_rnn, f'MLGRU')(dec_input_dims, self.embd_dims, [self.embd_dims]*(DECODER_LAYERS-1),
			in_dropout=0,
			dropout=self.dropout['p'],
			) for b in self.band_names})
		print('ml_rnn:', self.ml_rnn)

		### DEC MLP
		self.dz_projection = nn.ModuleDict({b:MLP(self.embd_dims, 1, [self.embd_dims]*DECODER_MLP_LAYERS,
			in_dropout=self.dropout['p'],
			dropout=self.dropout['p'],
			activation='relu',
			) for b in self.band_names})
		print('dz_projection:', self.dz_projection)

	def get_output_dims(self):
		return self.dz_projection.get_output_dims()

	def get_embd_dims_list(self):
		return {b:self.ml_rnn[b].get_embd_dims_list() for b in self.band_names}
		
	def forward(self, tdict:dict, **kwargs):
		for kb,b in enumerate(self.band_names):
			p_onehot = tdict[f'input/onehot.{b}'][...,0] # (b,t)
			#p_rtime = tdict[f'input/rtime.{b}'][...,0] # (b,t)
			p_dtime = tdict[f'input/dtime.{b}'][...,0] # (b,t)
			#p_x = tdict[f'input/x.{b}'] # (b,t,f)
			#p_rerror = tdict[f'target/rerror.{b}'] # (b,t,1)
			#p_rx = tdict[f'target/rec_x.{b}'] # (b,t,1)

			encz_last = tdict[f'model/encz_last'][None,:,:] # (b,f) > (1,b,f)
			p_decz = p_dtime[...,None] # (b,t) > (b,t,1)
			p_decz, _ = self.ml_rnn[b](p_decz, p_onehot, h0=encz_last) # out, (ht, ct)
			p_decx = self.dz_projection[b](p_decz)
			tdict[f'model/decx.{b}'] = p_decx

		return tdict

###################################################################################################################################################

'''class LatentGRUDecoderS(nn.Module):
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
		extra_dims = len_bands+1 # bands+dt
		embedding_dims = int(self.embd_dims*C_.DECODER_EMB_K)
		self.h0_projection = Linear(self.input_dims+extra_dims, embedding_dims, **linear_kwargs)
		print('h0_projection:', self.h0_projection)

		### RNN
		rnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'bidirectional':self.bidirectional,
			}
		self.ml_rnn = getattr(ft_rnn, f'ML{self.rnn_cell_name}')(embedding_dims, embedding_dims, [embedding_dims]*(self.rnn_layers-1), **rnn_kwargs)
		print('ml_rnn:', self.ml_rnn)

		### DEC MLP
		mlp_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'activation':'relu',
		}
		self.dz_projection = MLP(embedding_dims, 1, [embedding_dims]*C_.DECODER_MLP_LAYERS, **mlp_kwargs)
		print('dz_projection:', self.dz_projection)

	def get_output_dims(self):
		return self.dz_projection.get_output_dims()

	def get_embd_dims_list(self):
		return self.ml_rnn.get_embd_dims_list()
		
	def forward(self, tdict:dict, **kwargs):
		s_onehot = tdict[f'input/s_onehot'] # (b,t,d)
		onehot = tdict[f'input/onehot.*'][...,0] # (b,t)
		#rtime = tdict[f'input/rtime.*'][...,0] # (b,t)
		dtime = tdict[f'input/dtime.*'][...,0] # (b,t)
		#x = tdict[f'input/x.*'] # (b,t,f)
		#rerror = tdict[f'target/rerror.*'] # (b,t,1)
		#rx = tdict[f'target/rec_x.*'] # (b,t,1)

		decz = tdict[f'model/encz_last'][:,None,:].repeat(1,onehot.shape[1],1) # dz: decoder z
		decz = torch.cat([decz, s_onehot.float(), dtime[...,None]], dim=-1) # cat bands & dtime # (b,t,f+d+1)
		decz = self.h0_projection(decz)
		decz, _ = self.ml_rnn(decz, onehot, **kwargs) # out, (ht, ct)
		decx = self.dz_projection(decz)
		for kb,b in enumerate(self.band_names):
			p_decx = seq_utils.serial_to_parallel(decx, s_onehot[...,kb])
			tdict[f'model/decx.{b}'] = p_decx

		return tdict'''