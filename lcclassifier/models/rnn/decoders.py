from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F 
import fuzzytorch.models.rnn.basics as ft_rnn
from fuzzytorch.models.basics import MLP, Linear
from fuzzytorch.models.others import FILM
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

		### PRE-INPUT
		linear_kwargs = {
			'activation':'linear',
		}
		extra_dims = 0 if self.te_features else 1
		self.x_projection = nn.ModuleDict({b:Linear(self.input_dims+extra_dims, self.rnn_embd_dims, **linear_kwargs) for b in self.band_names})
		print('x_projection:',self.x_projection)

		### TE
		self.te_film = FILM(self.te_features, self.rnn_embd_dims)
		print('te_film:', self.te_film)

		### RNN STACK
		rnn_args = [self.rnn_embd_dims, self.rnn_embd_dims, [self.rnn_embd_dims]*(self.rnn_layers-1), self.curvelength_max]
		rnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'bidirectional':self.bidirectional,
		}
		self.ml_rnn = nn.ModuleDict({b:getattr(ft_rnn, f'ML{self.rnn_cell_name}')(*rnn_args, **rnn_kwargs) for b in self.band_names})
		print('ml_rnn:', self.ml_rnn)

		mlp_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'activation':'linear',
		}
		self.dz_projection = nn.ModuleDict({b:MLP(self.rnn_embd_dims, 1, [self.rnn_embd_dims]*1, **mlp_kwargs) for b in self.band_names})
		print('dz_projection:', self.dz_projection)

	def get_output_dims(self):
		return self.dz_projection.get_output_dims()

	def get_embd_dims_list(self):
		return {b:self.ml_rnn[b].get_embd_dims_list() for b in self.band_names}
		
	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		model_input = tdict['input']
		x = model_input['x']
		onehot = model_input['onehot']

		extra_info = {}
		b,t,_ = onehot.size()
		dz = tdict['model']['z.last'][:,None,:].repeat(1, t, 1) # dz: decoder z

		for kb,b in enumerate(self.band_names):
			p_onehot = seq_utils.serial_to_parallel(onehot, onehot[...,kb])[...,kb] # (b,t)
			p_dz = seq_utils.serial_to_parallel(dz, onehot[...,kb])

			if self.te_features>0:
				p_rx = self.x_projection[b](p_dz)
				p_te = seq_utils.serial_to_parallel(model_input['te'], onehot[...,kb])
				p_rx = self.te_film(p_rx, p_te)
			else:
				p_dt = seq_utils.serial_to_parallel(model_input['dt'], onehot[...,kb])
				p_dz = torch.cat([p_dz, p_dt], dim=-1) # cat dt
				p_rx = self.x_projection[b](p_dz)

			p_rx, p_extra_info_rnn = self.ml_rnn[b](p_rx, p_onehot, **kwargs) # out, (ht, ct)
			p_rx = self.dz_projection[b](p_rx)
			tdict['model'].update({f'raw-x.{b}':p_rx})

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

		### PRE-INPUT
		linear_kwargs = {
			'activation':'linear',
		}
		extra_dims = len(self.band_names) if self.te_features else len(self.band_names)+1
		self.x_projection = Linear(self.input_dims+extra_dims, self.rnn_embd_dims, **linear_kwargs)
		print('x_projection:', self.x_projection)

		### TE
		self.te_film = FILM(self.te_features, self.rnn_embd_dims)
		print('te_film:', self.te_film)

		### RNN STACK
		rnn_args = [self.rnn_embd_dims, self.rnn_embd_dims, [self.rnn_embd_dims]*(self.rnn_layers-1), self.curvelength_max]
		rnn_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'bidirectional':self.bidirectional,
		}
		self.ml_rnn = getattr(ft_rnn, f'ML{self.rnn_cell_name}')(*rnn_args, **rnn_kwargs)
		print('ml_rnn:', self.ml_rnn)

		mlp_kwargs = {
			'in_dropout':self.dropout['p'],
			'dropout':self.dropout['p'],
			'activation':'linear',
		}
		self.dz_projection = MLP(self.rnn_embd_dims, 1, [self.rnn_embd_dims]*1, **mlp_kwargs)
		print('dz_projection:', self.dz_projection)

	def get_output_dims(self):
		return self.dz_projection.get_output_dims()

	def get_embd_dims_list(self):
		return self.ml_rnn.get_embd_dims_list()
		
	def forward(self, tdict:dict, **kwargs):
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		model_input = tdict['input']
		x = model_input['x']
		onehot = model_input['onehot']

		b,t,_ = onehot.size()
		dz = tdict['model']['z.last'][:,None,:].repeat(1,t,1) # dz: decoder z
		dz = torch.cat([dz, onehot.float()], dim=-1)

		if self.te_features>0:
			rx = self.x_projection(dz)
			rx = self.te_film(rx, model_input['te'])
		else:
			dz = torch.cat([dz, model_input['dt']], dim=-1) # cat dt
			rx = self.x_projection(dz)

		s_onehot = onehot.sum(dim=-1).bool()
		rx, extra_info_rnn = self.ml_rnn(rx, s_onehot, **kwargs) # out, (ht, ct)
		rx = self.dz_projection(rx)
			
		for kb,b in enumerate(self.band_names):
			p_rx = seq_utils.serial_to_parallel(rx, onehot[...,kb])
			tdict['model'].update({f'raw-x.{b}':p_rx})

		return tdict