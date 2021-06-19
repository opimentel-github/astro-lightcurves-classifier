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
		len_bands = len(self.band_names)
		extra_dims = 0
		band_embedding_dims = self.embd_dims//len_bands
		self.x_projection = nn.ModuleDict({b:Linear(self.input_dims+extra_dims, band_embedding_dims,
			activation='linear',
			bias=False,
			) for b in self.band_names})
		print('x_projection:', self.x_projection)

		### ATTN
		self.ml_attn = nn.ModuleDict({b:ft_attn.MLTimeSelfAttn(band_embedding_dims, band_embedding_dims, [band_embedding_dims]*(self.layers-1), self.te_features, self.max_period,
			num_heads=self.heads,
			kernel_size=self.kernel_size,
			time_noise_window=self.time_noise_window,
			fourier_dims=self.fourier_dims,
			in_dropout=self.dropout['p'],
			residual_dropout=self.dropout['r'],
			dropout=self.dropout['p'],
			activation='relu',
			last_activation='relu',
			) for b in self.band_names})
		print('ml_attn:', self.ml_attn)

		self.seft = nn.ModuleDict({b:seq_utils.LinearSEFT(band_embedding_dims,
			in_dropout=self.dropout['p'],
			) for b in self.band_names})
		print('seft:', self.seft)

		### POST-PROJECTION
		self.mb_projection = Linear(band_embedding_dims*len_bands, band_embedding_dims*len_bands,
			in_dropout=self.dropout['p'],
			activation='linear',
			bias=False,
			)
		print('mb_projection:', self.mb_projection)

		### XENTROPY REG
		self.xentropy_projection = Linear(band_embedding_dims*len_bands, self.output_dims,
			in_dropout=self.dropout['p'],
			activation='linear',
			)
		print('xentropy_projection:', self.xentropy_projection)

	def get_info(self):
		d = {}
		for kb,b in enumerate(self.band_names):
			d[f'ml_attn.{b}'] = self.ml_attn[b].get_info()
		return d

	def init_fine_tuning(self):
		for kb,b in enumerate(self.band_names):
			assert hasattr(self.ml_attn[b].te_film, 'time_noise_window')
			print('self.ml_attn[b].te_film.temporal_encoder.time_noise_window = 0')
			self.ml_attn[b].te_film.temporal_encoder.time_noise_window = 0

	def get_output_dims(self):
		return self.embd_dims
	
	def get_embd_dims_list(self):
		return {b:self.ml_attn[b].get_embd_dims_list() for b in self.band_names}

	def forward(self, tdict:dict, **kwargs):
		encz_bdict = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = tdict[f'input/onehot.{b}'][...,0] # (b,t)
			p_rtime = tdict[f'input/rtime.{b}'][...,0] # (b,t)
			#p_dtime = tdict[f'input/dtime.{b}'][...,0] # (b,t)
			p_x = tdict[f'input/x.{b}'] # (b,t,f)
			#p_rerror = tdict[f'target/rerror.{b}'] # (b,t,1)
			#p_rx = tdict[f'target/rec_x.{b}'] # (b,t,1)

			p_encz = self.x_projection[b](p_x)
			p_encz, p_scores = self.ml_attn[b](p_encz, p_onehot, p_rtime, return_only_actual_scores=True)
			encz_bdict[f'encz.{b}'] = self.seft[b](p_encz, p_onehot) # (b,t,f) > (b,f)
			
			if self.add_extra_return:
				tdict[f'model/attn_scores/encz.{b}'] = p_scores
		
		### return
		encz_last = self.mb_projection(torch.cat([encz_bdict[f'encz.{b}'] for b in self.band_names], dim=-1))
		tdict[f'model/encz_last'] = encz_last
		# tdict[f'model/y_last_pt'] = self.xentropy_projection(encz_last)
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
		assert self.te_features>0, 'attn needs to work with temporal encoding'
		self.reset()

	def reset(self):
		### PRE-INPUT
		len_bands = len(self.band_names)
		extra_dims = len_bands
		self.x_projection = Linear(self.input_dims+extra_dims, self.embd_dims,
			activation='linear',
			bias=False,
			)
		print('x_projection:', self.x_projection)
		
		### ATTN
		self.ml_attn = ft_attn.MLTimeSelfAttn(self.embd_dims, self.embd_dims, [self.embd_dims]*(self.layers-1), self.te_features, self.max_period,
			num_heads=self.heads*len_bands,
			kernel_size=self.kernel_size,
			time_noise_window=self.time_noise_window,
			fourier_dims=self.fourier_dims,
			in_dropout=self.dropout['p'],
			residual_dropout=self.dropout['r'],
			dropout=self.dropout['p'],
			activation='relu',
			last_activation='relu',
			)
		print('ml_attn:', self.ml_attn)

		self.seft = seq_utils.LinearSEFT(self.embd_dims,
			in_dropout=self.dropout['p'],
			)
		print('seft:', self.seft)

		### XENTROPY REG
		self.xentropy_projection = Linear(self.embd_dims, self.output_dims,
			in_dropout=self.dropout['p'],
			activation='linear',
			)
		print('xentropy_projection:', self.xentropy_projection)

	def get_info(self):
		d = {
			'ml_attn':self.ml_attn.get_info(),
			}
		return d

	def init_fine_tuning(self):
		assert hasattr(self.ml_attn.te_film, 'time_noise_window')
		print('self.ml_attn.te_film.temporal_encoder.time_noise_window = 0')
		self.ml_attn.te_film.temporal_encoder.time_noise_window = 0

	def get_output_dims(self):
		return self.embd_dims
	
	def get_embd_dims_list(self):
		return self.ml_attn.get_embd_dims_list()

	def forward(self, tdict:dict, **kwargs):
		encz_bdict = {}
		s_onehot = tdict[f'input/s_onehot'] # (b,t,d)
		onehot = tdict[f'input/onehot.*'][...,0] # (b,t)
		rtime = tdict[f'input/rtime.*'][...,0] # (b,t)
		#dtime = tdict[f'input/dtime.*'][...,0] # (b,t)
		x = tdict[f'input/x.*'] # (b,t,f)
		#rerror = tdict[f'target/rerror.*'] # (b,t,1)
		#rx = tdict[f'target/rec_x.*'] # (b,t,1)

		encz = self.x_projection(torch.cat([x, s_onehot.float()], dim=-1)) # (b,t,f+d)
		encz, scores = self.ml_attn(encz, onehot, rtime, return_only_actual_scores=True)
		encz_bdict[f'encz'] = self.seft(encz, onehot) # (b,t,f) > (b,f)

		if self.add_extra_return:
			tdict[f'model/attn_scores/encz'] = scores

		### return
		encz_last = encz_bdict[f'encz']
		tdict[f'model/encz_last'] = encz_last
		tdict[f'model/y_last_pt'] = self.xentropy_projection(encz_last)
		return tdict