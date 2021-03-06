from __future__ import print_function
from __future__ import division
from . import C_

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fuzzytorch.utils import get_model_name, print_tdict
from copy import copy, deepcopy
import torch.autograd.profiler as profiler
from .rnn import decoders as rnn_decoders
from .rnn import encoders as rnn_encoders
from .attn import encoders as attn_encoders

GLOBAL_DECODER_CLASS = rnn_decoders.LatentGRUDecoderP
# GLOBAL_DECODER_CLASS = rnn_decoders.LatentGRUDecoderS

###################################################################################################################################################

def get_enc_emb_str(mdl, band_names):
	dims = mdl.get_embd_dims_list()
	if isinstance(dims, dict):
		txts = ['-'.join([f'{b}{d}' for d in dims[b]]) for b in band_names]
		return '.'.join(txts)
	else:
		txts = [f'{d}' for d in dims]
		return '-'.join(txts)

###################################################################################################################################################

class ModelBaseline(nn.Module):
	def __init__(self, **raw_kwargs):
		super().__init__()

	def get_output_dims(self):
		return self.encoder.get_output_dims()

	def get_finetuning_parameters(self):
		encoder = self.autoencoder['encoder']
		# finetuning_parameters = [self]
		finetuning_parameters = self.classifier.get_finetuning_parameters()
		# finetuning_parameters = encoder.get_finetuning_parameters()+self.classifier.get_finetuning_parameters()
		return finetuning_parameters

	def init_finetuning(self):
		encoder = self.autoencoder['encoder']
		encoder.init_finetuning()

###################################################################################################################################################

class ParallelRNNClassifier(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### ATTRIBUTES
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.band_names = self.mdl_kwargs['band_names']
		self.rnn_cell_name = self.mdl_kwargs['rnn_cell_name']

		### ENCODER
		encoder = rnn_encoders.RNNEncoderP(**self.mdl_kwargs)
		embd_dims = self.mdl_kwargs['embd_dims']
		
		# ### DECODER
		dec_mdl_kwargs = deepcopy(self.mdl_kwargs)
		dec_mdl_kwargs['embd_dims'] = dec_mdl_kwargs['embd_dims']
		decoder = GLOBAL_DECODER_CLASS(**dec_mdl_kwargs)

		### MODEL
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':embd_dims})
		self.classifier = self.class_mdl_kwargs['C'](**self.class_mdl_kwargs)

	def get_name(self):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		return get_model_name({
			'mdl':f'ParallelRNN',
			'input_dims':f'{self.input_dims}',
			'enc_emb':get_enc_emb_str(encoder, self.band_names),
			'dec_emb':get_enc_emb_str(decoder, self.band_names),
			'cell':f'{self.rnn_cell_name}',
		})

	def forward(self, tdict, **kwargs):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		encoder_tdict = encoder(tdict, **kwargs)
		decoder_tdict = decoder(encoder_tdict)
		classifier_tdict = self.classifier(encoder_tdict)
		#print_tdict(encoder_tdict)
		return classifier_tdict

###################################################################################################################################################

class SerialRNNClassifier(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### ATTRIBUTES
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.band_names = self.mdl_kwargs['band_names']
		self.rnn_cell_name = self.mdl_kwargs['rnn_cell_name']

		### ENCODER
		encoder = rnn_encoders.RNNEncoderS(**self.mdl_kwargs)
		embd_dims = self.mdl_kwargs['embd_dims']

		### DECODER
		dec_mdl_kwargs = deepcopy(self.mdl_kwargs)
		dec_mdl_kwargs['embd_dims'] = dec_mdl_kwargs['embd_dims']
		decoder = GLOBAL_DECODER_CLASS(**dec_mdl_kwargs)
		
		### MODEL
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':embd_dims})
		self.classifier = self.class_mdl_kwargs['C'](**self.class_mdl_kwargs)

	def get_name(self):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		return get_model_name({
			'mdl':f'SerialRNN',
			'input_dims':f'{self.input_dims}',
			'enc_emb':get_enc_emb_str(encoder, self.band_names),
			'dec_emb':get_enc_emb_str(decoder, self.band_names),
			'cell':f'{self.rnn_cell_name}',
		})

	def forward(self, tdict:dict, **kwargs):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		encoder_tdict = encoder(tdict, **kwargs)
		decoder_tdict = decoder(encoder_tdict)
		classifier_tdict = self.classifier(encoder_tdict)
		return classifier_tdict

###################################################################################################################################################

class ParallelTimeSelfAttn(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### ATTRIBUTES
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.band_names = self.mdl_kwargs['band_names']

		### ENCODER
		encoder = attn_encoders.TimeSelfAttnEncoderP(**self.mdl_kwargs)
		embd_dims = self.mdl_kwargs['embd_dims']
			
		### DECODER
		dec_mdl_kwargs = deepcopy(self.mdl_kwargs)
		dec_mdl_kwargs['embd_dims'] = dec_mdl_kwargs['embd_dims']
		decoder = GLOBAL_DECODER_CLASS(**dec_mdl_kwargs)
		
		### MODEL
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':embd_dims})
		self.classifier = self.class_mdl_kwargs['C'](**self.class_mdl_kwargs)

	def get_info(self):
		d = {
			'encoder':self.autoencoder['encoder'].get_info(),
			}
		return d

	def get_name(self):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		return get_model_name({
			'mdl':f'ParallelTimeModAttn',
			'input_dims':f'{self.input_dims}',
			'm':self.mdl_kwargs['te_features'],
			'kernel_size':self.mdl_kwargs['kernel_size'],
			'heads':self.mdl_kwargs['heads'],
			'fourier_dims':self.mdl_kwargs['fourier_dims'],
			'time_noise_window':self.mdl_kwargs['time_noise_window'],
			'enc_emb':get_enc_emb_str(encoder, self.band_names),
			'dec_emb':get_enc_emb_str(decoder, self.band_names),
		})

	def forward(self, tdict:dict, **kwargs):
		# with profiler.record_function('encoder'):
		encoder = self.autoencoder['encoder']
		# with profiler.record_function('decoder'):
		decoder = self.autoencoder['decoder']
		encoder_tdict = encoder(tdict, **kwargs)
		decoder_tdict = decoder(encoder_tdict)
		classifier_tdict = self.classifier(encoder_tdict)
		return classifier_tdict

###################################################################################################################################################

class SerialTimeSelfAttn(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()
		### ATTRIBUTES
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.band_names = self.mdl_kwargs['band_names']

		### ENCODER
		encoder = attn_encoders.TimeSelfAttnEncoderS(**self.mdl_kwargs)
		embd_dims = self.mdl_kwargs['embd_dims']
		
		### DECODER
		dec_mdl_kwargs = deepcopy(self.mdl_kwargs)
		dec_mdl_kwargs['embd_dims'] = dec_mdl_kwargs['embd_dims']
		decoder = GLOBAL_DECODER_CLASS(**dec_mdl_kwargs)

		### MODEL
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':embd_dims})
		self.classifier = self.class_mdl_kwargs['C'](**self.class_mdl_kwargs)

	def get_info(self):
		d = {
			'encoder':self.autoencoder['encoder'].get_info(),
			}
		return d

	def get_name(self):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		return get_model_name({
			'mdl':f'SerialTimeModAttn',
			'input_dims':f'{self.input_dims}',
			'm':self.mdl_kwargs['te_features'],
			'kernel_size':self.mdl_kwargs['kernel_size'],
			'heads':self.mdl_kwargs['heads'],
			'fourier_dims':self.mdl_kwargs['fourier_dims'],
			'time_noise_window':self.mdl_kwargs['time_noise_window'],
			'enc_emb':get_enc_emb_str(encoder, self.band_names),
			'dec_emb':get_enc_emb_str(decoder, self.band_names),
		})

	def forward(self, tdict:dict, **kwargs):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		encoder_tdict = encoder(tdict, **kwargs)
		decoder_tdict = decoder(encoder_tdict)
		classifier_tdict = self.classifier(encoder_tdict)
		return classifier_tdict