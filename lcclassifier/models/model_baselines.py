from __future__ import print_function
from __future__ import division
from . import C_

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fuzzytorch.utils import get_model_name, print_tdict
from copy import copy, deepcopy

from .rnn import decoders as rnn_decoders
GLOBAL_DECODER_CLASS = rnn_decoders.LatentGRUDecoderP

###################################################################################################################################################

def get_enc_emb_str(mdl, band_names):
	dims = mdl.get_embd_dims_list()
	if isinstance(dims, dict):
		txts = ['-'.join([f'{b}{d}' for d in dims[b]]) for b in band_names]
		return '.'.join(txts)
	else:
		txts = [f'{d}' for d in dims]
		return '-'.join(txts)

class ModelBaseline(nn.Module):
	def __init__(self, **raw_kwargs):
		super().__init__()

	def get_output_dims(self):
		return self.encoder.get_output_dims()

	def get_autoencoder_model(self):
		return self.autoencoder

	def get_classifier_model(self):
		return self.classifier

	def init_fine_tuning(self):
		encoder = self.autoencoder['encoder']
		encoder.init_fine_tuning()

###################################################################################################################################################

from .rnn import encoders as rnn_encoders

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
			'enc-emb':get_enc_emb_str(encoder, self.band_names),
			'dec-emb':get_enc_emb_str(decoder, self.band_names),
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
			'enc-emb':get_enc_emb_str(encoder, self.band_names),
			'dec-emb':get_enc_emb_str(decoder, self.band_names),
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

from .attn import encoders as attn_encoders

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
			'fourier_dims':self.mdl_kwargs['fourier_dims'],
			'time_noise_window':self.mdl_kwargs['time_noise_window'],
			'enc-emb':get_enc_emb_str(encoder, self.band_names),
			'dec-emb':get_enc_emb_str(decoder, self.band_names),
		})

	def forward(self, tdict:dict, **kwargs):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		encoder_tdict = encoder(tdict, **kwargs)
		decoder_tdict = decoder(encoder_tdict)
		classifier_tdict = self.classifier(encoder_tdict)
		return classifier_tdict

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
			'fourier_dims':self.mdl_kwargs['fourier_dims'],
			'time_noise_window':self.mdl_kwargs['time_noise_window'],
			'enc-emb':get_enc_emb_str(encoder, self.band_names),
			'dec-emb':get_enc_emb_str(decoder, self.band_names),
		})

	def forward(self, tdict:dict, **kwargs):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		encoder_tdict = encoder(tdict, **kwargs)
		decoder_tdict = decoder(encoder_tdict)
		classifier_tdict = self.classifier(encoder_tdict)
		return classifier_tdict