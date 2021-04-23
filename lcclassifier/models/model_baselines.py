from __future__ import print_function
from __future__ import division
from . import C_

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fuzzytorch.utils import get_model_name, print_tdict

###################################################################################################################################################

def get_enc_emb_str(mdl, band_names):
	dims = mdl.get_embd_dims_list()
	if isinstance(dims, dict):
		txts = ['-'.join([f'{b}{d}' for d in dims[b]]) for b in band_names]
		return '.'.join(txts)
	else:
		txts = [f'{d}' for d in dims]
		return '-'.join(txts)

from .rnn import encoders as rnn_encoders

class ModelBaseline(nn.Module):
	def __init__(self, **raw_kwargs):
		super().__init__()

	def get_output_dims(self):
		return self.encoder.get_output_dims()

	def get_autoencoder_model(self):
		return self.autoencoder

	def get_classifier_model(self):
		return self.classifier

###################################################################################################################################################

from .rnn import encoders as rnn_encoders
from .rnn import decoders as rnn_decoders

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
		embd_dims = self.mdl_kwargs['rnn_embd_dims']
		
		### DECODER
		dec_mdl_kwargs = self.mdl_kwargs.copy()
		dec_mdl_kwargs['input_dims'] = embd_dims
		decoder = rnn_decoders.RNNDecoderP(**dec_mdl_kwargs)

		### MODEL
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':embd_dims})
		self.classifier = self.class_mdl_kwargs['C'](**self.class_mdl_kwargs)

	def get_name(self):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		return get_model_name({
			'mdl':f'ParallelRNN',
			'in-dims':f'{self.input_dims}',
			'te-dims':'0',
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
		embd_dims = self.mdl_kwargs['rnn_embd_dims']

		### DECODER
		dec_mdl_kwargs = self.mdl_kwargs.copy()
		dec_mdl_kwargs['input_dims'] = embd_dims
		decoder = rnn_decoders.RNNDecoderS(**dec_mdl_kwargs)
		
		### MODEL
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':embd_dims})
		self.classifier = self.class_mdl_kwargs['C'](**self.class_mdl_kwargs)

	def get_name(self):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		return get_model_name({
			'mdl':f'SerialRNN',
			'in-dims':f'{self.input_dims}',
			'te-dims':f'0',
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
from .attn import decoders as attn_decoders

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
		embd_dims = self.mdl_kwargs['attn_embd_dims']
			
		### DECODER
		dec_mdl_kwargs = self.mdl_kwargs.copy()
		dec_mdl_kwargs['input_dims'] = embd_dims
		decoder = attn_decoders.TimeSelfAttnDecoderP(**dec_mdl_kwargs)
		
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
			'mdl':f'ParallelTimeSelfAttn',
			'in-dims':f'{self.input_dims}',
			'te-dims':self.mdl_kwargs['te_features'],
			'scale':self.mdl_kwargs['scale_mode'],
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
		embd_dims = self.mdl_kwargs['attn_embd_dims']
		
		### DECODER
		dec_mdl_kwargs = self.mdl_kwargs.copy()
		dec_mdl_kwargs['input_dims'] = embd_dims
		decoder = attn_decoders.TimeSelfAttnDecoderS(**dec_mdl_kwargs)

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
			'mdl':f'SerialTimeSelfAttn',
			'in-dims':f'{self.input_dims}',
			'te-dims':self.mdl_kwargs['te_features'],
			'scale':self.mdl_kwargs['scale_mode'],
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

###################################################################################################################################################

from .tcnn import encoders as tcnn_encoders

class ParallelTCNNClassifier(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### ATTRIBUTES
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.band_names = self.mdl_kwargs['band_names']
		self.aggregation = self.mdl_kwargs['aggregation']

		### MODEL DEFINITION
		encoder = tcnn_encoders.TCNNEncoderP(**self.mdl_kwargs)
		embd_dims = self.mdl_kwargs['tcnn_embd_dims']
		self.dec_mdl_kwargs.update({'input_dims':embd_dims, 'rnn_embd_dims':embd_dims})
		decoder = self.dec_mdl_kwargs['C'](**self.dec_mdl_kwargs)
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':embd_dims})
		self.classifier = self.class_mdl_kwargs['C'](**self.class_mdl_kwargs)

	def get_name(self):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		return get_model_name({
			'mdl':f'ParallelTCNN',
			'in-dims':f'{self.input_dims}',
			'te-dims':f'0',
			'enc-emb':get_enc_emb_str(encoder, self.band_names),
			'dec-emb':get_enc_emb_str(decoder, self.band_names),
			'aggr':f'{self.aggregation}',
		})

	def forward(self, tdict, **kwargs):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		encoder_tdict = encoder(tdict, **kwargs)
		decoder_tdict = decoder(encoder_tdict)
		classifier_tdict = self.classifier(encoder_tdict)
		#print_tdict(encoder_tdict)
		return classifier_tdict

class SerialTCNNClassifier(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### ATTRIBUTES
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.band_names = self.mdl_kwargs['band_names']
		self.aggregation = self.mdl_kwargs['aggregation']

		### MODEL DEFINITION
		encoder = tcnn_encoders.TCNNEncoderS(**self.mdl_kwargs)
		embd_dims = self.mdl_kwargs['tcnn_embd_dims']
		self.dec_mdl_kwargs.update({'input_dims':embd_dims, 'rnn_embd_dims':embd_dims})
		decoder = self.dec_mdl_kwargs['C'](**self.dec_mdl_kwargs)
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':embd_dims})
		self.classifier = self.class_mdl_kwargs['C'](**self.class_mdl_kwargs)

	def get_name(self):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		return get_model_name({
			'mdl':f'SerialTCNN',
			'in-dims':f'{self.input_dims}',
			'te-dims':f'0',
			'enc-emb':get_enc_emb_str(encoder, self.band_names),
			'dec-emb':get_enc_emb_str(decoder, self.band_names),
			'aggr':f'{self.aggregation}',
		})

	def forward(self, tdict:dict, **kwargs):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		encoder_tdict = encoder(tdict, **kwargs)
		decoder_tdict = decoder(encoder_tdict)
		classifier_tdict = self.classifier(encoder_tdict)
		return classifier_tdict