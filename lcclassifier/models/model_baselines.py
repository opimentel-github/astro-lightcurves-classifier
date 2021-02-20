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

class ParallelRNNClassifier(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### ATTRIBUTES
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.te_features = self.mdl_kwargs['te_features']
		self.band_names = self.mdl_kwargs['band_names']
		self.rnn_cell_name = self.mdl_kwargs['rnn_cell_name']

		### MODEL DEFINITION
		encoder = rnn_encoders.RNNEncoderP(**self.mdl_kwargs)
		embd_dims = self.mdl_kwargs['rnn_embd_dims']
		self.dec_mdl_kwargs.update({'input_dims':embd_dims, 'rnn_embd_dims':embd_dims})
		decoder = self.dec_mdl_kwargs['C'](**self.dec_mdl_kwargs)
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':embd_dims})
		self.classifier = self.class_mdl_kwargs['C'](**self.class_mdl_kwargs)

	def get_name(self):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		return get_model_name({
			'mdl':f'ParallelRNN',
			'in-dims':f'{self.input_dims}',
			'te-dims':f'{self.te_features}',
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
		self.te_features = self.mdl_kwargs['te_features']
		self.band_names = self.mdl_kwargs['band_names']
		self.rnn_cell_name = self.mdl_kwargs['rnn_cell_name']

		### MODEL DEFINITION
		encoder = rnn_encoders.RNNEncoderS(**self.mdl_kwargs)
		embd_dims = self.mdl_kwargs['rnn_embd_dims']
		self.dec_mdl_kwargs.update({'input_dims':embd_dims, 'rnn_embd_dims':embd_dims})
		decoder = self.dec_mdl_kwargs['C'](**self.dec_mdl_kwargs)
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':embd_dims})
		self.classifier = self.class_mdl_kwargs['C'](**self.class_mdl_kwargs)

	def get_name(self):
		encoder = self.autoencoder['encoder']
		decoder = self.autoencoder['decoder']
		return get_model_name({
			'mdl':f'SerialRNN',
			'in-dims':f'{self.input_dims}',
			'te-dims':f'{self.te_features}',
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

from .tcnn import encoders as tcnn_encoders

class ParallelTCNNClassifier(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### ATTRIBUTES
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.te_features = self.mdl_kwargs['te_features']
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
			'te-dims':f'{self.te_features}',
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
		self.te_features = self.mdl_kwargs['te_features']
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
			'te-dims':f'{self.te_features}',
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

###################################################################################################################################################

from .attn import encoders as attn_encoders

class ParallelAttnTCNNClassifier(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### ATTRIBUTES
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.te_features = self.mdl_kwargs['te_features']
		self.band_names = self.mdl_kwargs['band_names']

		### MODEL DEFINITION
		encoder = attn_encoders.AttnTCNNEncoderP(**self.mdl_kwargs)
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
			'mdl':f'ParallelTimeSelfAttn',
			'in-dims':f'{self.input_dims}',
			'te-dims':f'{self.te_features}',
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

class SerialAttnTCNNClassifier(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### ATTRIBUTES
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.te_features = self.mdl_kwargs['te_features']
		self.band_names = self.mdl_kwargs['band_names']

		### MODEL DEFINITION
		encoder = attn_encoders.AttnTCNNEncoderS(**self.mdl_kwargs)
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
			'mdl':f'SerialTimeSelfAttn',
			'in-dims':f'{self.input_dims}',
			'te-dims':f'{self.te_features}',
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

'''

#class SerialSelfAttentionClassifier(nn.Module):

#class ParallelSelfAttentionClassifier(nn.Module):

class SerialCustomSelfAttentionClassifier(nn.Module):
	def __init__(self,
		**old_kwargs):
		super().__init__()
		### ATTRIBUTES
		for name, val in old_kwargs.items():
			setattr(self, name, val)
		### CHECKS
		assert self.attn_embd_dims%self.attn_heads==0

		### CHANGES
		kwargs = old_kwargs.copy()
		kwargs['attn_layers'] = get_pow2_layers(self.attn_embd_dims, self.attn_layers)
		for name, val in kwargs.items():
			setattr(self, name, val)

		### MODEL DEF
		#self.embedding = attn_models.SelfAttentionEmbedding(**kwargs)
		self.embedding = attn_models.TimeSelfAttentionEmbeddingS(**kwargs)
		#self.embedding = attn_models.TimeErrorSelfAttentionEmbedding(**kwargs)
		class_kwargs = {
			'dropout_info':self.dropout_info,
			'band_names':self.band_names,
		}
		self.classifier = self.classifier_class(self.attn_layers[-1], self.output_dims, **class_kwargs)
		self.get_name()

	def get_name(self):
		name = f'mdl-SerialCustomSelfAttention_'
		name += f'inD-{self.input_dims}_'
		name += f'teD-{self.te_features}_'

		name += f'attnL-{self.attn_layers}_'
		name += f'attnH-{self.attn_heads}_'
		name += f'attnU-{self.attn_embd_dims}_'
		self.name = name[:-1]
		return self.name

	def get_output_dims(self):
		return self.embedding.get_output_dims()

	def forward(self, data:dict, **kwargs):
		model_out = self.embedding(data, **kwargs)
		return self.classifier(model_out)

class ParallelCustomSelfAttentionClassifier(nn.Module):
	def __init__(self,
		**old_kwargs):
		super().__init__()
		### ATTRIBUTES
		for name, val in old_kwargs.items():
			setattr(self, name, val)
		### CHECKS
		assert self.attn_embd_dims%self.attn_heads==0

		### CHANGES
		kwargs = old_kwargs.copy()
		kwargs['attn_layers'] = get_pow2_layers(self.attn_embd_dims, self.attn_layers)
		for name, val in kwargs.items():
			setattr(self, name, val)

		### MODEL DEF
		#self.embedding = attn_models.SelfAttentionEmbedding(**kwargs)
		self.embedding = attn_models.TimeSelfAttentionEmbeddingP(**kwargs)
		#self.embedding = attn_models.TimeErrorSelfAttentionEmbedding(**kwargs)
		class_kwargs = {
			'dropout_info':self.dropout_info,
			'band_names':self.band_names,
		}
		self.classifier = self.classifier_class(self.attn_layers[-1]*2, self.output_dims, **class_kwargs)
		self.get_name()

	def get_name(self):
		name = f'mdl-SerialCustomSelfAttention_'
		name += f'inD-{self.input_dims}_'
		name += f'teD-{self.te_features}_'

		name += f'attnL-{self.attn_layers}_'
		name += f'attnH-{self.attn_heads}_'
		name += f'attnU-{self.attn_embd_dims}_'
		self.name = name[:-1]
		return self.name

	def get_output_dims(self):
		return self.embedding.get_output_dims()

	def forward(self, data:dict, **kwargs):
		model_out = self.embedding(data, **kwargs)
		return self.classifier(model_out)



from .set_functions import models as set_functions

class SetFunctionsEmb(nn.Module):
	def __init__(self, **kwargs):
		#### ATTRIBUTES
		setattr(self, 'aggr_fun', 'mean')
		setattr(self, 'h_units', 60)
		setattr(self, 'h_layers', 2)
		setattr(self, 'g_units', 60)
		setattr(self, 'g_layers', 1)
		setattr(self, 'attn_fun', 'scaled_dot')
		setattr(self, 'attn_units', None)
		setattr(self, 'classifier_hidden_layers', 1)
		setattr(self, 'encoding_mode', 'power2')
		setattr(self, 'attn_heads', 2)
		setattr(self, 'encoding_features', 4)
		for name, val in kwargs.items():
			setattr(self, name, val)

		if self.attn_units is None:
			self.attn_units = self.g_layers

		# ATTRIBUTES
		input_dims = self.MB['input_indexs']['features'].sum()
		self.MB_constructor_dic = {
			'MB':self.MB,
			'P':{
				'curvelength_max':self.curvelength_max,
				'dropout_info':self.dropout_info,
				'class':set_functions.SetFunctionsEmbedding,
				'aggr_fun':'mean',
				'h_layers':self.h_layers,
				'h_units':self.h_units,
				'g_layers':self.g_layers,
				'g_units':self.g_units,
				'encoding_mode':self.encoding_mode,
				'encoding_features':self.encoding_features,
				'min_day':self.min_day,
				'max_day':self.max_day,
			},
			'S':{
				'curvelength_max':self.curvelength_max,
				'dropout_info':self.dropout_info,
				'class':attn_models.SetFunctionsClassifier,
				'attn_fun':self.attn_fun,
				'attn_units':self.attn_units,
				'classifier_hidden_layers':self.classifier_hidden_layers,
				'attn_heads':self.attn_heads,
				'band_names':self.band_names,
				'v_in':(input_dims+self.encoding_features),
				'k_in':self.g_units+(input_dims+self.encoding_features),
				'q_in':80,
			},
		}
	
		super().__init__()
		self.get_name()

	def get_name(self):
		name = 'mdl-SeFT_'
		name += '{}-{}_'.format('inD', self.MB['input_indexs']['features'].sum())
		name += '{}-{}_'.format('fun', self.aggr_fun)
		name += '{}-{}_'.format('hU', self.h_units)
		name += '{}-{}_'.format('gU', self.g_units)
		name += '{}-{}_'.format('TEf', self.encoding_features)
		name += '{}-{}_'.format('TEm', self.encoding_mode)
		self.name = name[:-1] + super().get_name()
		return self.name


class CNNSetFunctionsClassifier(ParallelSerialConstructor):
	def __init__(self, **kwargs):
		#### ATTRIBUTES
		setattr(self, 'aggr_fun', 'mean')
		setattr(self, 'h_units', 60)
		setattr(self, 'h_layers', 2)
		setattr(self, 'g_units', 60)
		setattr(self, 'g_layers', 1)
		setattr(self, 'attn_fun', 'scaled_dot')
		setattr(self, 'attn_units', None)
		setattr(self, 'classifier_hidden_layers', 1)
		setattr(self, 'encoding_mode', 'power2')
		setattr(self, 'attn_heads', 2)
		setattr(self, 'encoding_features', 4)
		for name, val in kwargs.items():
			setattr(self, name, val)

		if self.attn_units is None:
			self.attn_units = self.g_layers

		# ATTRIBUTES
		input_dims = self.MB['input_indexs']['features'].sum()
		self.MB_constructor_dic = {
			'MB':self.MB,
			'P':{
				'curvelength_max':self.curvelength_max,
				'dropout_info':self.dropout_info,
				'class':set_functions.CNNSetFunctionsEmbedding,
				'aggr_fun':'mean',
				'h_layers':self.h_layers,
				'h_units':self.h_units,
				'g_layers':self.g_layers,
				'g_units':self.g_units,
				'encoding_mode':self.encoding_mode,
				'encoding_features':self.encoding_features,
				'min_day':self.min_day,
				'max_day':self.max_day,
			},
			'S':{
				'curvelength_max':self.curvelength_max,
				'dropout_info':self.dropout_info,
				'class':attn_models.SetFunctionsClassifier,
				'attn_fun':self.attn_fun,
				'attn_units':self.attn_units,
				'classifier_hidden_layers':self.classifier_hidden_layers,
				'attn_heads':self.attn_heads,
				'band_names':self.band_names,
				'v_in':(input_dims+self.encoding_features),
				'k_in':self.g_units+(input_dims+self.encoding_features),
				'q_in':80,
			},
		}
	
		super().__init__()
		self.get_name()

	def get_name(self):
		name = 'mdl-SeFT_'
		name += '{}-{}_'.format('inD', self.MB['input_indexs']['features'].sum())
		name += '{}-{}_'.format('fun', self.aggr_fun)
		name += '{}-{}_'.format('hU', self.h_units)
		name += '{}-{}_'.format('gU', self.g_units)
		name += '{}-{}_'.format('TEf', self.encoding_features)
		name += '{}-{}_'.format('TEm', self.encoding_mode)
		self.name = name[:-1] + super().get_name()
		return self.name

'''