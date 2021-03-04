from __future__ import print_function
from __future__ import division
from __future__ import annotations
from . import C_

from flamingchoripan.datascience.grid_search import GDIter, GridSeacher
from . import model_baselines as mbls
from . import classifiers as mclass
from .rnn import decoders as rnn_decoders

###################################################################################################################################################

class ModelCollections():
	def __init__(self, lcdataset):
		self.lcdataset = lcdataset
		self.max_day = C_.MAX_DAY
		self.embd_dims = GDIter(64) # importante 60 80 100
		self.embd_layers = GDIter(2)
		self.rnn_cell_names = GDIter('GRU', 'LSTM')
		self.te_features_iter = GDIter(32)
		#self.te_features_iter = GDIter(32, 4, 8, 16)
		self.dropout_p = .3 # .1 .2 .25 .3
		self.common_dict = {
			'max_te_period':self.max_day*2,
			'band_names':lcdataset['raw'].band_names,
			'output_dims':len(lcdataset['raw'].class_names),
		}
		self.base_dict = {
			'dec_mdl_kwargs':{
				'C':rnn_decoders.RNNDecoderP, # RNNDecoderP RNNDecoderS
				'rnn_cell_name':'GRU',
				'rnn_layers':1,
				'dropout':{'p':self.dropout_p},
			},
			'class_mdl_kwargs':{
				'C':mclass.SimpleClassifier,
				'embd_layers':1, # 1 2
				'dropout':{'p':.5}, # .2 .25
			},
		}
		self.reset()

	def reset(self):
		self.mps = []

	def __repr__(self):
		txt = ''
		for k,mp in enumerate(self.mps):
			txt += f'({k}) - mdl_kwargs: {mp["mdl_kwargs"]}\n'
			txt += f'({k}) - dataset_kwargs: {mp["dataset_kwargs"]}\n'
			txt += f'({k}) - dec_mdl_kwargs: {mp["dec_mdl_kwargs"]}\n'
			txt += f'({k}) - class_mdl_kwargs: {mp["class_mdl_kwargs"]}\n'
			txt += f'---\n'
		return txt

	def add_gs(self, gs):
		new_mps = []
		mps = gs.get_dicts()
		for mp in mps:
			bd = self.base_dict.copy()
			bd.update(mp)
			new_mps += GridSeacher(bd).get_dicts()

		for mp in new_mps:
			for k in self.common_dict.keys():
				for k2 in mp.keys():
					if k2!='dataset_kwargs':
						mp[k2][k] = self.common_dict[k]
		self.mps += new_mps

	def update_dt(self, gs):
		gs.update({
			'dataset_kwargs':{
				'in_attrs':['d_days', 'obs', 'obse'],
				#'in_attrs':['d_days', 'obs'],
				'rec_attr':'obs',
				'max_day':self.max_day,
				'te_features':0,
			}})
		return gs

	def update_te(self, gs):
		gs.update({
			'dataset_kwargs':{
				'in_attrs':['obs', 'obse'],
				#'in_attrs':['obs'],
				'rec_attr':'obs',
				'max_day':self.max_day,
				'te_features':self.te_features_iter,
			}})
		return gs

###################################################################################################################################################

	def parallel_rnn_models_dt(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelRNNClassifier,
				'rnn_cell_name':self.rnn_cell_names,
				'rnn_embd_dims':self.embd_dims,
				'rnn_layers':self.embd_layers,
				'dropout':{'p':self.dropout_p},
			},
		})
		self.add_gs(self.update_dt(gs))

	def serial_rnn_models_dt(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialRNNClassifier,
				'rnn_cell_name':self.rnn_cell_names,
				'rnn_embd_dims':self.embd_dims,
				'rnn_layers':self.embd_layers,
				'dropout':{'p':self.dropout_p},
			},
		})
		self.add_gs(self.update_dt(gs))

	def all_rnn_models(self):
		self.parallel_rnn_models_dt()
		self.serial_rnn_models_dt()

###################################################################################################################################################

	def parallel_tcnn_models_dt(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTCNNClassifier,
				'tcnn_embd_dims':self.embd_dims,
				'tcnn_layers':self.embd_layers,
				'dropout':{'p':self.dropout_p},
				'aggregation':GDIter('max', 'avg'),
			},
		})
		self.add_gs(self.update_dt(gs))

	def serial_tcnn_models_dt(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTCNNClassifier,
				'tcnn_embd_dims':self.embd_dims,
				'tcnn_layers':self.embd_layers,
				'dropout':{'p':self.dropout_p},
				'aggregation':GDIter('max', 'avg'),
			},
		})
		self.add_gs(self.update_dt(gs))

	def all_tcnn_models(self):
		self.parallel_tcnn_models_dt()
		self.serial_tcnn_models_dt()

###################################################################################################################################################

	def parallel_attn_models_te(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelAttnTCNNClassifier,
				'tcnn_embd_dims':self.embd_dims,
				'tcnn_layers':self.embd_layers,
				'dropout':{'p':self.dropout_p},
			},
		})
		self.add_gs(self.update_te(gs))

	def serial_attn_models_te(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialAttnTCNNClassifier,
				'tcnn_embd_dims':self.embd_dims,
				'tcnn_layers':self.embd_layers,
				'dropout':{'p':self.dropout_p},
			},
		})
		self.add_gs(self.update_te(gs))

	def all_attn_models(self):
		self.parallel_attn_models_te()
		self.serial_attn_models_te()
