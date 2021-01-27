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
		self.max_day = 70.
		self.embd_dims = GDIter(32,64)
		self.embd_layers = GDIter(2)
		self.rnn_cell_names = GDIter('GRU','LSTM')
		self.te_features_iter = GDIter(8)#4,8,16)
		self.common_dict = {
			'band_names':lcdataset['raw'].band_names,
			'output_dims':len(lcdataset['raw'].class_names),
		}
		self.base_dict = {
			'dec_mdl_kwargs':{
				'C':rnn_decoders.RNNDecoderS, # RNNDecoderP RNNDecoderS
				'rnn_cell_name':'GRU',
				'rnn_layers':1,
				'dropout':{'p':0.1},
			},
			'class_mdl_kwargs':{
				'C':mclass.SimpleClassifier,
				'embd_layers':1,
				'dropout':{'p':0.1},
			},
		}
		self.reset()

	def reset(self):
		self.pms = []

	def __repr__(self):
		txt = ''
		for k,pm in enumerate(self.pms):
			txt += f'({k}) - mdl_kwargs: {pm["mdl_kwargs"]}\n'
			txt += f'({k}) - dataset_kwargs: {pm["dataset_kwargs"]}\n'
			txt += f'({k}) - dec_mdl_kwargs: {pm["dec_mdl_kwargs"]}\n'
			txt += f'({k}) - class_mdl_kwargs: {pm["class_mdl_kwargs"]}\n'
			txt += f'---\n'
		return txt

	def add_gs(self, gs):
		new_pms = []
		pms = gs.get_dicts()
		for pm in pms:
			bd = self.base_dict.copy()
			bd.update(pm)
			new_pms += GridSeacher(bd).get_dicts()

		for pm in new_pms:
			for k in self.common_dict.keys():
				for k2 in pm.keys():
					if k2!='dataset_kwargs':
						pm[k2][k] = self.common_dict[k]
		self.pms += new_pms

	def update_dt(self, gs):
		gs.update({
			'dataset_kwargs':{
				'attrs':['d_days', 'log_obs', 'log_obse'],
				'max_day':self.max_day,
				'te_features':0,
			}})
		return gs

	def update_te(self, gs):
		gs.update({
			'dataset_kwargs':{
				'attrs':['log_obs', 'log_obse'],
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
				'dropout':{'p':0.1},
			},
		})
		self.add_gs(self.update_dt(gs))

	def parallel_rnn_models_te(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelRNNClassifier,
				'rnn_cell_name':self.rnn_cell_names,
				'rnn_embd_dims':self.embd_dims,
				'rnn_layers':self.embd_layers,
				'dropout':{'p':0.1},
			},
		})
		self.add_gs(self.update_te(gs))

	def parallel_rnn_models(self):
		self.parallel_rnn_models_dt()
		self.parallel_rnn_models_te()

	def serial_rnn_models_dt(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialRNNClassifier,
				'rnn_cell_name':self.rnn_cell_names,
				'rnn_embd_dims':self.embd_dims,
				'rnn_layers':self.embd_layers,
				'dropout':{'p':0.1},
			},
		})
		self.add_gs(self.update_dt(gs))

	def serial_rnn_models_te(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialRNNClassifier,
				'rnn_cell_name':self.rnn_cell_names,
				'rnn_embd_dims':self.embd_dims,
				'rnn_layers':self.embd_layers,
				'dropout':{'p':0.1},
			},
		})
		self.add_gs(self.update_te(gs))

	def serial_rnn_models(self):
		self.serial_rnn_models_dt()
		self.serial_rnn_models_te()

###################################################################################################################################################

	def parallel_tcn_models_dt(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTCNClassifier,
				'tcn_embd_dims':self.embd_dims,
				'tcn_layers':self.embd_layers,
				'dropout':{'p':0.1},
			},
		})
		self.add_gs(self.update_dt(gs))

	def parallel_tcn_models_te(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTCNClassifier,
				'tcn_embd_dims':self.embd_dims,
				'tcn_layers':self.embd_layers,
				'dropout':{'p':0.1},
			},
		})
		self.add_gs(self.update_te(gs))

	def parallel_tcn_models(self):
		self.parallel_tcn_models_dt()
		self.parallel_tcn_models_te()

	def serial_tcn_models_dt(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTCNClassifier,
				'tcn_embd_dims':self.embd_dims,
				'tcn_layers':self.embd_layers,
				'dropout':{'p':0.1},
			},
		})
		self.add_gs(self.update_dt(gs))

	def serial_tcn_models_te(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTCNClassifier,
				'tcn_embd_dims':self.embd_dims,
				'tcn_layers':self.embd_layers,
				'dropout':{'p':0.1},
			},
		})
		self.add_gs(self.update_te(gs))

	def serial_tcn_models(self):
		self.serial_tcn_models_dt()
		self.serial_tcn_models_te()

###################################################################################################################################################

	def parallel_atcn_models_dt(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelAttnTCNClassifier,
				'tcn_embd_dims':self.embd_dims,
				'tcn_layers':self.embd_layers,
				'dropout':{'p':0.1},
			},
		})
		self.add_gs(self.update_dt(gs))

	def parallel_atcn_models_te(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelAttnTCNClassifier,
				'tcn_embd_dims':self.embd_dims,
				'tcn_layers':self.embd_layers,
				'dropout':{'p':0.1},
			},
		})
		self.add_gs(self.update_te(gs))

	def parallel_atcn_models(self):
		self.parallel_atcn_models_dt()
		self.parallel_atcn_models_te()

	def serial_atcn_models_dt(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialAttnTCNClassifier,
				'tcn_embd_dims':self.embd_dims,
				'tcn_layers':self.embd_layers,
				'dropout':{'p':0.1},
			},
		})
		self.add_gs(self.update_dt(gs))

	def serial_atcn_models_te(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialAttnTCNClassifier,
				'tcn_embd_dims':self.embd_dims,
				'tcn_layers':self.embd_layers,
				'dropout':{'p':0.1},
			},
		})
		self.add_gs(self.update_te(gs))

	def serial_atcn_models(self):
		self.serial_atcn_models_dt()
		self.serial_atcn_models_te()
