from __future__ import print_function
from __future__ import division
from __future__ import annotations
from . import C_

from fuzzytools.datascience.grid_search import GDIter, GridSeacher
from . import model_baselines as mbls
from . import classifiers as mclass
from .rnn import decoders as rnn_decoders

MAX_DAY = C_.MAX_DAY
USES_OBSE_INPUT = 0

###################################################################################################################################################

class ModelCollections():
	def __init__(self, lcdataset):
		self.lcdataset = lcdataset
		self.max_day = MAX_DAY

		bands = 2
		d = 64 # 16 32 64 128
		self.gd_embd_dims = GDIter(d*bands)
		self.gd_layers = GDIter(1) # 1 2 3

		p = 0/100 # 0 1
		self.dropout_d = {
			'p':p,
			'r':p,
			}
		self.common_dict = {
			'max_period':self.max_day*1.25, # ***
			'band_names':lcdataset[lcdataset.get_lcset_names()[0]].band_names,
			'output_dims':len(lcdataset[lcdataset.get_lcset_names()[0]].class_names),
			}
		self.base_dict = {
			'class_mdl_kwargs':{
				'C':mclass.SimpleClassifier,
				'layers':2, # 1 2
				'dropout':{
					'p':50/100,
					},
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
			txt += f'({k}) - class_mdl_kwargs: {mp["class_mdl_kwargs"]}\n'
			txt += f'---\n'
		return txt

	def add_gs(self, gs):
		new_mps = []
		gs.update({
			'dataset_kwargs':{
				'in_attrs':['obs']+(['obse'] if USES_OBSE_INPUT else []),
				'rec_attr':'obs',
				'max_day':self.max_day,
			}})
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

###################################################################################################################################################

	def p_rnn_models(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelRNNClassifier,
				'rnn_cell_name':GDIter('GRU', 'LSTM'),
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,
			},
		})
		self.add_gs(gs)

	def s_rnn_models(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialRNNClassifier,
				'rnn_cell_name':GDIter('GRU', 'LSTM'),
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,
			},
		})
		self.add_gs(gs)

###################################################################################################################################################

	def p_attn_model(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTimeSelfAttn,
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,

				'fourier_dims':GDIter(1),
				'te_features':GDIter(16),
				'kernel_size':GDIter(1),
				'heads':GDIter(4),
				'time_noise_window':GDIter('6*24**-1'),
			},
		})
		self.add_gs(gs)

	def p_attn_models_te(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTimeSelfAttn,
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,

				'fourier_dims':GDIter(1),
				'te_features':GDIter(2, 4, 8, 16),
				'kernel_size':GDIter(1),
				'heads':GDIter(4),
				'time_noise_window':GDIter('0*24**-1'),
			},
		})
		self.add_gs(gs)

	def p_attn_models_noise(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTimeSelfAttn,
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,

				'fourier_dims':GDIter(1),
				'te_features':GDIter(2),
				'kernel_size':GDIter(1),
				'heads':GDIter(4),
				'time_noise_window':GDIter('6*24**-1', '12*24**-1', '24*24**-1'), # ***
			},
		})
		self.add_gs(gs)

	def p_attn_models_heads(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTimeSelfAttn,
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,

				'fourier_dims':GDIter(1),
				'te_features':GDIter(2),
				'kernel_size':GDIter(1),
				'heads':GDIter(8, 16), # ***
				'time_noise_window':GDIter('0*24**-1'),
			},
		})
		self.add_gs(gs)

###################################################################################################################################################

	def s_attn_model(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTimeSelfAttn,
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,

				'fourier_dims':GDIter(1),
				'te_features':GDIter(16),
				'kernel_size':GDIter(1),
				'heads':GDIter(4),
				'time_noise_window':GDIter('6*24**-1'),
			},
		})
		self.add_gs(gs)

	def s_attn_models_te(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTimeSelfAttn,
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,

				'fourier_dims':GDIter(1),
				'te_features':GDIter(2, 4, 8, 16),
				'kernel_size':GDIter(1),
				'heads':GDIter(4),
				'time_noise_window':GDIter('0*24**-1'),
			},
		})
		self.add_gs(gs)

	def s_attn_models_noise(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTimeSelfAttn,
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,

				'fourier_dims':GDIter(1),
				'te_features':GDIter(2),
				'kernel_size':GDIter(1),
				'heads':GDIter(4),
				'time_noise_window':GDIter('6*24**-1', '12*24**-1', '24*24**-1'), # ***
			},
		})
		self.add_gs(gs)

	def s_attn_models_heads(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTimeSelfAttn,
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,

				'fourier_dims':GDIter(1),
				'te_features':GDIter(2),
				'kernel_size':GDIter(1),
				'heads':GDIter(8, 16), # ***
				'time_noise_window':GDIter('0*24**-1'),
			},
		})
		self.add_gs(gs)