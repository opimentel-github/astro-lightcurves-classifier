from __future__ import print_function
from __future__ import division
from __future__ import annotations
from . import C_

from fuzzytools.datascience.grid_search import GDIter, GridSeacher
from . import model_baselines as mbls
from . import classifiers as mclass
from .rnn import decoders as rnn_decoders

MAX_DAY = C_.MAX_DAY
USES_OBSE = False

###################################################################################################################################################

class ModelCollections():
	def __init__(self, lcdataset):
		self.lcdataset = lcdataset
		self.max_day = MAX_DAY

		bands = 2
		d = 64 # 16 32 64 128
		self.gd_embd_dims = GDIter(d*bands)
		self.gd_layers = GDIter(1) # 1 2 3

		### rnn
		self.gd_rnn_cell_names = GDIter('GRU', 'LSTM')

		### attn
		self.gd_fourier_dims = GDIter(1/2)
		self.gd_te_features = GDIter(1*2, 8*2)
		self.gd_time_noise_window = GDIter('0*24**-1')
		# self.gd_time_noise_window = GDIter('0*24**-1', '24*24**-1')
		# self.gd_time_noise_window = GDIter('0*24**-1', '1*24**-1', '24*24**-1')
		self.gd_kernel_size = GDIter(1,2)
		# self.gd_kernel_size = GDIter(1, 2, 3)
		self.gd_heads = GDIter(8,4)

		p = 1/100
		self.dropout_d = {
			'p':p,
			'r':p,
			}
		self.common_dict = {
			'max_period':self.max_day*1.25, # ***
			'band_names':lcdataset['raw'].band_names,
			'output_dims':len(lcdataset['raw'].class_names),
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
				'in_attrs':['d_days', 'obs']+(['obse'] if USES_OBSE else []),
				'rec_attr':'obs',
				'max_day':self.max_day,
			}})
		return gs

	def update_te(self, gs):
		gs.update({
			'dataset_kwargs':{
				'in_attrs':['obs']+(['obse'] if USES_OBSE else []),
				'rec_attr':'obs',
				'max_day':self.max_day,
			}})
		return gs

###################################################################################################################################################

	def parallel_rnn_models(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelRNNClassifier,
				'rnn_cell_name':self.gd_rnn_cell_names,
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,
			},
		})
		self.add_gs(self.update_dt(gs))

	def serial_rnn_models(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialRNNClassifier,
				'rnn_cell_name':self.gd_rnn_cell_names,
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,
			},
		})
		self.add_gs(self.update_dt(gs))

	def all_rnn_models(self):
		self.parallel_rnn_models()
		self.serial_rnn_models()

###################################################################################################################################################

	def parallel_attn_models(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTimeSelfAttn,
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,
				'te_features':self.gd_te_features,
				'fourier_dims':self.gd_fourier_dims,
				'kernel_size':self.gd_kernel_size,
				'heads':self.gd_heads,
				'time_noise_window':self.gd_time_noise_window,
			},
		})
		self.add_gs(self.update_te(gs))

	def serial_attn_models(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTimeSelfAttn,
				'embd_dims':self.gd_embd_dims,
				'layers':self.gd_layers,
				'dropout':self.dropout_d,
				'te_features':self.gd_te_features,
				'fourier_dims':self.gd_fourier_dims,
				'kernel_size':self.gd_kernel_size,
				'heads':self.gd_heads,
				'time_noise_window':self.gd_time_noise_window,
			},
		})
		self.add_gs(self.update_te(gs))

	def all_attn_models(self):
		self.parallel_attn_models()
		self.serial_attn_models()