from __future__ import print_function
from __future__ import division
from . import C_

import math
import torch
import torch.tensor as Tensor
import numpy as np
from torch.utils.data import Dataset
import random
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from .scalers import LogStandardScaler, LogQuantileTransformer
import flamingchoripan.strings as strings
from joblib import Parallel, delayed
from flamingchoripan.lists import get_list_chunks, get_random_subsampled_list, get_random_item
from flamingchoripan.progress_bars import ProgressBar
from fuzzytorch.utils import print_tdict
import fuzzytorch.models.seq_utils as seq_utils

###################################################################################################################################################

def fix_new_len(tdict, uses_len_clip, max_len):
	new_tdict = {}
	for key in tdict.keys():
		x = tdict[key]
		is_seq_tensor = len(x.shape)==2
		if uses_len_clip and is_seq_tensor:
			new_tdict[key] = seq_utils.get_seq_clipped_shape(x, max_len)
		else:
			new_tdict[key] = x
	return new_tdict

###################################################################################################################################################

class CustomDataset(Dataset):
	def __init__(self, lcdataset, lcset_name, in_attrs, rec_attr,
		max_day:float=np.infty,
		max_te_period:float=None,
		max_len:int=None,
		te_features:int=6,
		effective_beta_eps=C_.EFFECTIVE_BETA_EPS,

		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		std_scale:float=C_.OBSE_STD_SCALE,
		cpds_p:float=C_.CPDS_P,

		uses_precomputed_samples=True,
		):
		assert te_features%2==0

		self.training = False
		self.lcset = lcdataset[lcset_name]
		self.lcset_name = lcset_name
		self.in_attrs = in_attrs.copy()
		self.rec_attr = rec_attr
		self.max_day = self.get_max_duration() if max_day is None else max_day
		self.__max_day__ = max_day
		self.max_te_period = self.max_day*2 if max_te_period is None else max_te_period
		self.max_len = self.calcule_max_len() if max_len is None else max_len
		self.te_features = te_features
		self.effective_beta_eps = effective_beta_eps

		self.hours_noise_amp = hours_noise_amp
		self.std_scale = std_scale
		self.cpds_p = cpds_p

		self.uses_precomputed_samples = uses_precomputed_samples

		self.reset()

	def reset(self):
		self.training = False
		self.precomputed_dict = {}
		self.band_names = self.lcset.band_names
		self.class_names = self.lcset.class_names
		self.survey = self.lcset.survey
		self.automatic_diff()
		self.get_te_periods()
		self.calcule_in_scaler_bdict()
		self.calcule_rec_scaler_bdict()
		self.calcule_ddays_scaler_bdict()
		self.reset_max_day()
		self.calcule_poblation_weights()
		self.generate_balanced_lcobj_names()

	def automatic_diff(self):
		attrs = self.in_attrs+[self.rec_attr]
		for attr in attrs: # calcule derivates!
			if attr=='d_days':
				self.lcset.set_diff_parallel('days')
			if attr=='d_obs':
				self.lcset.set_diff_parallel('obs')
			if attr=='d_obse':
				self.lcset.set_diff_parallel('obse')

		### need always
		self.lcset.set_diff_parallel('days')

	def get_te_periods(self):
		self.te_periods = list(np.array([self.max_te_period]*(self.te_features//2)/2**np.arange(self.te_features//2)))
		return self.te_periods

	def calcule_in_scaler_bdict(self):
		self.in_scaler_bdict = {}
		for kb,b in enumerate(self.band_names):
			values = np.concatenate([self.lcset.get_lcset_values_b(b, in_attr)[...,None] for ka,in_attr in enumerate(self.in_attrs)], axis=-1)
			#print(values.shape)
			#qt = StandardScaler()
			qt = LogStandardScaler()
			#qt = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal') # slow
			#qt = LogQuantileTransformer(n_quantiles=100, random_state=0) # slow
			qt.fit(values)
			self.in_scaler_bdict[b] = qt

	def calcule_rec_scaler_bdict(self):
		self.rec_scaler_bdict = {}
		for kb,b in enumerate(self.band_names):
			values = self.lcset.get_lcset_values_b(b, self.rec_attr)[...,None]
			#print(values.shape)
			#qt = StandardScaler()
			qt = LogStandardScaler()
			#qt = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal') # slow
			#qt = LogQuantileTransformer(n_quantiles=100, random_state=0) # slow
			qt.fit(values)
			self.rec_scaler_bdict[b] = qt

	def calcule_ddays_scaler_bdict(self):
		self.ddays_scaler_bdict = {}
		for kb,b in enumerate(self.band_names):
			values = self.lcset.get_lcset_values_b(b, 'd_days')[...,None]
			#print(values.shape)
			#qt = StandardScaler()
			qt = LogStandardScaler()
			#qt = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal') # slow
			#qt = LogQuantileTransformer(n_quantiles=100, random_state=0) # slow
			qt.fit(values)
			self.ddays_scaler_bdict[b] = qt

	def reset_max_day(self):
		self.max_day = self.__max_day__

	def calcule_poblation_weights(self):
		self.populations_cdict = self.lcset.get_populations_cdict()
		#self.poblation_weights = self.lcset.get_class_effective_weigths_cdict(1-self.effective_beta_eps) # get_class_freq_weights_cdict get_class_effective_weigths_cdict
	'''
	def generate_balanced_lcobj_names(self):
			max_index = np.argmax([self.populations_cdict[c] for c in self.class_names])
			max_c = self.class_names[max_index]
			max_c_pop = self.populations_cdict[max_c]
			#print(min_c_pop, min_c)
			to_fill_cdict = {c:max_c_pop-self.populations_cdict[c] for c in self.class_names}
			self.balanced_lcobj_names = self.lcset.get_lcobj_names().copy()
			for c in self.class_names:
				lcobj_names_c = get_random_subsampled_list(self.lcset.get_lcobj_names(c), to_fill_cdict[c])
				self.balanced_lcobj_names += lcobj_names_c
	'''

	def generate_balanced_lcobj_names(self):
		min_index = np.argmin([self.populations_cdict[c] for c in self.class_names])
		min_c = self.class_names[min_index]
		min_c_pop = self.populations_cdict[min_c]
		#print(min_c_pop, min_c)
		self.balanced_lcobj_names = self.lcset.get_lcobj_names(min_c).copy()
		#print(self.balanced_lcobj_names)
		#assert 0
		#to_fill_cdict = {c:max_pop-self.populations_cdict[c] for c in self.class_names}
		for c in self.class_names:
			if c==min_c:
				continue
			lcobj_names_c = get_random_subsampled_list(self.lcset.get_lcobj_names(c).copy(), min_c_pop)
			self.balanced_lcobj_names += lcobj_names_c
		

	def get_poblation_weights(self):
		return self.poblation_weights

	def get_te_features_dims(self):
		return self.te_features

	def get_output_dims(self):
		return len(self.in_attrs)

	def __repr__(self):
		txt = f'CustomDataset('
		txt += strings.get_string_from_dict({
			'lcset_len':f'{len(self.lcset):,}',
			'max_day':f'{self.max_day:.2f}',
			'max_len': f'{self.max_len:,}',
			'te_periods':f'{self.te_periods}',
			'in_attrs':self.in_attrs,
			'rec_attr':self.rec_attr,
			}, ', ', '=')
		txt += ')'
		return txt

	def calcule_max_len(self):
		max_len = max([len(self.lcset[lcobj_name].get_custom_x_serial(['days'], max_day=self.max_day)) for lcobj_name in self.get_lcobj_names()])
		return max_len

	def get_max_len(self):
		return self.max_len

	def set_max_len(self, max_len:int):
		self.max_len = max_len
		
	def get_max_day(self):
		return self.max_day

	def set_max_day(self, max_day:float):
		self.max_day = max_day

	def get_max_duration(self):
		self.max_duration = max([self.lcset[lcobj_name].get_days_serial_duration() for lcobj_name in self.get_lcobj_names()])
		return self.max_duration

	def set_te_periods(self, te_periods:list):
		self.te_periods = te_periods.copy()

	def set_poblation_weights(self, poblation_weights):
		self.poblation_weights = poblation_weights

	def transfer_metadata_to(self, other):
		other.set_max_day(self.get_max_day())
		other.set_in_scaler_bdict(self.get_in_scaler_bdict())
		other.set_rec_scaler_bdict(self.get_rec_scaler_bdict())
		other.set_ddays_scaler_bdict(self.get_ddays_scaler_bdict())
		other.set_te_periods(self.get_te_periods())
		#other.set_poblation_weights(self.get_poblation_weights()) # sure?
		#other.set_max_len(self.get_max_len())
	
	def get_in_scaler_bdict(self):
		return self.in_scaler_bdict

	def get_rec_scaler_bdict(self):
		return self.rec_scaler_bdict

	def get_ddays_scaler_bdict(self):
		return self.ddays_scaler_bdict

	def set_in_scaler_bdict(self, scaler_bdict):
		self.in_scaler_bdict = {k:scaler_bdict[k] for k in scaler_bdict.keys()}

	def set_rec_scaler_bdict(self, scaler_bdict):
		self.rec_scaler_bdict = {k:scaler_bdict[k] for k in scaler_bdict.keys()}

	def set_ddays_scaler_bdict(self, scaler_bdict):
		self.ddays_scaler_bdict = {k:scaler_bdict[k] for k in scaler_bdict.keys()}

	def get_rec_inverse_transform(self, model_rec_x_b, b):
		'''
		x (t)
		'''
		assert len(model_rec_x_b.shape)==1
		return self.rec_scaler_bdict[b].inverse_transform(model_rec_x_b[...,None])[...,0]

	def in_normalize(self, x, onehot):
		'''
		x (t,f)
		'''
		assert len(x.shape)==2
		assert x.shape[-1]==len(self.in_attrs)
		new_x = np.zeros_like(x) # starts with zeros!!!
		for kb,b in enumerate(self.band_names):
			onehot_b = onehot[...,kb][...,None]
			qt = self.in_scaler_bdict[b]
			new_x += qt.transform(x)*onehot_b
		return new_x

	def rec_normalize(self, x, onehot):
		'''
		x (t,1)
		'''
		assert len(x.shape)==2
		assert x.shape[-1]==1
		new_x = np.zeros_like(x) # starts with zeros!!!
		for kb,b in enumerate(self.band_names):
			onehot_b = onehot[...,kb][...,None]
			qt = self.rec_scaler_bdict[b]
			new_x += qt.transform(x)*onehot_b
		return new_x

	def ddays_normalize(self, x, onehot):
		'''
		x (t,1)
		'''
		assert len(x.shape)==2
		assert x.shape[-1]==1
		new_x = np.zeros_like(x) # starts with zeros!!!
		for kb,b in enumerate(self.band_names):
			onehot_b = onehot[...,kb][...,None]
			qt = self.ddays_scaler_bdict[b]
			new_x += qt.transform(x)*onehot_b
		return new_x
	
	def train(self):
		self.training = True
		self.generate_balanced_lcobj_names() # important

	def eval(self):
		self.training = False

	###################################################################################################################################################

	def get_lcobj_names(self):
		if self.training:
			return self.balanced_lcobj_names
		else:
			return self.lcset.get_lcobj_names()

	def has_precomputed_samples(self):
		return len(self.precomputed_dict.keys())>0

	def get_random_stratified_lcobj_names(self,
		nc=2,
		):
		# stratified, mostly used for images in experiments
		lcobj_names = []
		random_ndict = self.lcset.get_random_stratified_lcobj_names(nc)
		for c in self.class_names:
			lcobj_names += random_ndict[c]
		return lcobj_names

	def __len__(self):
		lcobj_names = self.get_lcobj_names()
		return len(lcobj_names)

	def __getitem__(self, idx:int):
		lcobj_names = self.get_lcobj_names()
		lcobj_name = lcobj_names[idx]
		if self.training:
			if self.has_precomputed_samples():
				return get_random_item(self.precomputed_dict[lcobj_name])
			else:
				return self.get_item(self.lcset[lcobj_name].copy(), uses_daugm=True)
		else:
			return self.get_item(self.lcset[lcobj_name].copy(), uses_daugm=False)

	'''
	def precompute_samples(self, precomputed_copies,
		n_jobs=C_.N_JOBS,
		chunk_size=C_.CHUNK_SIZE,
		backend=C_.JOBLIB_BACKEND,
		):
		chunk_size = n_jobs if chunk_size is None else chunk_size

		def job(lcobj):
			return [self.get_item(lcobj, uses_daugm=True) for _ in range(precomputed_copies)]

		lcobj_names = self.get_lcobj_names()
		chunks = get_list_chunks(lcobj_names, chunk_size)
		bar = ProgressBar(len(chunks))
		for k,chunk in enumerate(chunks):
			bar(f'{self.lcset_name} {chunk}')
			results = Parallel(n_jobs=n_jobs, backend=backend)([delayed(job)() for lcobj_name in chunk]) # None threading
			for lcobj_name,r in zip(chunk,results):
				self.precomputed_dict[lcobj_name] = r

		bar.done()
	'''

	def precompute_samples(self, precomputed_copies,
		backend=C_.JOBLIB_BACKEND,
		):
		def job(lcobj):
			return self.get_item(lcobj, uses_daugm=True)

		lcobj_names = self.get_lcobj_names()
		bar = ProgressBar(len(lcobj_names))
		for k,lcobj_name in enumerate(lcobj_names):
			bar(f'{lcobj_name}')
			results = Parallel(n_jobs=precomputed_copies, backend=backend)([delayed(job)(self.lcset[lcobj_name_].copy()) for lcobj_name_ in [lcobj_name]*precomputed_copies])
			self.precomputed_dict[lcobj_name] = results+[self.get_item(self.lcset[lcobj_name].copy(), uses_daugm=False)]

		bar.done()


	def get_item(self, lcobj,
		uses_daugm=False,
		uses_len_clip=True,
		return_lcobjs=False,
		):
		'''
		apply data augmentation, this overrides obj information
		be sure to copy the input lcobj!!!!
		'''
		lcobj = self.lcset[lcobj].copy() if isinstance(lcobj, str) else lcobj
		if uses_daugm:
			for b in lcobj.bands:
				lcobjb = lcobj.get_b(b)
				lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
				lcobjb.add_obs_noise_gaussian(0, self.std_scale) # add obs noise
				lcobjb.apply_downsampling(self.cpds_p) # curve points downsampling # bugs?
				pass

		### remove day offset!
		day_offset = lcobj.reset_day_offset_serial()

		### prepare to export
		max_day = self.max_day
		sorted_time_indexs = lcobj.get_sorted_days_indexs_serial() # get just once for performance purposes
		onehot = lcobj.get_onehot_serial(sorted_time_indexs, max_day)
		in_x = self.in_normalize(lcobj.get_custom_x_serial(self.in_attrs, sorted_time_indexs, max_day), onehot)
		rec_x =self.rec_normalize(lcobj.get_custom_x_serial([self.rec_attr], sorted_time_indexs, max_day), onehot)
		d_days =self.ddays_normalize(lcobj.get_custom_x_serial(['d_days'], sorted_time_indexs, max_day), onehot)
		days = lcobj.get_custom_x_serial(['days'], sorted_time_indexs, max_day)
		error = lcobj.get_custom_x_serial(['obse'], sorted_time_indexs, max_day)
		#print(in_x.shape, rec_x.shape)

		### input
		model_input = {
			'onehot':torch.as_tensor(onehot),
			'x':torch.as_tensor(in_x, dtype=torch.float32),
			'time':torch.as_tensor(days, dtype=torch.float32),
			'dtime':torch.as_tensor(d_days, dtype=torch.float32),
			'error':torch.as_tensor(error, dtype=torch.float32),
		}

		### target
		target = {
			'y':torch.as_tensor(lcobj.y),
			#'poblation_weights':torch.as_tensor([self.poblation_weights[c] for c in self.class_names], dtype=torch.float32),
			'rec-x':torch.as_tensor(rec_x, dtype=torch.float32),
		}

		tdict = {
			'input':fix_new_len(model_input, uses_len_clip, self.max_len),
			'target':fix_new_len(target, uses_len_clip, self.max_len),
			}
		#print_tdict(tdict)
		if return_lcobjs:
			return tdict, lcobj
		return tdict