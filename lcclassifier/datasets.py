from __future__ import print_function
from __future__ import division
from . import C_

import math
import torch
import torch.tensor as Tensor
import numpy as np
from torch.utils.data import Dataset
import random
from .scalers import CustomStandardScaler, LogStandardScaler, LogQuantileTransformer
import fuzzytools.strings as strings
from joblib import Parallel, delayed
from fuzzytools.lists import get_list_chunks, get_random_item
from fuzzytools.progress_bars import ProgressBar
from fuzzytorch.utils import print_tdict
import fuzzytorch.models.seq_utils as seq_utils
from fuzzytorch.utils import TDictHolder

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
	def __init__(self, lcset_name, lcdataset, in_attrs, rec_attr,
		max_day:float=np.infty,
		max_len:int=None,
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		std_scale:float=C_.OBSE_STD_SCALE,
		cpds_p:float=C_.CPDS_P,
		balanced_repeats=1,
		training=False,
		rooted=False, # False True
		):
		self.training = training

		self.lcset_name = lcset_name
		self.lcset = lcdataset[lcset_name]
		self.lcset.reset_boostrap() # fixme
		self.lcset_info = self.lcset.get_info()

		self.append_in_ddays = 'd_days' in in_attrs
		self.in_attrs = [ia for ia in in_attrs if not ia=='d_days']
		self.rec_attr = rec_attr
		self.max_day = self.get_max_duration() if max_day is None else max_day
		self._max_day = max_day
		self.max_len = self.calcule_max_len() if max_len is None else max_len

		self.hours_noise_amp = hours_noise_amp
		self.std_scale = std_scale
		self.cpds_p = cpds_p

		self.balanced_repeats = balanced_repeats
		self.rooted = rooted
		self.reset()

	def reset(self):
		self.training = False
		self.precomputed_dict = {}
		self.band_names = self.lcset.band_names
		self.class_names = self.lcset.class_names
		self.survey = self.lcset.survey
		self.automatic_diff()
		self.calcule_in_scaler_bdict()
		self.calcule_rec_scaler_bdict()
		self.calcule_ddays_scaler_bdict()
		self.reset_max_day()
		self.calcule_poblation_weights()
		self.calcule_balanced_w_cdict()
		self.generate_balanced_lcobj_names()

	def automatic_diff(self):
		attrs = self.in_attrs+[self.rec_attr]
		for attr in attrs: # calcule derivates!
			if attr=='d_obs':
				self.lcset.set_diff_parallel('obs')
			if attr=='d_obse':
				self.lcset.set_diff_parallel('obse')

		### needed always!
		self.lcset.set_diff_parallel('days')

	def reset_max_day(self):
		self.max_day = self._max_day

	def calcule_balanced_w_cdict(self):
		self.balanced_w_cdict = self.lcset.get_class_balanced_weights_cdict()

	def calcule_poblation_weights(self):
		self.populations_cdict = self.lcset.get_populations_cdict()

	def generate_balanced_lcobj_names(self):
		min_index = np.argmin([self.populations_cdict[c] for c in self.class_names])
		min_c = self.class_names[min_index]
		#min_c_pop = self.populations_cdict[min_c]
		#print(min_c_pop, min_c)
		self.balanced_lcobj_names = self.lcset.get_lcobj_names(min_c).copy()*self.balanced_repeats
		boostrap_n = len(self.balanced_lcobj_names)
		#print(self.balanced_lcobj_names)
		#assert 0
		#to_fill_cdict = {c:max_pop-self.populations_cdict[c] for c in self.class_names}
		for c in self.class_names:
			if c==min_c:
				continue
			lcobj_names_c = self.lcset.get_boostrap_samples(c, boostrap_n, uses_counter=False, replace=True)
			#lcobj_names_c = get_random_subsampled_list(self.lcset.get_lcobj_names(c).copy(), boostrap_n)
			self.balanced_lcobj_names += lcobj_names_c

		#self.balanced_lcobj_names = balanced_lcobj_names*repeats
		

	def get_balanced_w_cdict(self):
		return self.balanced_w_cdict

	def get_output_dims(self):
		return len(self.in_attrs)+int(self.append_in_ddays)

	def __repr__(self):
		txt = f'CustomDataset('
		txt += strings.get_string_from_dict({
			'lcset_len':f'{len(self.lcset):,}',
			'class_names':self.class_names,
			'band_names':self.band_names,
			'max_day':f'{self.max_day:.2f}',
			'max_len': f'{self.max_len:,}',
			'in_attrs':self.in_attrs,
			'rec_attr':self.rec_attr,
			'append_in_ddays':self.append_in_ddays,
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

	def transfer_metadata_to(self, other):
		other.set_max_day(self.get_max_day())
		other.set_in_scaler_bdict(self.get_in_scaler_bdict())
		other.set_rec_scaler_bdict(self.get_rec_scaler_bdict())
		other.set_ddays_scaler_bdict(self.get_ddays_scaler_bdict())
	
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

###################################################################################################################################################

	def calcule_ddays_scaler_bdict(self):
		self.ddays_scaler_bdict = {}
		for kb,b in enumerate(self.band_names):
			values = self.lcset.get_lcset_values_b(b, 'd_days')[...,None]
			qt = CustomStandardScaler()
			#qt = LogStandardScaler()
			#qt = LogQuantileTransformer(n_quantiles=100, random_state=0) # slow
			qt.fit(values)
			self.ddays_scaler_bdict[b] = qt

	def calcule_in_scaler_bdict(self):
		self.in_scaler_bdict = {}
		for kb,b in enumerate(self.band_names):
			values = np.concatenate([self.lcset.get_lcset_values_b(b, in_attr)[...,None] for ka,in_attr in enumerate(self.in_attrs)], axis=-1)
			#qt = CustomStandardScaler()
			qt = LogStandardScaler()
			#qt = LogQuantileTransformer(n_quantiles=100, random_state=0) # slow
			qt.fit(values)
			self.in_scaler_bdict[b] = qt

	def calcule_rec_scaler_bdict(self):
		self.rec_scaler_bdict = {}
		for kb,b in enumerate(self.band_names):
			values = self.lcset.get_lcset_values_b(b, self.rec_attr)[...,None]
			#qt = CustomStandardScaler()
			qt = LogStandardScaler()
			#qt = LogQuantileTransformer(n_quantiles=100, random_state=0) # slow
			qt.fit(values)
			self.rec_scaler_bdict[b] = qt

###################################################################################################################################################

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

	def train(self):
		self.training = True
		self.generate_balanced_lcobj_names() # important

	def eval(self):
		self.training = False

	###################################################################################################################################################

	def get_lcobj_names(self):
		if self.training: # training
			return self.balanced_lcobj_names
		else: # eval
			return self.lcset.get_lcobj_names()

	def has_precomputed_samples(self):
		return len(self.precomputed_dict.keys())>0

	def get_random_stratified_lcobj_names(self,
		nc=1,
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
				#print('get_random_item!!')
				return get_random_item(self.precomputed_dict[lcobj_name]) # all with data augmentation
			else:
				return self.get_item(self.lcset[lcobj_name], uses_daugm=True) # all with data augmentation
		else:
			return self.get_item(self.lcset[lcobj_name], uses_daugm=False) # no data augmentation

	def precompute_samples(self, precomputed_copies,
		device='cpu',
		):
		if precomputed_copies<=0:
			return
		if not self.has_precomputed_samples():
			lcobj_names = self.get_lcobj_names()
			bar = ProgressBar(len(lcobj_names))
			for k,lcobj_name in enumerate(lcobj_names):
				bar(f'precomputed_copies={precomputed_copies} - device={device} - lcobj_name={lcobj_name}')
				self.precomputed_dict[lcobj_name] = []
				for _k,_lcobj_name in enumerate([lcobj_name]*precomputed_copies):
					r = self.get_item(self.lcset[_lcobj_name], uses_daugm=True)
					self.precomputed_dict[lcobj_name] += [r if device=='cpu' else TDictHolder(r).to(device)]

			bar.done()

	def precompute_samples_joblib(self, precomputed_copies,
		device='cpu',
		backend='threading',
		n_jobs=1, # bug?
		):
		# don't use this!
		if precomputed_copies<=0:
			return
		if not self.has_precomputed_samples():
			def job(lcobj, uses_daugm):
				return self.get_item(lcobj, uses_daugm=uses_daugm)

			lcobj_names = self.get_lcobj_names()
			bar = ProgressBar(len(lcobj_names))
			for k,lcobj_name in enumerate(lcobj_names):
				bar(f'precomputed_copies={precomputed_copies} - device={device} - lcobj_name={lcobj_name}')
				self.precomputed_dict[lcobj_name] = []
				jobs = []
				for _k,_lcobj_name in enumerate([lcobj_name]*precomputed_copies):
					jobs.append(delayed(job)(
						self.lcset[_lcobj_name],
						_k>0,
					))
				results = Parallel(n_jobs=n_jobs, backend=backend)(jobs)
				#self.precomputed_dict[lcobj_name] = results+[self.get_item(self.lcset[lcobj_name].copy(), uses_daugm=False)]
				for r in results:
					self.precomputed_dict[lcobj_name] += [r if device=='cpu' else TDictHolder(r).to(device)]

			bar.done()


	def get_item(self, _lcobj,
		uses_daugm=False,
		uses_len_clip=True,
		return_lcobjs=False,
		):
		'''
		apply data augmentation, this overrides obj information
		be sure to copy the input lcobj!!!!
		'''
		lcobj = self.lcset[_lcobj].copy() if isinstance(_lcobj, str) else _lcobj.copy()
		if uses_daugm:
			for b in lcobj.bands:
				lcobjb = lcobj.get_b(b)
				#lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
				lcobjb.add_obs_noise_gaussian(0, self.std_scale) # add obs noise
				lcobjb.apply_downsampling_window(rooted=self.rooted, apply_prob=.9) # curve points downsampling we need to ensure the model to see compelte curves
				lcobjb.apply_downsampling(0.1) # curve points downsampling

		### remove day offset!
		day_offset = lcobj.reset_day_offset_serial()

		### prepare to export
		max_day = self.max_day
		sorted_time_indexs = lcobj.get_sorted_days_indexs_serial() # get just once for performance purposes
		onehot = lcobj.get_onehot_serial(sorted_time_indexs, max_day)
		x = self.in_normalize(lcobj.get_custom_x_serial(self.in_attrs, sorted_time_indexs, max_day), onehot)
		d_days =self.ddays_normalize(lcobj.get_custom_x_serial(['d_days'], sorted_time_indexs, max_day), onehot)
		days = lcobj.get_custom_x_serial(['days'], sorted_time_indexs, max_day)
		x = np.concatenate([x, d_days], axis=-1) if self.append_in_ddays else x
		#print('x',x.shape)
		### tensor dict
		model_input = {
			'onehot':torch.as_tensor(onehot),
			'x':torch.as_tensor(x, dtype=torch.float32),
			'time':torch.as_tensor(days, dtype=torch.float32),
			'dtime':torch.as_tensor(d_days, dtype=torch.float32),
			}

		###
		rec_x =self.rec_normalize(lcobj.get_custom_x_serial([self.rec_attr], sorted_time_indexs, max_day), onehot)
		error = lcobj.get_custom_x_serial(['obse'], sorted_time_indexs, max_day)
		balanced_w = np.array([self.get_balanced_w_cdict()[self.class_names[lcobj.y]]])
		#print(balanced_w, balanced_w.shape, c)
		target = {
			'y':torch.as_tensor(lcobj.y),
			'rec_x':torch.as_tensor(rec_x, dtype=torch.float32),
			'error':torch.as_tensor(error, dtype=torch.float32),
			'balanced_w':torch.as_tensor(balanced_w, dtype=torch.float32),
			}

		###
		tdict = {
			'input':fix_new_len(model_input, uses_len_clip, self.max_len),
			'target':fix_new_len(target, uses_len_clip, self.max_len),
			}
		#print_tdict(tdict)
		if return_lcobjs:
			return tdict, lcobj
		return tdict