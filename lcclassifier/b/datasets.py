from __future__ import print_function
from __future__ import division
from . import C_

import math
import torch
import torch.tensor as Tensor
import numpy as np
from torch.utils.data import Dataset
import random
from .scalers import CustomStandardScaler, LogStandardScaler
import fuzzytools.strings as strings
from joblib import Parallel, delayed
from fuzzytools.lists import get_list_chunks, get_random_item
from fuzzytools.progress_bars import ProgressBar
from fuzzytorch.utils import print_tdict
import fuzzytorch.models.seq_utils as seq_utils
from fuzzytorch.utils import TDictHolder
from lchandler.lc_classes import diff_vector
from nested_dict import nested_dict
from copy import copy, deepcopy

###################################################################################################################################################

def fix_new_len(tdict, uses_len_clip, max_len):
	new_tdict = {}
	for key in tdict.keys():
		#print('key',key)
		x = tdict[key]
		is_seq_tensor = len(x.shape)==2
		if uses_len_clip and is_seq_tensor:
			new_tdict[key] = seq_utils.get_seq_clipped_shape(x, max_len)
		else:
			new_tdict[key] = x
	return new_tdict

###################################################################################################################################################

class CustomDataset(Dataset):
	def __init__(self, lcset_name, lcset, device,
		in_attrs=None,
		rec_attr=None,
		max_day:float=np.infty,
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		std_scale:float=C_.OBSE_STD_SCALE,
		cpds_p:float=C_.CPDS_P,
		balanced_repeats=1,
		precomputed_copies=1,
		uses_precomputed_copies=True,
		uses_daugm=False,
		uses_dynamic_balance=False,
		ds_mode={},
		):
		self.lcset_name = lcset_name
		self.lcset = lcset
		self.device = device

		self.in_attrs = in_attrs
		self.rec_attr = rec_attr
		self.max_day = max_day
		self.hours_noise_amp = hours_noise_amp
		self.std_scale = std_scale
		self.cpds_p = cpds_p
		self.balanced_repeats = balanced_repeats
		self.precomputed_copies = precomputed_copies
		self.uses_precomputed_copies = uses_precomputed_copies
		self.uses_daugm = uses_daugm
		self.uses_dynamic_balance = uses_dynamic_balance
		self.ds_mode = ds_mode
		self.reset()

	def reset(self):
		torch.cuda.empty_cache()
		self.generate_serial()
		self.lcset_info = self.lcset.get_info()
		self.band_names = self.lcset.band_names
		self.class_names = self.lcset.class_names
		self.survey = self.lcset.survey

		self.append_in_ddays = 'd_days' in self.in_attrs
		self.in_attrs = [ia for ia in self.in_attrs if not ia=='d_days']
		self._max_day = self.max_day # save original to perform reset
		self.max_len = self.calcule_max_len()
		self.automatic_diff()

		self.scalers = nested_dict()
		self.calcule_dtime_scaler()
		self.calcule_in_scaler()
		self.calcule_rec_scaler()
		self.scalers = self.scalers.to_dict()

		self.calcule_poblation_weights()
		self.calcule_balanced_w_cdict()

	def calcule_precomputed(self):
		self.precomputed_dict = {}
		for lcobj_name in self.lcset.get_lcobj_names():
			self.precomputed_dict[lcobj_name] = []
			for k in range(0, self.precomputed_copies):
				in_tdict = self.get_item(lcobj_name)
				self.precomputed_dict[lcobj_name] += [TDictHolder(in_tdict).to(self.device)]

	def generate_serial(self):
		# extra band used to statistics as scales
		for lcobj_name in self.lcset.get_lcobj_names():
			lcobj = self.lcset[lcobj_name]
			lcobj.add_sublcobj_b('*', sum([lcobj.get_b(b) for b in self.lcset.band_names]))
			lcobj.reset_day_offset_serial() # remove day offset!

###################################################################################################################################################

	def automatic_diff(self):
		# used to statistics as scales
		attrs = self.in_attrs+[self.rec_attr]
		for attr in attrs: # calcule derivates!
			if attr=='d_obs':
				self.lcset.set_diff_parallel('obs')
			if attr=='d_obse':
				self.lcset.set_diff_parallel('obse')

		### almost needed always for RNN
		self.lcset.set_diff_parallel('days')

	def reset_max_day(self):
		self.max_day = self._max_day

	def calcule_balanced_w_cdict(self):
		self.balanced_w_cdict = self.lcset.get_class_balanced_weights_cdict()

	def calcule_poblation_weights(self):
		self.populations_cdict = self.lcset.get_populations_cdict()

	def resample_lcobj_names(self):
		min_index = np.argmin([self.populations_cdict[c] for c in self.class_names])
		min_c = self.class_names[min_index]
		balanced_lcobj_names = self.lcset.get_lcobj_names(min_c)*self.balanced_repeats
		boostrap_n = len(balanced_lcobj_names)
		for c in self.class_names:
			if c==min_c:
				continue
			lcobj_names_c = self.lcset.get_boostrap_samples(c, boostrap_n)
			balanced_lcobj_names += lcobj_names_c
		return balanced_lcobj_names

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
			'balanced_w_cdict':self.balanced_w_cdict,
			'populations_cdict':self.populations_cdict,
			}, ', ', '=')
		txt += ')'
		return txt

	def calcule_max_len(self):
		lens = []
		for lcobj_name in self.get_lcobj_names():
			lcobjb = copy(self.lcset[lcobj_name].get_b('*')) # copy
			lcobjb.clip_attrs_given_max_day(self.max_day)
			lens += [len(lcobjb)]
		return max(lens)

	def get_max_len(self):
		return self.max_len

	def set_max_day(self, max_day):
		assert max_day<=self._max_day
		self.max_day = max_day

	def set_scalers_from(self, other):
		self.set_scalers(other.get_scalers())
		return self

	def get_scalers(self):
		return self.scalers

	def set_scalers(self, scalers):
		self.scalers = deepcopy(scalers)

###################################################################################################################################################

	def calcule_dtime_scaler(self):
		SCALER_CLASS = CustomStandardScaler
		#SCALER_CLASS = LogStandardScaler
		#SCALER_CLASS = LogQuantileTransformer # slow
		for kb,b in enumerate(self.band_names+['*']):
			values = self.lcset.get_lcset_values_b(b, 'd_days')[...,None]
			scaler = SCALER_CLASS()
			scaler.fit(values)
			self.scalers['dtime'][b] = scaler

	def calcule_in_scaler(self):
		#SCALER_CLASS = CustomStandardScaler
		SCALER_CLASS = LogStandardScaler
		#SCALER_CLASS = LogQuantileTransformer # slow
		for kb,b in enumerate(self.band_names+['*']):
			values = np.concatenate([self.lcset.get_lcset_values_b(b, in_attr)[...,None] for ka,in_attr in enumerate(self.in_attrs)], axis=-1)
			scaler = SCALER_CLASS()
			scaler.fit(values)
			self.scalers['in'][b] = scaler

	def calcule_rec_scaler(self):
		#SCALER_CLASS = CustomStandardScaler
		SCALER_CLASS = LogStandardScaler
		#SCALER_CLASS = LogQuantileTransformer # slow
		for kb,b in enumerate(self.band_names+['*']):
			values = self.lcset.get_lcset_values_b(b, self.rec_attr)[...,None]
			scaler = SCALER_CLASS()
			scaler.fit(values)
			self.scalers['rec'][b] = scaler

###################################################################################################################################################

	def dtime_normalize(self, x, b):
		'''
		x (t,1)
		'''
		if len(x)==0:
			return x
		assert len(x.shape)==2
		assert x.shape[-1]==1
		return self.scalers['dtime'][b].transform(x)

	def in_normalize(self, x, b):
		'''
		x (t,f)
		'''
		if len(x)==0:
			return x
		assert len(x.shape)==2
		assert x.shape[-1]==len(self.in_attrs)
		return self.scalers['in'][b].transform(x)
	
	def rec_normalize(self, x, b):
		'''
		x (t,1)
		'''
		if len(x)==0:
			return x
		assert len(x.shape)==2
		assert x.shape[-1]==1
		return self.scalers['rec'][b].transform(x)

	def get_rec_inverse_transform(self, model_rec_x_b, b):
		'''
		x (t)
		'''
		assert len(model_rec_x_b.shape)==1
		return self.scalers['rec'][b].inverse_transform(model_rec_x_b[...,None])[...,0]

	###################################################################################################################################################

	def get_random_stratified_lcobj_names(self,
		nc=1,
		):
		# stratified, mostly used for images in experiments
		lcobj_names = []
		random_ndict = self.lcset.get_random_stratified_lcobj_names(nc)
		for c in self.class_names:
			lcobj_names += random_ndict[c]
		return lcobj_names

	###################################################################################################################################################

	def get_lcobj_names(self):
		return self.lcset.get_lcobj_names()

	def get_train_lcobj_names(self):
		if self.uses_dynamic_balance:
			return self.balanced_lcobj_names
		else:
			return self.get_lcobj_names()

	def pre_training_step(self):
		if self.uses_dynamic_balance:
			self.balanced_lcobj_names = self.resample_lcobj_names() # important

	def __len__(self):
		self.pre_training_step()
		lcobj_names = self.get_train_lcobj_names()
		return len(lcobj_names)

	###################################################################################################################################################

	def __getitem__(self, idx:int):
		lcobj_names = self.get_train_lcobj_names()
		lcobj_name = lcobj_names[idx]
		if self.uses_precomputed_copies:
			item = get_random_item(self.precomputed_dict[lcobj_name])
			return item
		else:
			item = self.get_item(lcobj_name) # cpu
			return item

	def get_item(self, lcobj_name,
		uses_len_clip=True,
		return_lcobjs=False,
		):
		'''
		apply data augmentation, this overrides obj information
		be sure to copy the input lcobj!!!!
		'''
		lcobj = copy(self.lcset[lcobj_name]) # copy
		### perform da ignoring *
		if self.uses_daugm:
			for kb,b in enumerate(self.band_names):
				lcobjb = lcobj.get_b(b)
				lcobjb.apply_downsampling_window(self.ds_mode, .1) # curve points downsampling we need to ensure the model to see compelte curves
				lcobjb.add_obs_noise_gaussian(0, self.std_scale) # add obs noise
			lcobj.reset_day_offset_serial(bands=self.band_names) # remove day offset!

		### clip by max day
		for kb,b in enumerate(self.band_names):
			lcobjb = lcobj.get_b(b)
			lcobjb.clip_attrs_given_max_day(self.max_day)

		### recompute *
		lcobj.add_sublcobj_b('*', sum([lcobj.get_b(b) for b in self.band_names])) # redefine serial band because da

		###
		tdict = nested_dict()
		s_onehot = lcobj.get_onehot_serial(bands=self.band_names) # ignoring *
		#print(s_onehot.shape, s_onehot)
		tdict['input'][f's_onehot'] = torch.as_tensor(s_onehot) # (t,b)
		for kb,b in enumerate(self.band_names+['*']):
			lcobjb = lcobj.get_b(b)
			lcobjb.set_diff('days') # recompute dtime just in case (it's already implemented in da)

			onehot = np.ones((len(lcobjb),), dtype=bool)[...,None] # (t,1)
			rx = lcobjb.get_custom_x(self.in_attrs) # raw_x (t,f)
			x = self.in_normalize(rx, b) # norm_x (t,f)
			rtime = lcobjb.days[...,None] # raw_time t,1) - timeselfattn
			#time
			rdtime = lcobjb.d_days[...,None] # raw_dtime (t,1) - gru-d
			dtime = self.dtime_normalize(rdtime, b) # norm_dtime (t,1) - rnn/tcnn

			x = np.concatenate([x, dtime], axis=-1) if self.append_in_ddays else x # new x

			tdict['input'][f'onehot.{b}'] = torch.as_tensor(onehot)
			tdict['input'][f'rtime.{b}'] = torch.as_tensor(rtime)
			#time
			tdict['input'][f'rdtime.{b}'] = torch.as_tensor(rdtime)
			tdict['input'][f'dtime.{b}'] = torch.as_tensor(dtime)
			tdict['input'][f'x.{b}'] = torch.as_tensor(x, dtype=torch.float32) # fixme

			rrecx = lcobjb.get_custom_x([self.rec_attr]) # raw_recx (t,1)
			recx = self.rec_normalize(rrecx, b) # norm_recx (t,1)
			rerror = lcobjb.obse[...,None] # raw_error (t,1)
			assert np.all(rerror>=0)

			tdict['target'][f'recx.{b}'] = torch.as_tensor(recx)
			tdict['target'][f'rerror.{b}'] = torch.as_tensor(rerror)

		tdict['target']['y'] = torch.LongTensor([lcobj.y])[0] # ()
		tdict['target'][f'balanced_w'] = torch.Tensor([self.balanced_w_cdict[self.class_names[lcobj.y]]])[0] # ()

		tdict = tdict.to_dict()
		tdict = {k:fix_new_len(tdict[k], uses_len_clip, self.max_len) for k in tdict.keys()}
		if return_lcobjs:
			return tdict, lcobj
		return tdict