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
import flamingchoripan.strings as strings
from joblib import Parallel, delayed
from flamingchoripan.lists import get_list_chunks
from flamingchoripan.progress_bars import ProgressBar
from fuzzytorch.utils import print_tdict
from fuzzytorch.models.seq_utils import get_seq_clipped_shape

###################################################################################################################################################

class CustomDataset(Dataset):
	def __init__(self, lcdataset, lcset_name,
		attrs:list=['d_days','obs', 'obse'],
		max_day:float=np.infty,
		max_te_period:float=None,
		max_len:int=None,
		te_features:int=6,
		effective_beta_eps=0.00001, # same weight -> 0.01 0.001 0.0001 0.00001 -> 1/freq

		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		std_scale:float=C_.OBSE_STD_SCALE,
		cpds_p:float=C_.CPDS_P,
		):
		assert te_features%2==0

		self.lcset = lcdataset[lcset_name]
		self.lcset_name = lcset_name
		self.attrs = attrs.copy()
		self.max_day = self.get_max_duration() if max_day is None else max_day
		self.__max_day__ = max_day
		self.max_te_period = self.max_day*2 if max_te_period is None else max_te_period
		self.max_len = self.calcule_max_len() if max_len is None else max_len
		self.te_features = te_features
		self.effective_beta_eps = effective_beta_eps

		self.hours_noise_amp = hours_noise_amp
		self.std_scale = std_scale
		self.cpds_p = cpds_p

		self.band_names = self.lcset.band_names
		self.class_names = self.lcset.class_names
		self.survey = self.lcset.survey
		self.reset()

	def reset(self):
		self.uses_precomputed_samples = True
		self.automatic_diff_log()
		self.get_te_periods()
		self.get_norm_bdict()
		self.reset_max_day()
		self.calculate_poblation_weights()
		self.precomputed_tdict = []

	def calculate_poblation_weights(self):
		self.poblation_weights = self.lcset.get_class_effective_weigths_cdict(1-self.effective_beta_eps) # get_class_freq_weights_cdict get_class_effective_weigths_cdict

	def get_poblation_weights(self):
		return self.poblation_weights

	def reset_max_day(self):
		self.max_day = self.__max_day__

	def automatic_diff_log(self):
		for attr in self.attrs: # calcule the desired attrs
			if attr=='d_days':
				self.lcset.set_diff_parallel('days')
			elif attr=='log_obs':
				self.lcset.set_log_parallel('obs')
			elif attr=='log_obse':
				self.lcset.set_log_parallel('obse')
			else:
				raise Exception(f'no attr: {attr}')

	def get_te_features_dims(self):
		return self.te_features

	def get_output_dims(self):
		return len(self.attrs)

	def __repr__(self):
		txt = f'CustomDataset('
		txt += strings.get_string_from_dict({
			'lcset_len':f'{len(self.lcset):,}',
			'max_day':f'{self.max_day:.2f}',
			'max_len': f'{self.max_len:,}',
			'te_periods':f'{self.te_periods}',
			'attrs':self.attrs,
			'norm_bdict':self.norm_bdict,
			'poblation_weights':self.poblation_weights,
			}, ', ', '=')
		txt += ')'
		return txt

	def calcule_max_len(self):
		max_len = max([len(self.lcset[lcobj_name].get_custom_x_serial(['days'], max_day=self.max_day)) for lcobj_name in self.lcset.get_lcobj_names()])
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
		self.max_duration = max([self.lcset[lcobj_name].get_days_serial_duration() for lcobj_name in self.lcset.get_lcobj_names()])
		return self.max_duration

	def get_te_periods(self):
		self.te_periods = list(np.array([self.max_te_period]*(self.te_features//2)/2**np.arange(self.te_features//2)))
		return self.te_periods

	def set_te_periods(self, te_periods:list):
		self.te_periods = te_periods.copy()
	
	def get_norm_bdict(self):
		self.norm_bdict = {}
		for kb,b in enumerate(self.band_names):
			values = np.concatenate([self.lcset.get_lcset_values_b(b, attr)[...,None] for ka,attr in enumerate(self.attrs)], axis=-1)
			qt = StandardScaler()
			#qt = QuantileTransformer(n_quantiles=10000, random_state=0, output_distribution='normal') # slow
			qt.fit(values)
			self.norm_bdict[b] = qt
		return self.norm_bdict

	def set_norm_bdict(self, norm_bdict):
		self.norm_bdict = {k:norm_bdict[k] for k in norm_bdict.keys()}

	def set_poblation_weights(self, poblation_weights):
		self.poblation_weights = poblation_weights

	def transfer_to(self, other):
		other.set_max_len(self.get_max_len()) # sure?
		other.set_max_day(self.get_max_day())
		other.set_norm_bdict(self.get_norm_bdict())
		other.set_te_periods(self.get_te_periods())
		other.set_poblation_weights(self.get_poblation_weights()) # sure?
	
	def normalize(self, x, onehot):
		new_x = np.zeros_like(x) # starts with zero!!!
		for kb,b in enumerate(self.band_names):
			onehot_b = onehot[...,kb][...,None]
			qt = self.norm_bdict[b]
			new_x += qt.transform(x)*onehot_b
		return new_x
	
	def get_te(self, time):
		assert np.all(~np.isnan(time))
		encoding = np.zeros((len(time), len(self.te_periods)*2))
		for kp,p in enumerate(self.te_periods):
			w = 2*math.pi*(time)/p
			encoding[...,2*kp] = np.sin(w)
			encoding[...,2*kp+1] = np.cos(w)
		return encoding
	
	###################################################################################################################################################

	def get_attr_index(self, attr):
		return self.attrs.index(attr)

	def has_precomputed_samples(self):
		return len(self.precomputed_tdict)>0

	def get_random_stratified_lcobj_names(self,
		nc=2,
		):
		# stratified
		lcobj_names = []
		random_ndict = self.lcset.get_random_stratified_lcobj_names(nc)
		for c in self.class_names:
			lcobj_names += random_ndict[c]
		return lcobj_names

	def __len__(self):
		uses_precomputed_samples = self.has_precomputed_samples() and self.uses_precomputed_samples
		return len(self.precomputed_tdict) if uses_precomputed_samples else len(self.lcset)

	def __getitem__(self, idx:int):
		uses_precomputed_samples = self.has_precomputed_samples() and self.uses_precomputed_samples
		return self.precomputed_tdict[idx] if uses_precomputed_samples else self.get_item(self.lcset.get_lcobj_names()[idx])

	def precompute_samples(self, pre_daugm_copies,
		n_jobs=1,
		chunk_size=None,
		):
		chunk_size = n_jobs if chunk_size is None else chunk_size
		def job(lcobj_name):
			return [self.get_item(lcobj_name, uses_daugm=True) for _ in range(pre_daugm_copies)]
		lcobj_names = self.lcset.get_lcobj_names()
		chunks = get_list_chunks(lcobj_names, chunk_size)
		bar = ProgressBar(len(chunks))
		for kc,chunk in enumerate(chunks):
			bar(f'{self.lcset_name} {chunk}')
			results = Parallel(n_jobs=n_jobs, backend='threading')([delayed(job)(lcobj_name) for lcobj_name in chunk]) # None threading 
			for r in results:
				self.precomputed_tdict += r
		bar.done()

	def get_item(self, lcobj_name,
		uses_daugm=False,
		uses_len_clip=True,
		return_lcobjs=False,
		):
		def fix_new_len(tdict, uses_len_clip, max_len):
			new_tdict = {}
			for key in tdict.keys():
				x = tdict[key]
				shape = list(x.shape)
				assert 0
				get_seq_clipped_shape
				if uses_len_clip and len(shape)==2:
					new_shape = [max_len if k==0 else s for k,s in enumerate(shape)]
					new_x = torch.zeros(new_shape, dtype=x.dtype, device=x.device)
					new_x[:min(new_shape[0], len(x))] = x[:min(new_shape[0], len(x))]
					new_tdict[key] = new_x
				else:
					new_tdict[key] = x
			return new_tdict

		### get and copy object
		lcobj = self.lcset[lcobj_name].copy() # important to copy!!!!

		### apply data augmentation, this overrides obj information
		if uses_daugm:
			for b in lcobj.bands:
				lcobjb = lcobj.get_b(b)
				lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
				lcobjb.add_obs_noise_gaussian(0, self.std_scale) # add obs noise
				lcobjb.apply_downsampling(self.cpds_p) # curve points downsampling

		### remove day offset!
		day_offset = lcobj.reset_day_offset_serial()

		### prepare to export
		max_day = self.max_day
		sorted_time_indexs = lcobj.get_sorted_days_indexs_serial() # get just once for performance purposes
		onehot = lcobj.get_onehot_serial(sorted_time_indexs, max_day)
		x = self.normalize(lcobj.get_custom_x_serial(self.attrs, sorted_time_indexs, max_day), onehot) # norm
		time = lcobj.get_custom_x_serial(['days'], sorted_time_indexs, max_day)
		error = lcobj.get_custom_x_serial(['obse'], sorted_time_indexs, max_day)

		### build output
		model_input = {
			'onehot':torch.as_tensor(onehot),
			'x':torch.as_tensor(x, dtype=torch.float32),
			'time':torch.as_tensor(time, dtype=torch.float32),
			'error':torch.as_tensor(error, dtype=torch.float32),
		}
		if self.te_features>0:
			model_input['te'] = torch.as_tensor(self.get_te(time[:,0]), dtype=torch.float32)
		if 'd_days' in self.attrs:
			model_input['dt'] = torch.as_tensor(x[:,self.get_attr_index('d_days')][:,None], dtype=torch.float32) # max_day ?

		target = {
			'y':torch.as_tensor(lcobj.y),
			'poblation_weights':torch.as_tensor([self.poblation_weights[c] for c in self.class_names], dtype=torch.float32),
		}
		if 'log_obs' in self.attrs:
			target['raw-x'] = torch.as_tensor(x[:,self.get_attr_index('log_obs')][:,None], dtype=torch.float32)

		tdict = {
			'input':fix_new_len(model_input, uses_len_clip, self.max_len),
			'target':fix_new_len(target, uses_len_clip, self.max_len),
			}

		#print_tdict(tdict)
		if return_lcobjs:
			return tdict, lcobj
		return tdict