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

def get_binned_obs(times, obs, dt):
	assert len(times)==len(obs)
	if len(times)==0: # all is missing
		return np.array([0]), np.array([0]).astype(np.bool)
	if len(times)==1:
		return np.array([obs[0]]), np.array([1]).astype(np.bool)

	assert times[-1]>times[0]
	binned_obs = []
	missing_mask = []

	t = times[0]
	time_mesh = [t]
	while t<times[-1]:
		t += dt
		time_mesh.append(t)
	time_mesh = np.array(time_mesh)

	last_obs = obs[0]
	for k,it in enumerate(time_mesh[:-1]):
		ft = time_mesh[k+1]
		idxs = np.where((times>=it) & (times<=ft))[0]
		#print(it, ft, idxs)
		if len(idxs)>0: # exist elemtns
			last_obs = obs[idxs].mean()
			missing_mask.append(1)
		else: # no elements in this range
			missing_mask.append(0)

		binned_obs.append(last_obs)
		
	#print(time_mesh)
	return np.array(binned_obs), np.array(missing_mask).astype(np.bool)

###################################################################################################################################################

class CustomDataset(Dataset):
	def __init__(self, lcdataset, lcset_name,
		dt=2,
		effective_beta_eps=0.00001, # same weight -> 0.01 0.001 0.0001 0.00001 -> 1/freq
		):
		self.lcset = lcdataset[lcset_name]
		self.lcset_name = lcset_name
		self.dt = dt
		self.effective_beta_eps = effective_beta_eps

		self.max_day = None
		self.max_len = self.calcule_max_len()

		self.band_names = self.lcset.band_names
		self.class_names = self.lcset.class_names
		self.survey = self.lcset.survey
		self.reset()

	def reset(self):
		self.uses_precomputed_samples = True
		self.automatic_diff_log()
		self.get_norm_bdict()
		self.calculate_poblation_weights()
		self.precomputed_tdict = []

	def calcule_max_len(self):
		max_len = max([len(self.lcset[lcobj_name].get_custom_x_serial(['days'], max_day=self.max_day)) for lcobj_name in self.lcset.get_lcobj_names()])
		return max_len

	def calculate_poblation_weights(self):
		self.poblation_weights = self.lcset.get_class_effective_weigths_cdict(1-self.effective_beta_eps) # get_class_freq_weights_cdict get_class_effective_weigths_cdict

	def get_poblation_weights(self):
		return self.poblation_weights

	def automatic_diff_log(self):
		self.lcset.set_log_parallel('obs')

	def get_output_dims(self):
		return 0

	def __repr__(self):
		txt = f'CustomDataset('
		txt += strings.get_string_from_dict({
			'lcset_len':f'{len(self.lcset):,}',
			'max_len':f'{self.max_len:,}',
			'poblation_weights':self.poblation_weights,
			}, ', ', '=')
		txt += ')'
		return txt
	
	def get_norm_bdict(self):
		self.norm_bdict = {}
		for kb,b in enumerate(self.band_names):
			values = np.array(self.lcset.get_lcset_values_b(b, 'log_obs'))[...,None]
			qt = StandardScaler()
			#qt = QuantileTransformer(n_quantiles=10000, random_state=0, output_distribution='normal') # slow
			qt.fit(values)
			self.norm_bdict[b] = qt
		return self.norm_bdict

	def set_norm_bdict(self, norm_bdict):
		self.norm_bdict = {k:norm_bdict[k] for k in norm_bdict.keys()}

	def transfer_to(self, other):
		other.set_norm_bdict(self.get_norm_bdict())
		pass
	
	###################################################################################################################################################

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
		### get and copy object
		lcobj = self.lcset[lcobj_name].copy() # important to copy!!!!

		### remove day offset!
		day_offset = lcobj.reset_day_offset_serial()

		model_input = {}
		for kb,b in enumerate(self.band_names):
			lcobjb = lcobj.get_b(b)
			qt = self.norm_bdict[b]
			time = lcobjb.days
			log_obs = lcobjb.log_obs

			binned_obs, missing_mask  = get_binned_obs(time, log_obs, self.dt)
			binned_obs = qt.transform(binned_obs[...,None])
			#binned_obs = binned_obs[...,None]
			missing_mask = missing_mask[...,None]
			onehot = np.ones_like(binned_obs).astype(np.bool)

			### build output
			max_len = self.max_len if uses_len_clip else None
			model_input[f'onehot.{b}'] = get_seq_clipped_shape(torch.as_tensor(onehot), max_len)
			model_input[f'binned_obs.{b}'] = get_seq_clipped_shape(torch.as_tensor(binned_obs, dtype=torch.float32), max_len)
			model_input[f'missing_mask.{b}'] = get_seq_clipped_shape(torch.as_tensor(missing_mask), max_len)

		target = {
			'y':torch.as_tensor(lcobj.y),
			'poblation_weights':torch.as_tensor([self.poblation_weights[c] for c in self.class_names], dtype=torch.float32),
		}
		tdict = {
			'input':model_input,
			'target':target,
			}

		#print_tdict(tdict)
		if return_lcobjs:
			return tdict, lcobj
		return tdict