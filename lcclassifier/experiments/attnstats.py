from __future__ import print_function
from __future__ import division
from . import C_

import torch
from fuzzytorch.utils import TDictHolder, tensor_to_numpy, minibatch_dict_collate
from fuzzytorch.models.utils import count_parameters
import numpy as np
from fuzzytools.progress_bars import ProgressBar, ProgressBarMulti
import fuzzytools.files as files
import fuzzytools.datascience.metrics as fcm
from fuzzytools.cuteplots.utils import save_fig
import matplotlib.pyplot as plt
import fuzzytorch.models.seq_utils as seq_utils
from scipy.optimize import curve_fit
from lchandler import C_ as C_lchandler
from lchandler.plots.lc import plot_lightcurve

###################################################################################################################################################

def get_local_slope(days, obs, j, dj,
	p0=[0,0],
	):
	def local_slope_f(time, m, n):
		return time*m+n
	sub_days = days[max(0, j-dj):j] # (dj)
	sub_obs = obs[max(0, j-dj):j] # (dj)
	popt, pcov = curve_fit(local_slope_f, sub_days, sub_obs, p0=p0)
	local_slope_m, local_slope_n = popt
	return local_slope_m, local_slope_n, sub_days, sub_obs

def min_max_norm(x,
	eps:float=C_.EPS,
	):
	# x = (t)
	_min = x.min()
	_max = x.max()
	_diff = _max-_min
	return (x-_min)/(_diff+eps)

###################################################################################################################################################

def save_attnstats(train_handler, data_loader, save_rootdir,
	eps:float=C_.EPS,
	djs=[2,3],
	**kwargs):
	train_handler.load_model() # important, refresh to best model
	train_handler.model.eval() # important, model eval mode
	dataset = data_loader.dataset # get dataset

	is_parallel = 'Parallel' in train_handler.model.get_name()
	if not is_parallel:
		return

	attn_scores_collection = {b:[] for kb,b in enumerate(dataset.band_names)}
	with torch.no_grad():
		tdicts = []
		for ki,in_tdict in enumerate(data_loader):
			train_handler.model.autoencoder['encoder'].add_extra_return = True
			_tdict = train_handler.model(TDictHolder(in_tdict).to(train_handler.device))
			train_handler.model.autoencoder['encoder'].add_extra_return = False
			tdicts += [_tdict]
		tdict = minibatch_dict_collate(tdicts)

		for kb,b in enumerate(dataset.band_names):
			p_onehot = tdict[f'input/onehot.{b}'][...,0] # (b,t)
			#p_rtime = tdict[f'input/rtime.{b}'][...,0] # (b,t)
			#p_dtime = tdict[f'input/dtime.{b}'][...,0] # (b,t)
			#p_x = tdict[f'input/x.{b}'] # (b,t,f)
			#p_rerror = tdict[f'target/rerror.{b}'] # (b,t,1)
			#p_rx = tdict[f'target/recx.{b}'] # (b,t,1)

			# print(tdict.keys())
			uses_attn = any([f'attn_scores' in k for k in tdict.keys()])
			if not uses_attn:
				return

			### attn scores
			attn_scores = tdict[f'model/attn_scores/encz.{b}'] # (b,h,qt)
			attn_scores = attn_scores.mean(dim=1)[...,None] # (b,h,qt)>(b,qt,1) # mean along heads 
			#print('attn_scores',attn_scores.shape)
			attn_scores_min_max = seq_utils.seq_min_max_norm(attn_scores, p_onehot) # (b,qt,1)

			### stats
			lcobj_names = dataset.get_lcobj_names()
			bar = ProgressBar(len(lcobj_names))
			for k,lcobj_name in enumerate(lcobj_names):
				lcobj = dataset.lcset[lcobj_name]
				lcobjb = lcobj.get_b(b) # complete
				p_onehot_k = tensor_to_numpy(p_onehot[k]) # (b,t) > (t)
				b_len = p_onehot_k.sum()
				assert b_len<=len(lcobjb), f'{b_len}<={len(lcobjb)}'
				bar(f'b={b} - lcobj_name={lcobj_name} - b_len={b_len}')

				if b_len<=min(djs):
					continue

				attn_scores_k = tensor_to_numpy(attn_scores[k,:b_len,0]) # (b,qt,1)>(t)
				attn_scores_min_max_k = tensor_to_numpy(attn_scores_min_max[k,:b_len,0]) # (b,qt,1)>(t)
				attn_entropy = tensor_to_numpy(torch.sum(-attn_scores[k,:b_len,0]*torch.log(attn_scores[k,:b_len,0]+1e-10), dim=0)) # (t)>()
				# print(f'attn_scores_k={attn_scores_k}; attn_entropy={attn_entropy}')

				days = lcobjb.days[:b_len] # (t)
				obs = lcobjb.obs[:b_len] # (t)
				obse = lcobjb.obse[:b_len] # (t)
				snr = lcobjb.get_snr(max_len=b_len)
				peak_day = days[np.argmax(obs)]

				obs_min_max = min_max_norm(obs) # (t)
				obse_min_max = min_max_norm(obse) # (t)
				
				for j in range(min(djs), b_len): # dj,dj+1,...,b_len-1
					r = {
						#'lcobj_name':lcobj_name,
						f'c':dataset.class_names[lcobj.y],
						f'len':b_len,
						f'peak_day':peak_day,
						f'attn_entropy':attn_entropy,
						f'snr':snr,

						f'j':j,
						f'attn_scores_k.j':attn_scores_k[j],
						f'attn_scores_min_max_k.j':attn_scores_min_max_k[j],
						f'days.j':days[j],
						f'obs.j':obs[j],
						f'obs_min_max.j':obs_min_max[j],
						f'obse.j':obse[j],
						f'obse_min_max.j':obse_min_max[j],
						}
					for dj in djs:
						local_slope_m, local_slope_n, sub_days, sub_obs = get_local_slope(days, obs, j, dj)
						r.update({
							f'local_slope_m.j~dj={dj}':local_slope_m,
							f'local_slope_n.j~dj={dj}':local_slope_n,
							f'peak_distance.j~dj={dj}~mode=local':days[j]-peak_day,
							f'peak_distance.j~dj={dj}~mode=mean':np.mean(sub_days)-peak_day,
							f'peak_distance.j~dj={dj}~mode=median':np.median(sub_days)-peak_day,
							})
					attn_scores_collection[b] += [r]

	bar.done()
	results = {
		'model_name':train_handler.model.get_name(),
		'survey':dataset.survey,
		'band_names':dataset.band_names,
		'class_names':dataset.class_names,

		'max_day':dataset.max_day,
		'dj':dj,
		'attn_scores_collection':attn_scores_collection,
	}

	### save file
	save_filedir = f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}.d'
	files.save_pickle(save_filedir, results) # save file
	dataset.reset_max_day() # very important!!
	dataset.calcule_precomputed()
	return