from __future__ import print_function
from __future__ import division
from . import C_

import torch
from fuzzytorch.utils import get_model_name, TDictHolder
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

def lineal_trend_f(days, m, n):
	return days*m+n

def min_max_norm(x, i,
	eps:float=C_.EPS,
	):
	min_ = x.min()
	max_ = x.max()
	diff_ = max_-min_
	return (x[i]-min_)/(diff_+eps)

###################################################################################################################################################

def save_attention_statistics(train_handler, data_loader,
	figsize:tuple=C_.DEFAULT_FIGSIZE_REC,
	save_rootdir:str='results',
	eps:float=C_.EPS,
	di=3,
	**kwargs):
	### dataloader and extract dataset - important
	train_handler.load_model() # important, refresh to best model
	train_handler.model.eval() # model eval
	data_loader.eval() # set mode
	dataset = data_loader.dataset # get dataset
	dataset.reset_max_day() # always reset max day
	dataset.uses_precomputed_samples = False

	attn_scores_collection = []
	with torch.no_grad():
		lcobj_names = dataset.get_lcobj_names()

		for k,lcobj_name in enumerate(lcobj_names):
			tdict, lcobj = dataset.get_item(lcobj_name, return_lcobjs=True)
			train_handler.model.autoencoder['encoder'].return_scores = True
			out_tdict = train_handler.model(TDictHolder(tdict).to(train_handler.device, add_dummy_dim=True))
			train_handler.model.autoencoder['encoder'].return_scores = False
			onehot = out_tdict['input']['onehot']
			t = onehot.shape[1]

			uses_attn = 'layer_scores' in out_tdict['model'].keys()
			is_parallel = 'Parallel' in train_handler.complete_save_roodir
			if not uses_attn or not is_parallel:
				return

			for kb,b in enumerate(dataset.band_names):
				b_len = onehot[...,kb].sum().item()
				if b_len<10:
					continue
				dummy_p_onehot = seq_utils.get_seq_onehot_mask(onehot[...,kb].sum(dim=-1), t)
				lcobjb = lcobj.get_b(b)

				### attn scores
				raw_attn_scores = out_tdict['model']['layer_scores'][b][:,-1,...] # (b,layers,h,t,qt) > (b,h,t,qt)
				raw_attn_scores = raw_attn_scores.mean(dim=1) # (b,h,t,qt) > (b,t,qt)
				raw_attn_scores = seq_utils.seq_last_element(raw_attn_scores, dummy_p_onehot)[...,None] # get elements from last step (b,t,q) > (b,qt,1)
				attn_scores = seq_utils.seq_sum_norm(raw_attn_scores, dummy_p_onehot) # (b,qt,1)
				attn_scores = attn_scores.cpu().numpy()
				attn_scores_min_max = seq_utils.seq_min_max_norm(raw_attn_scores, dummy_p_onehot) # (b,qt,1)
				attn_scores_min_max = attn_scores_min_max.cpu().numpy()

				attn_entropy = -np.sum(attn_scores*np.log(attn_scores+eps))

				days = lcobjb.days[:b_len]
				obs = lcobjb.obs[:b_len]
				obse = lcobjb.obse[:b_len]
				wobs = obs/(obse**2+eps)
				peak_day = days[np.argmax(obs)]
				
				for i in range(di, b_len):
					#print(days[i-di:i])
					lineal_trend_days = days[i-di:i]
					lineal_trend_obs = obs[i-di:i]
					popt, pcov = curve_fit(lineal_trend_f, lineal_trend_days, lineal_trend_obs, p0=[0,0])
					lineal_trend = popt[0]

					r = {
						'i':i,
						'b':b,
						'c':dataset.class_names[lcobj.y],
						'b_len':b_len,

						'attn_entropy':attn_entropy,
						'attn_entropy/len':attn_entropy/b_len,
						
						'attn_scores':attn_scores[:,i,:].item(),
						'attn_scores_min_max':attn_scores_min_max[:,i,:].item(),

						'obs':obs[i],
						'obs_min_max':min_max_norm(obs, i),

						'obse':obse[i],
						'obse_min_max':min_max_norm(obse, i),

						'wobs':wobs[i],
						'wobs_min_max':min_max_norm(wobs, i),

						'lineal_trend':lineal_trend,
						
						'peak_day':peak_day,
						'day':days[i],
						'days_from_peak':lineal_trend_days.mean()-peak_day,
					}
					#print(r)
					attn_scores_collection.append(r)

	dataset.uses_precomputed_samples = True  # very important!!
	dataset.reset_max_day() # very important!!

	### more info
	complete_save_roodir = train_handler.complete_save_roodir.split('/')[-1] # train_handler.get_complete_save_roodir().split('/')[-1]
	results = {
		'day':dataset.max_day,
		'attn_scores_collection':attn_scores_collection,

		'complete_save_roodir':complete_save_roodir,
		'model_name':train_handler.model.get_name(),
		'survey':dataset.survey,
		'band_names':dataset.band_names,
		'class_names':dataset.class_names,
		'parameters':count_parameters(train_handler.model),
	}
	for lmonitor in train_handler.lmonitors:
		results[lmonitor.name] = {
			'save_dict':lmonitor.get_save_dict(),
			'best_epoch':lmonitor.get_best_epoch(),
			'time_per_iteration':lmonitor.get_time_per_iteration(),
			#'time_per_epoch_set':{set_name:lmonitor.get_time_per_epoch_set(set_name) for set_name in ['train', 'val']},
			'time_per_epoch':lmonitor.get_time_per_epoch(),
			'total_time':lmonitor.get_total_time(),
		}

	### save file
	#print(results)
	file_save_dir = f'{save_rootdir}/{complete_save_roodir}'
	filedir = f'{file_save_dir}/id={train_handler.id}°set={dataset.lcset_name}.d'
	files.save_pickle(filedir, results) # save file
	dataset.reset_max_day() # very important!!
	return