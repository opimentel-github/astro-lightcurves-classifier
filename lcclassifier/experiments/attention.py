from __future__ import print_function
from __future__ import division
from . import C_

import torch
from fuzzytorch.utils import get_model_name, TDictHolder
from fuzzytorch.models.utils import count_parameters
import numpy as np
from flamingchoripan.progress_bars import ProgressBar, ProgressBarMulti
import flamingchoripan.files as files
import flamingchoripan.datascience.metrics as fcm
from flamingchoripan.cuteplots.utils import save_fig
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

def attn_scores_m(train_handler, data_loader,
	m:int=2,
	figsize:tuple=C_.DEFAULT_FIGSIZE_BOX,
	nc:int=1,
	save_rootdir:str='results',
	experiment_id:int=0,
	**kwargs):
	results = []
	for experiment_id in range(m):
		r = attn_scores(train_handler, data_loader,
			figsize,
			nc,
			save_rootdir,
			experiment_id,
			**kwargs)
		results.append(r)

	return results

def attn_scores(train_handler, data_loader,
	figsize:tuple=C_.DEFAULT_FIGSIZE_BOX,
	nc:int=1,
	save_rootdir:str='results',
	experiment_id:int=0,
	eps=C_.EPS,
	alpha=0.333,
	**kwargs):
	### dataloader and extract dataset - important
	train_handler.load_model() # important, refresh to best model
	train_handler.model.eval() # important, model eval mode
	data_loader.eval() # set mode
	dataset = data_loader.dataset # get dataset
	
	with torch.no_grad():
		lcobj_names = dataset.get_random_stratified_lcobj_names(nc)
		fig, axs = plt.subplots(len(lcobj_names), 1, figsize=figsize)

		for k,lcobj_name in enumerate(lcobj_names):
			ax = axs[k]
			tdict, lcobj = dataset.get_item(lcobj_name, return_lcobjs=True)
			train_handler.model.autoencoder['encoder'].return_scores = True
			out_tdict = train_handler.model(TDictHolder(tdict).to(train_handler.device, add_dummy_dim=True))
			train_handler.model.autoencoder['encoder'].return_scores = False
			onehot = out_tdict['input']['onehot']
			t = onehot.shape[1]

			uses_attn = 'layer_scores' in out_tdict['model'].keys()
			is_parallel = 'Parallel' in train_handler.complete_save_roodir
			if not uses_attn or not is_parallel:
				plt.close(fig)
				return

			for kb,b in enumerate(dataset.band_names):
				b_len = onehot[...,kb].sum()
				dummy_p_onehot = seq_utils.get_seq_onehot_mask(onehot[...,kb].sum(dim=-1), t)
				lcobjb = lcobj.get_b(b)
				plot_lightcurve(ax, lcobj, b, label=f'{b} observation', max_day=dataset.max_day)

				### attn scores
				raw_attn_scores = out_tdict['model']['layer_scores'][b][:,-1,...] # (b,layers,h,t,qt) > (b,h,t,qt)
				raw_attn_scores = raw_attn_scores.mean(dim=1) # (b,h,t,qt) > (b,t,qt)
				raw_attn_scores = seq_utils.seq_last_element(raw_attn_scores, dummy_p_onehot)[...,None] # get elements from last step (b,t,q) > (b,qt,1)
				attn_scores = seq_utils.seq_avg_norm(raw_attn_scores, dummy_p_onehot) # (b,qt,1)
				attn_scores_min_max = seq_utils.seq_min_max_norm(raw_attn_scores, dummy_p_onehot) # (b,qt,1)
				attn_scores = attn_scores.cpu().numpy()
				attn_scores_min_max = attn_scores_min_max.cpu().numpy()
				attn_entropy = -np.sum(attn_scores*np.log(attn_scores+eps))

				days = out_tdict['input']['time'][0,onehot[0,:,kb]].cpu().numpy()

				for ki,i in enumerate(range(b_len)):
					markersize = attn_scores_min_max[0,i,0]*25
					ax.plot(days[i], lcobjb.obs[i], 'o', markersize=markersize, markeredgewidth=0, c=C_lchandler.COLOR_DICT[b], alpha=alpha)
				ax.plot([None], [None], 'o', markeredgewidth=0, c=C_lchandler.COLOR_DICT[b], label=f'{b} attention score', alpha=alpha)

			title = f'survey: {dataset.survey} - set: {dataset.lcset_name} - lcobj: {lcobj_names[k]} - class: {dataset.class_names[lcobj.y]}'
			ax.set_title(title)
			ax.set_ylabel('flux')
			ax.legend(loc='upper right')
			ax.grid(alpha=0.5)

		ax.set_xlabel('days')
		fig.tight_layout()

	### save file
	complete_save_roodir = train_handler.complete_save_roodir.split('/')[-1] # train_handler.get_complete_save_roodir().split('/')[-1]
	image_save_dir = f'{save_rootdir}/{complete_save_roodir}'
	image_save_filedir = f'{image_save_dir}/exp_id={experiment_id}°id={train_handler.id}°set={dataset.lcset_name}.attn.png'
	#prints.print_green(f'> saving: {image_save_filedir}')
	save_fig(image_save_filedir, fig)
	return image_save_filedir

###################################################################################################################################################

def attention_statistics(train_handler, data_loader,
	figsize:tuple=C_.DEFAULT_FIGSIZE_REC,
	save_rootdir:str='results',
	save_fext:str='attnscores',
	days_N:int=C_.DEFAULT_DAYS_N,
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
				attn_scores = seq_utils.seq_avg_norm(raw_attn_scores, dummy_p_onehot) # (b,qt,1)
				attn_scores_min_max = seq_utils.seq_min_max_norm(raw_attn_scores, dummy_p_onehot) # (b,qt,1)
				attn_scores = attn_scores.cpu().numpy()
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
	filedir = f'{file_save_dir}/id={train_handler.id}°set={dataset.lcset_name}.{save_fext}'
	files.save_pickle(filedir, results) # save file
	return