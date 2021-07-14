from __future__ import print_function
from __future__ import division
from . import C_

import torch
from fuzzytorch.utils import TDictHolder, tensor_to_numpy, minibatch_dict_collate
import numpy as np
import fuzzytools.files as files
from fuzzytools.cuteplots.utils import save_fig
from fuzzytools.cuteplots.animators import PlotAnimator
import matplotlib.pyplot as plt
import fuzzytorch.models.seq_utils as seq_utils
from lchandler import C_ as C_lchandler
from lchandler.plots.lc import plot_lightcurve
import random

FIGSIZE = (10,10)

###################################################################################################################################################

def save_attn_scores_animation(train_handler, data_loader, save_rootdir,
	m:int=2,
	figsize:tuple=FIGSIZE,
	nc:int=1,
	**kwargs):
	results = []
	for experiment_id in range(0, m):
		random.seed(experiment_id)
		np.random.seed(experiment_id)
		r = _save_attn_scores_animation(train_handler, data_loader, save_rootdir, str(experiment_id),
			figsize,
			nc,
			**kwargs)
		results.append(r)
	return results

def _save_attn_scores_animation(train_handler, data_loader, save_rootdir, experiment_id,
	figsize:tuple=FIGSIZE,
	nc:int=1,
	alpha=0.333,
	days_n:int=C_.DEFAULT_DAYS_N_AN,
	animation_duration=10,
	**kwargs):
	train_handler.load_model() # important, refresh to best model
	train_handler.model.eval() # important, model eval mode
	dataset = data_loader.dataset # get dataset
	
	is_parallel = 'Parallel' in train_handler.model.get_name()
	if not is_parallel:
		return

	plot_animator = PlotAnimator(animation_duration, save_end_frame=True)
	days = np.linspace(C_.DEFAULT_MIN_DAY, dataset.max_day, days_n)#[::-1]
	with torch.no_grad():
		lcobj_names = dataset.get_random_stratified_lcobj_names(nc)
		xlims = {lcobj_name:None for lcobj_name in lcobj_names}
		ylims = {lcobj_name:None for lcobj_name in lcobj_names}
		for day in days[::-1]: # along days
			dataset.set_max_day(day)
			dataset.calcule_precomputed()

			fig, axs = plt.subplots(len(lcobj_names), 1, figsize=figsize)
			for k,lcobj_name in enumerate(lcobj_names):
				ax = axs[k]
				in_tdict, lcobj = dataset.get_item(lcobj_name)
				in_tdict = dataset.fix_tdict(in_tdict)
				train_handler.model.autoencoder['encoder'].add_extra_return = True
				tdict = train_handler.model(TDictHolder(in_tdict).to(train_handler.device, add_dummy_dim=True))
				train_handler.model.autoencoder['encoder'].add_extra_return = False

				#print(tdict['model'].keys())
				uses_attn = any([f'attn_scores' in k for k in tdict.keys()])
				if not uses_attn:
					plt.close(fig)
					dataset.reset_max_day() # very important!!
					dataset.calcule_precomputed()
					return

				for kb,b in enumerate(dataset.band_names):
					lcobjb = lcobj.get_b(b)
					plot_lightcurve(ax, lcobj, b, label=f'{b} obs', max_day=day)
					if kb==0:
						lcobj_max_day = max([lcobj.get_b(_b).days[-1] for _b in dataset.band_names if len(lcobj.get_b(_b))>0])
						threshold_day = min([day, lcobj_max_day])
						ax.axvline(threshold_day, linestyle='--', c='k', label=f'threshold day {threshold_day:.3f}')

					### attn scores
					p_onehot = tdict[f'input/onehot.{b}'][...,0] # (b,t)
					attn_scores = tdict[f'model/attn_scores/encz.{b}'] # (b,h,qt)
					attn_scores = attn_scores.mean(dim=1)[...,None] # (b,h,qt)>(b,qt,1) # mean along heads 
					#print('attn_scores',attn_scores.shape)
					attn_scores_min_max = tensor_to_numpy(seq_utils.seq_min_max_norm(attn_scores, p_onehot)) # (b,qt,1)
					
					b_len = p_onehot.sum().item()
					assert b_len<=len(lcobjb), f'{b_len}=={len(lcobjb)}'
					for i in range(0, b_len):
						c = C_lchandler.COLOR_DICT[b]
						min_makersize = 12
						max_makersize = 30
						p = attn_scores_min_max[0,i,0]
						markersize = max_makersize*p+min_makersize*(1-p)
						ax.plot(lcobjb.days[i], lcobjb.obs[i], 'o', markersize=markersize, markeredgewidth=0, c=c, alpha=alpha)
					ax.plot([None], [None], 'o', markeredgewidth=0, c=c, label=f'{b} attention scores', alpha=alpha)

				title = ''
				if k==0:
					title += f'model attention scores mapping'+'\n'
					title += f'set={dataset.survey} [{dataset.lcset_name.replace(".@", "")}]'+'\n'
				title += '$\\bf{'+ALPHABET[k]+'}$'+f' lcobj={lcobj_names[k]} [{dataset.class_names[lcobj.y]}]'+'\n'
				ax.set_title(title[:-1])
				ax.set_ylabel('observations [flux]')
				ax.legend(loc='upper right')
				ax.grid(alpha=0.5)
				xlims[lcobj_name] = ax.get_xlim() if xlims[lcobj_name] is None else xlims[lcobj_name]
				ylims[lcobj_name] = ax.get_ylim() if ylims[lcobj_name] is None else ylims[lcobj_name]
				ax.set_xlim(xlims[lcobj_name])
				ax.set_ylim(ylims[lcobj_name])

			ax.set_xlabel('time [days]')
			fig.tight_layout()
			plot_animator.append(fig)

	### save file
	image_save_filedir = f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}~exp_id={experiment_id}.mp4' # gif mp4
	plot_animator.save(image_save_filedir, reverse=True)
	dataset.reset_max_day() # very important!!
	dataset.calcule_precomputed()
	return