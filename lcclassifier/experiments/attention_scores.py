from __future__ import print_function
from __future__ import division
from . import C_

import torch
from fuzzytorch.utils import get_model_name, TDictHolder, tensor_to_numpy
import numpy as np
import flamingchoripan.files as files
from flamingchoripan.cuteplots.utils import save_fig
from flamingchoripan.cuteplots.animations import PlotAnimation
import matplotlib.pyplot as plt
import fuzzytorch.models.seq_utils as seq_utils
from lchandler import C_ as C_lchandler
from lchandler.plots.lc import plot_lightcurve
import random

###################################################################################################################################################

def save_attn_scores_animation(train_handler, data_loader, save_rootdir,
	m:int=2,
	figsize:tuple=C_.DEFAULT_FIGSIZE_BOX,
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
	figsize:tuple=C_.DEFAULT_FIGSIZE_BOX,
	nc:int=1,
	alpha=0.333,
	days_n:int=C_.DEFAULT_DAYS_N_AN,
	animation_duration=10,
	**kwargs):
	### dataloader and extract dataset - important
	train_handler.load_model() # important, refresh to best model
	train_handler.model.eval() # important, model eval mode
	data_loader.eval() # set mode
	dataset = data_loader.dataset # get dataset
	
	animation = PlotAnimation(animation_duration, save_end_frame=True)
	days = np.linspace(C_.DEFAULT_MIN_DAY, dataset.max_day, days_n)#[::-1]
	with torch.no_grad():
		lcobj_names = dataset.get_random_stratified_lcobj_names(nc)
		xlims = {lcobj_name:None for lcobj_name in lcobj_names}
		ylims = {lcobj_name:None for lcobj_name in lcobj_names}
		for day in days[::-1]: # along days
			dataset.set_max_day(day)
			fig, axs = plt.subplots(len(lcobj_names), 1, figsize=figsize)
			for k,lcobj_name in enumerate(lcobj_names):
				ax = axs[k]
				tdict, lcobj = dataset.get_item(lcobj_name, return_lcobjs=True)
				train_handler.model.autoencoder['encoder'].add_extra_return = True
				out_tdict = train_handler.model(TDictHolder(tdict).to(train_handler.device, add_dummy_dim=True))
				train_handler.model.autoencoder['encoder'].add_extra_return = False
				onehot = out_tdict['input']['onehot']
				t = onehot.shape[1]

				#print(out_tdict['model'].keys())
				uses_attn = 'attn_scores' in out_tdict['model'].keys()
				is_parallel = 'Parallel' in train_handler.model.get_name()
				if not all([uses_attn, is_parallel]):
					plt.close(fig)
					dataset.reset_max_day() # very important!!
					return

				for kb,b in enumerate(dataset.band_names):
					lcobjb = lcobj.get_b(b)
					b_len = onehot[...,kb].sum()
					dummy_p_onehot = seq_utils.get_seq_onehot_mask(onehot[...,kb].sum(dim=-1), t)
					plot_lightcurve(ax, lcobj, b, label=f'{b} obs', max_day=day)
					if kb==0:
						threshold_day = min([day, max([lcobj.get_b(_b).days[-1] for _b in dataset.band_names])])
						ax.axvline(threshold_day, linestyle='--', c='k', label=f'threshold day')

					### attn scores
					attn_layers = train_handler.model.autoencoder['encoder'].attn_layers
					attn_scores = out_tdict['model']['attn_scores'][f'z-{attn_layers-1}.{b}'] # (b,h,qt)
					attn_scores = attn_scores.mean(dim=1)[...,None] # (b,h,qt) > (b,qt,1)
					#print('attn_scores',attn_scores.shape)
					attn_scores_min_max = tensor_to_numpy(seq_utils.seq_min_max_norm(attn_scores, dummy_p_onehot)) # (b,qt,1)

					c = C_lchandler.COLOR_DICT[b]
					for i in range(0, b_len):
						markersize = attn_scores_min_max[0,i,0]*25
						ax.plot(lcobjb.days[i], lcobjb.obs[i], 'o', markersize=markersize, markeredgewidth=0, c=c, alpha=alpha)
					ax.plot([None], [None], 'o', markeredgewidth=0, c=c, label=f'{b} attention scores', alpha=alpha)

				title = ''
				title += f'model attention scores mapping'+'\n' if k==0 else ''
				title += f'survey={dataset.survey}-{"".join(dataset.band_names)} [{dataset.lcset_name}] - lcobj={lcobj_names[k]} [{dataset.class_names[lcobj.y]}]'+'\n'
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
			animation.append(fig)

	### save file
	image_save_filedir = f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}~exp_id={experiment_id}.gif'
	animation.save(image_save_filedir, reverse=True)
	dataset.reset_max_day() # very important!!
	return