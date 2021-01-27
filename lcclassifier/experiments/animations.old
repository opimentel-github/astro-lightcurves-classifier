from __future__ import print_function
from __future__ import division
from . import C_

import random
import torch
import numpy as np
from flamingchoripan.tinyFlame import utils as tfutils
from . utils import prepare_individual_sample
from astrolightcurveshandler.src import C_ as aC_

from flamingchoripan.cutePlots.animations import PlotAnimation
import matplotlib.pyplot as plt
import astrolightcurveshandler.src.plots as aplots

def predict_tmax_multiple(train_handler, data_loader,
	n:int=2,
	figsize:tuple=(10,10),
	nc:int=1,
	save_rootdir:str='results',
	days_N:int=C_.DEFAULT_DAYS_N,
	animation_time:float=6,
	dummy_animation:bool=False,
	**kwargs):
	for exp_id in range(n):
		predict_tmax(train_handler, data_loader,
		figsize=figsize,
		nc=nc,
		save_rootdir=save_rootdir,
		days_N=days_N,
		animation_time=animation_time,
		dummy_animation=dummy_animation,
		exp_id=exp_id,
		**kwargs)

def predict_tmax(train_handler, data_loader,
	figsize:tuple=(10,10),
	nc:int=1,
	save_rootdir:str='results',
	days_N:int=C_.DEFAULT_DAYS_N,
	animation_time:float=6,
	dummy_animation:bool=False,
	exp_id:int=0,
	**kwargs):
	
	data_loader.performance() # important
	dataset = data_loader.dataset # get dataset
	dataset.reset_max_day() # important
	train_handler.load_model() # importante, refresh to best model

	trainh = train_handler.train_handlers[0]
	days = np.linspace(1, dataset.max_day, days_N)
	idxs, keys = dataset.get_random_idxs_keys(nc)
	#idx = 0
	tmax_target_list = []
	tmax_pred_list = []
	
	fps = days_N/animation_time
	animation = PlotAnimation(len(days), fps, dummy=dummy_animation)
	plot_lims = [{'x':None, 'y':None} for i in range(len(idxs))]
	with torch.no_grad():
		trainh.model.eval() # important, model eval
		can_be_in_loop = True
		for day in days[::-1]:
			try:
				if can_be_in_loop:
					fig, axs = plt.subplots(len(idxs), 1, figsize=figsize)
					for k,idx in enumerate(idxs):
						ax = axs[k]
						dataset.max_day = day
						data_, target_, lcobj = dataset.__getitem__(idx, return_lcobjs=True)
						#print(data, target)
						data, target = prepare_individual_sample(data_, target_, train_handler.GPU)
						model_out = trainh.model(data)
						onehot = data['onehot']
						for kb,b in enumerate(dataset.band_names):
							tmax_target = target[f'tmax.{b}'].cpu().numpy()[0]
							len_b = onehot.sum(dim=1)[0][kb].item()
							if len_b>0:
								last_t = model_out['last_time'].cpu().numpy()[0][0]
								tmax_pred = last_t+model_out[f'tmax_diff_pred.{b}'].cpu().numpy()[0][0]
								aplots.plot_lightcurve(ax, lcobj, b, label=f'{b} observation', max_day=dataset.max_day)
								if not np.isnan(tmax_target):
									ax.axvline(tmax_target, ls='-', c=aC_.COLOR_DICT[b], label=f'pm tmax target - {tmax_target:.1f}[days]')
								ax.axvline(tmax_pred, ls='--', c=aC_.COLOR_DICT[b], label=f'pm tmax pred - {tmax_pred:.1f}[days]')
						
						#ax.axvline(last_t, ls='-', c='k', label=f'last time - {last_t:.1f}[days]')
						ax.axvline(day, ls='-', c='k', alpha=0.5)

						if plot_lims[k]['x'] is None:
							plot_lims[k]['x'] = ax.get_xlim()
							plot_lims[k]['y'] = ax.get_ylim()

						title = f'survey: {dataset.survey} - key: {keys[k]} - total obs: {onehot.sum()} - class: {dataset.class_names[lcobj.y]}'
						ax.set_title(title)
						ax.set_ylabel('flux')
						ax.set_xlim(plot_lims[k]['x'])
						ax.set_ylim(plot_lims[k]['y'])
						ax.legend(loc='upper right')
						ax.grid(alpha=0.5)

					ax.set_xlabel('days')
					fig.tight_layout()
					animation.add_frame(fig)
					if animation.dummy:
						plt.show()
						break
					else:
						plt.close()
					
			except KeyboardInterrupt:
				can_be_in_loop = False

	dataset.reset_max_day() # very important
	video_save_dir = f'{save_rootdir}/{train_handler.model_name}'
	video_save_cfilename = f'exp-tmax{exp_id}_id-{train_handler.id}.{dataset.set_name}'
	animation.save(video_save_dir, video_save_cfilename, reverse=True)
	return