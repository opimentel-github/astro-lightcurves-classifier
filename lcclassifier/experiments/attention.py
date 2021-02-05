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
from scipy import stats

###################################################################################################################################################

def attention_statistics(train_handler, data_loader,
	figsize:tuple=C_.DEFAULT_FIGSIZE_REC,
	save_rootdir:str='results',
	save_fext:str='attnscores',
	days_N:int=C_.DEFAULT_DAYS_N,
	eps:float=C_.EPS,
	**kwargs):
	### dataloader and extract dataset - important
	train_handler.load_model() # important, refresh to best model
	data_loader.eval() # set mode
	dataset = data_loader.dataset # get dataset
	dataset.reset_max_day() # always reset max day
	dataset.uses_precomputed_samples = False

	results = []
	train_handler.model.eval() # model eval
	with torch.no_grad():
		lcobj_names = dataset.get_lcobj_names()

		for k,lcobj_name in enumerate(lcobj_names):
			tdict, lcobj = dataset.get_item(lcobj_name, return_lcobjs=True)
			out_tdict = train_handler.model(TDictHolder(tdict).to(train_handler.device, add_dummy_dim=True))
			uses_attn = 'layer_scores' in out_tdict['model'].keys()

			onehot = out_tdict['input']['onehot']
			s_onehot = onehot.sum(dim=-1).bool()
			lcobj_len = s_onehot.sum().item()
			attn_scores = out_tdict['model']['layer_scores'][-1] # from last layer (b,h,t,q)
			#attn_scores = attn_scores.mean(dim=1) # collapse along heads (b,h,t,q) > (b,t,q)
			attn_scores = attn_scores.max(dim=1)[0] # collapse along heads (b,h,t,q) > (b,t,q)
			attn_scores = seq_utils.seq_last_element(attn_scores, s_onehot)[0] # get elements from last step (b,t,q) > (b,t)
			attn_scores_np = attn_scores.cpu().numpy()
			attn_entropy = -np.sum(attn_scores_np*np.log(attn_scores_np+1e-10))
			#print(attn_scores.shape, attn_scores.sum(-1), attn_scores)

			'''
			### serial
			x = lcobj.get_custom_x_serial(['obs', 'obse'])
			obs = x[:lcobj_len,0]
			obse = x[:lcobj_len,1]
			wobs = obs/(obse**2+C_.EPS)

			for i in range(lcobj_len):
				results.append({
					'i':i,
					'lcobj_len':lcobj_len,
					'attn':attn_scores[i],
					'obs':obs[i],
					'obs_soft':obs[i]/np.sum(obs),
					'obse':obse[i],
					'obse_soft':obse[i]/np.sum(obse),
					'wobs':wobs[i],
					'wobs_soft':wobs[i]/np.sum(wobs),
				})
			'''
			#for kb,b in enumerate(['g']):
			di = 3
			for kb,b in enumerate(dataset.band_names):
				lcobjb = lcobj.get_b(b)
				p_onehot = onehot[...,kb]
				lcobjb_len = p_onehot.sum().item()
				
				def norm(x, i):
					return x[i]/(x.sum()+C_.EPS)

				def min_max_norm(x, i):
					return (x[i]-x.min())/(x.max()-x.min()+C_.EPS)

				if lcobjb_len<10:
					continue
				p_attn_scores = seq_utils.serial_to_parallel(attn_scores[None,:,None], p_onehot).cpu().numpy()
				p_attn_scores = p_attn_scores[0,:lcobjb_len,0]
				p_attn_scores = p_attn_scores/np.sum(p_attn_scores) # norm to dist
				p_attn_entropy = -np.sum(p_attn_scores*np.log(p_attn_scores+1e-10))
				#print(b, p_attn_scores.shape, p_attn_scores.sum(-1), p_attn_scores)

				days = lcobjb.days[:lcobjb_len]
				obs = lcobjb.obs[:lcobjb_len]
				obse = lcobjb.obse[:lcobjb_len]
				wobs = obs/(obse**2+C_.EPS)
				peak_day = days[np.argmax(obs)]
				
				for i in range(di, lcobjb_len):
					#print(days[i-di:i])
					slope = stats.linregress(days[i-di:i], obs[i-di:i]).slope
					r = {
						'i':i,
						'b':b,
						'c':c,
						'attn_entropy':attn_entropy,
						'p_attn_entropy':p_attn_entropy,
						'p_attn_entropy/len':p_attn_entropy/lcobjb_len,
						'lcobj_len':lcobj_len,
						'lcobjb_len':lcobjb_len,
						'attn':p_attn_scores[i],
						'attn_soft':norm(p_attn_scores, i),
						'attn_mm':min_max_norm(p_attn_scores, i),
						'obs':obs[i],
						'obs_soft':norm(obs, i),
						'obs_mm':min_max_norm(obs, i),
						'obse':obse[i],
						'obse_soft':norm(obse, i),
						'obse_mm':min_max_norm(obse, i),
						'wobs':wobs[i],
						'wobs_soft':norm(wobs, i),
						'degrees':np.rad2deg(np.arctan(slope)),
						'slope':slope,
						'days':days[i],
						'days-to-peak':days[i]-peak_day,
						#'slope':np.arctan(slope),
					}
					#print(r)
					results.append(r)

	dataset.uses_precomputed_samples = True  # very important!!
	dataset.reset_max_day() # very important!!
	return results, {}, {}

	### plot for sanity checks
	fig, ax = plt.subplots(1, 1, figsize=figsize)
	days = results['days']
	for metric_to_plot in ['mse', 'ase']:
		try:
			ax.plot(days, [results[d]['metrics_dict'][metric_to_plot] for d in days], label=metric_to_plot)
		except:
			pass
	ax.set_xlabel('days')
	ax.set_ylabel('metric-value')
	ax.legend()
	complete_save_roodir = train_handler.complete_save_roodir.split('/')[-1] # train_handler.get_complete_save_roodir().split('/')[-1]
	title = f'survey: {dataset.survey} - set: {dataset.lcset_name}\n'
	title += f'model: {complete_save_roodir} - id: {train_handler.id}'
	ax.set_title(title)
	ax.grid(alpha=0.25)
	fig.tight_layout()
	img_dir = f'../temp/reconstruction_along_days.png'
	save_fig(img_dir, fig)

	### more info
	results['complete_save_roodir'] = complete_save_roodir
	results['model_name'] = train_handler.model.get_name()
	results['survey'] = dataset.survey
	results['band_names'] = dataset.band_names
	results['class_names'] = dataset.class_names
	results['parameters'] = count_parameters(train_handler.model)
	set_names = ['train', 'val']
	for lmonitor in train_handler.lmonitors:
		results[lmonitor] = {
			'save_dict':lmonitor.get_save_dict(),
			'best_epoch':lmonitor.get_best_epoch(),
			'time_per_iteration':lmonitor.get_time_per_iteration(),
			#'time_per_epoch_set':{set_name:lmonitor.get_time_per_epoch_set(set_name) for set_name in set_names},
			'time_per_epoch':lmonitor.get_time_per_epoch(),
			'total_time':lmonitor.get_total_time(),
		}

	### save file
	file_save_dir = f'{save_rootdir}/{complete_save_roodir}'
	filedir = f'{file_save_dir}/id={train_handler.id}°set={dataset.lcset_name}.{save_fext}'
	files.save_pickle(filedir, results) # save file
	return
