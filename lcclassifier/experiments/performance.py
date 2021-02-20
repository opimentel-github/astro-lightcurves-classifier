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
import pandas as pd

###################################################################################################################################################

def metrics_along_days(train_handler, data_loader,
	target_is_onehot:bool=False,
	classifier_key='y.last',
	
	figsize:tuple=C_.DEFAULT_FIGSIZE_REC,
	save_rootdir:str='results',
	save_fext:str='metrics',
	days_N:int=C_.DEFAULT_DAYS_N,
	eps:float=C_.EPS,
	**kwargs):
	### dataloader and extract dataset - important
	train_handler.load_model() # important, refresh to best model
	train_handler.model.eval() # model eval
	data_loader.eval() # set mode
	dataset = data_loader.dataset # get dataset
	dataset.reset_max_day() # always reset max day
	dataset.uses_precomputed_samples = False

	days = np.linspace(C_.DEFAULT_MIN_DAY, dataset.max_day, days_N)#[::-1]
	bar_rows = 4
	bar = ProgressBarMulti(len(days), bar_rows)
	days_rec_metrics_df = []
	days_class_metrics_df = []
	days_class_metrics_cdf = {c:[] for c in dataset.class_names}
	days_cm = {}
	with torch.no_grad():
		can_be_in_loop = True
		for day in days: # along days
			try:
				if can_be_in_loop:
					dataset.set_max_day(day)
					mse_loss = []
					y_target = []
					y_pred_p = []
					for k,tdict in enumerate(data_loader):
						out_tdict = train_handler.model(TDictHolder(tdict).to(train_handler.device))
						onehot = out_tdict['input']['onehot']

						### decoder
						mse_loss_bdict = {}
						for kb,b in enumerate(dataset.band_names):
							p_error = seq_utils.serial_to_parallel(out_tdict['input']['error'], onehot[...,kb]) # (b,t,1)
							p_rx = seq_utils.serial_to_parallel(out_tdict['target']['rec-x'], onehot[...,kb]) # (b,t,1)
							p_rx_pred = out_tdict['model'][f'rec-x.{b}'] # (b,t,1)

							mse_loss_b = (p_rx-p_rx_pred)**2/(p_error**2 + C_.EPS) # (b,t,1)
							mse_loss_b = seq_utils.seq_avg_pooling(mse_loss_b, seq_utils.get_seq_onehot_mask(onehot[...,kb].sum(dim=-1), onehot.shape[1])) # (b,t,1) > (b,1)
							mse_loss_bdict[b] = mse_loss_b[...,0] # (b,1) > (b)

						mse_loss_ = torch.cat([mse_loss_bdict[b][...,None] for b in dataset.band_names], axis=-1).mean(dim=-1) # (b,d) > (b)
						mse_loss.append(mse_loss_.mean().cpu().numpy())

						### class
						y_target_ = out_tdict['target']['y']
						y_pred_p_ = torch.nn.functional.softmax(out_tdict['model'][classifier_key], dim=-1)

						if target_is_onehot:
							assert y_pred_.shape==y_target_.shape
							y_target_ = torch.argmax(y_target_, dim=-1)

						y_target.append(y_target_)
						y_pred_p.append(y_pred_p_)

					### decoder metrics
					mse_loss = np.mean(mse_loss)
					day_df = pd.DataFrame.from_dict({
						'day':[day],
						'mse':[mse_loss],
						})
					days_rec_metrics_df.append(day_df)

					### class metrics
					y_target = torch.cat(y_target, dim=0).cpu().numpy()
					y_pred_p = torch.cat(y_pred_p, dim=0).cpu().numpy()
					y_pred = np.argmax(y_pred_p, axis=-1)
					accuracy = (y_target==y_pred).astype(np.float)*100
					#print('accuracy', accuracy.shape, np.mean(accuracy))
					met_kwargs = {
						'pred_is_onehot':False,
						'target_is_onehot':False,
						'y_pred_p':y_pred_p,
					}
					metrics_cdict, metrics_dict, cm = fcm.get_multiclass_metrics(y_pred, y_target, dataset.class_names, **met_kwargs)
					
					d = {'day':[day]}
					for km in metrics_dict.keys():
						d[km] = [metrics_dict[km]]
					day_df = pd.DataFrame.from_dict(d)
					days_class_metrics_df.append(day_df)

					for c in dataset.class_names:
						d = {'day':[day]}
						for km in metrics_cdict.keys():
							d[km] = [metrics_cdict[km][c]]
						day_df = pd.DataFrame.from_dict(d)
						days_class_metrics_cdf[c].append(day_df)

					days_cm[day] = cm
					bar([f'day: {day:.4f}/{days[-1]:.4f}', f'mse_loss: {mse_loss}', f'metrics_dict: {metrics_dict}', f'metrics_cdict: {metrics_cdict["recall"]}'])
					#break # dummy

			except KeyboardInterrupt:
				can_be_in_loop = False

	bar.done()
	dataset.uses_precomputed_samples = True  # very important!!
	dataset.reset_max_day() # very important!!

	days_rec_metrics_df = pd.concat(days_rec_metrics_df)
	days_class_metrics_df = pd.concat(days_class_metrics_df)
	days_class_metrics_cdf = {c:pd.concat(days_class_metrics_cdf[c]) for c in dataset.class_names}

	### more info
	complete_save_roodir = train_handler.complete_save_roodir.split('/')[-1] # train_handler.get_complete_save_roodir().split('/')[-1]
	results = {
		'days':days,
		'days_rec_metrics_df':days_rec_metrics_df,
		'days_class_metrics_df':days_class_metrics_df,
		'days_class_metrics_cdf':days_class_metrics_cdf,
		'days_cm':days_cm,

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