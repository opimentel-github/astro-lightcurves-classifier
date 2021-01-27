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
import flamingchoripan.emails as emails
import fuzzytorch.models.seq_utils as seq_utils

###################################################################################################################################################

def reconstruction_along_days(train_handler, data_loader,
	figsize:tuple=C_.DEFAULT_FIGSIZE_REC,
	save_rootdir:str='results',
	save_fext:str='exprec',
	days_N:int=C_.DEFAULT_DAYS_N,
	eps:float=C_.EPS,
	send_email:bool=False,
	**kwargs):
	### dataloader and extract dataset - important
	train_handler.load_model() # important, refresh to best model
	data_loader.eval() # set mode
	dataset = data_loader.dataset # get dataset
	dataset.reset_max_day() # always reset max day
	dataset.uses_pre_generated_samples = False

	days = np.linspace(C_.DEFAULT_MIN_DAY, dataset.max_day, days_N)
	bar = ProgressBar(len(days), 3)
	results = {d:{} for d in days}
	results['days'] = []
	train_handler.model.eval() # model eval
	with torch.no_grad():
		can_be_in_loop = True
		for day in days: # along days
			try:
				if can_be_in_loop:
					dataset.set_max_day(day)
					results['days'].append(day)
					mse_loss = []
					ase_loss = []
					for k,tdict in enumerate(data_loader):
						out_tdict = train_handler.model(TDictHolder(tdict).to(train_handler.device))
						onehot = out_tdict['input']['onehot']
						b,t,_ = onehot.size()
						mse_loss_bdict = {}
						ase_loss_bdict = {}
						for kb,b in enumerate(dataset.band_names):
							p_error = seq_utils.serial_to_parallel(out_tdict['input']['error'], onehot[...,kb]) # (b,t,1)
							p_rx = seq_utils.serial_to_parallel(out_tdict['target']['raw-x'], onehot[...,kb]) # (b,t,1)
							p_rx_pred = out_tdict['model'][f'raw-x.{b}'] # (b,t,1)

							mse_loss_b = (p_rx-p_rx_pred)**2/(p_error**2 + C_.EPS) # (b,t,1)
							mse_loss_b = seq_utils.seq_avg_pooling(mse_loss_b, seq_utils.get_seq_onehot_mask(onehot[...,kb].sum(dim=-1), t)) # (b,t,1) > (b,1)
							mse_loss_bdict[b] = mse_loss_b[...,0]
			
							ase_loss_b = torch.abs(p_rx-p_rx_pred)/(p_error**2 + C_.EPS) # (b,t,1)
							ase_loss_b = seq_utils.seq_avg_pooling(ase_loss_b, seq_utils.get_seq_onehot_mask(onehot[...,kb].sum(dim=-1), t)) # (b,t,1) > (b,1)
							ase_loss_bdict[b] = ase_loss_b[...,0]

						mse_loss_ = torch.cat([mse_loss_bdict[b][...,None] for b in dataset.band_names], axis=-1).mean(dim=-1) # (b,d) > (b)
						mse_loss.append(mse_loss_.mean().cpu().numpy())

						ase_loss_ = torch.cat([ase_loss_bdict[b][...,None] for b in dataset.band_names], axis=-1).mean(dim=-1) # (b,d) > (b)
						ase_loss.append(ase_loss_.mean().cpu().numpy())

					mse_loss = np.mean(mse_loss)
					results[day]['__mse__'] = mse_loss
					ase_loss = np.mean(ase_loss)
					results[day]['__ase__'] = ase_loss
					bar(f'day: {day:.4f}/{days[-1]:.4f} - mse_loss: {mse_loss:.4f} - ase_loss: {ase_loss:.4f}')

			except KeyboardInterrupt:
				can_be_in_loop = False

	bar.done()
	dataset.uses_pre_generated_samples = True  # very important!!
	dataset.reset_max_day() # very important!!

	### plot for sanity checks
	fig, ax = plt.subplots(1, 1, figsize=figsize)
	days = results['days']
	for metric_to_plot in ['__mse__', '__ase__']:
		try:
			ax.plot(days, [results[d][metric_to_plot] for d in days], label=metric_to_plot.replace('__',''))
		except:
			pass
	ax.set_xlabel('days')
	ax.set_ylabel('metric-value')
	ax.legend()
	complete_save_roodir = train_handler.complete_save_roodir.split('/')[-1] # train_handler.get_complete_save_roodir().split('/')[-1]
	title = f'survey: {dataset.survey} - set: {dataset.set_name}\n'
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
	filedir = f'{file_save_dir}/id={train_handler.id}°set={dataset.set_name}.{save_fext}'
	files.save_pickle(filedir, results) # save file

	### send email
	if send_email:
		email_dict = {
			'subject':filedir,
			'content':complete_save_roodir,
			'images':[img_dir],
		}
		emails.send_mail(C_.EMAIL, email_dict)
	return

def metrics_along_days(train_handler, data_loader,
	target_is_onehot:bool=False,
	figsize:tuple=C_.DEFAULT_FIGSIZE_REC,
	save_rootdir:str='results',
	save_fext:str='expmet',
	days_N:int=C_.DEFAULT_DAYS_N,
	eps:float=C_.EPS,
	send_email:bool=False,
	**kwargs):
	### dataloader and extract dataset - important
	train_handler.load_model() # important, refresh to best model
	data_loader.eval() # set mode
	dataset = data_loader.dataset # get dataset
	dataset.reset_max_day() # always reset max day
	dataset.uses_pre_generated_samples = False

	days = np.linspace(C_.DEFAULT_MIN_DAY, dataset.max_day, days_N)
	bar = ProgressBarMulti(len(days), 3)
	results = {d:{} for d in days}
	results['days'] = []
	train_handler.model.eval() # model eval
	with torch.no_grad():
		can_be_in_loop = True
		for day in days: # along days
			try:
				if can_be_in_loop:
					dataset.set_max_day(day)
					results['days'].append(day)
					y_target = []
					y_pred = []
					for k,tdict in enumerate(data_loader):
						out_tdict = train_handler.model(TDictHolder(tdict).to(train_handler.device))
						onehot = out_tdict['input']['onehot']
						y_target_ = out_tdict['target']['y']
						y_pred_ = out_tdict['model']['y.last']

						if target_is_onehot:
							assert y_pred_.shape==y_target_.shape
							y_target_ = y_target_.argmax(dim=-1)

						y_pred_ = y_pred_.argmax(dim=-1)
						#print(y_target_.shape, y_pred_.shape)
						y_target.append(y_target_)
						y_pred.append(y_pred_)

					y_target = torch.cat(y_target, dim=0)
					y_pred = torch.cat(y_pred, dim=0)
					met_kwargs = {
						'pred_is_onehot':False,
						'target_is_onehot':False,
					}
					metrics_cdict, metrics_dict, cm = fcm.get_all_metrics_c(y_pred.cpu().numpy(), y_target.cpu().numpy(), dataset.class_names, **met_kwargs)
					results[day].update(metrics_cdict)
					results[day].update({f'__{key}__':metrics_dict[key] for key in metrics_dict.keys()})
					results[day]['cm'] = cm
					bar([f'day: {day:.4f}/{days[-1]:.4f}', f'metrics_dict: {metrics_dict}', f'metrics_cdict: {metrics_cdict}'])

			except KeyboardInterrupt:
				can_be_in_loop = False

	bar.done()
	dataset.uses_pre_generated_samples = True  # very important!!
	dataset.reset_max_day() # very important!!

	### plot for sanity checks
	fig, ax = plt.subplots(1, 1, figsize=figsize)
	days = results['days']
	for metric_to_plot in ['__b-accuracy__', '__b-f1score__']:
		try:
			ax.plot(days, [results[d][metric_to_plot] for d in days], label=metric_to_plot.replace('__',''))
		except:
			pass
	ax.set_xlabel('days')
	ax.set_ylabel('metric-value')
	ax.legend()
	complete_save_roodir = train_handler.complete_save_roodir.split('/')[-1] # train_handler.get_complete_save_roodir().split('/')[-1]
	title = f'survey: {dataset.survey} - set: {dataset.set_name}\n'
	title += f'model: {complete_save_roodir} - id: {train_handler.id}'
	ax.set_title(title)
	ax.grid(alpha=0.25)
	fig.tight_layout()
	img_dir = f'../temp/metrics_along_days.png'
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
	filedir = f'{file_save_dir}/id={train_handler.id}°set={dataset.set_name}.{save_fext}'
	files.save_pickle(filedir, results) # save file

	### send email
	if send_email:
		email_dict = {
			'subject':filedir,
			'content':complete_save_roodir,
			'images':[img_dir],
		}
		emails.send_mail(C_.EMAIL, email_dict)
	return