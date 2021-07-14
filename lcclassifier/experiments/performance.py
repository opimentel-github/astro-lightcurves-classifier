from __future__ import print_function
from __future__ import division
from . import C_

import torch
from fuzzytorch.utils import TDictHolder, tensor_to_numpy, minibatch_dict_collate
import numpy as np
from fuzzytools.progress_bars import ProgressBar, ProgressBarMulti
import fuzzytools.files as files
import fuzzytools.datascience.metrics as fcm
from fuzzytools.dataframes import DFBuilder
from fuzzytools.dicts import update_dicts
import fuzzytorch.models.seq_utils as seq_utils
import pandas as pd

DEFAULT_DAYS_N = C_.DEFAULT_DAYS_N

###################################################################################################################################################

def save_performance(train_handler, data_loader, save_rootdir,
	target_is_onehot:bool=False,
	target_y_key='target/y',
	pred_y_key='model/y',
	days_n:int=DEFAULT_DAYS_N,
	**kwargs):
	train_handler.load_model() # important, refresh to best model
	train_handler.model.eval() # important, model eval mode
	dataset = data_loader.dataset # get dataset

	dataset.reset_max_day() # always reset max day
	days = np.linspace(C_.DEFAULT_MIN_DAY, dataset.max_day, days_n)#[::-1]
	days_rec_metrics_df = DFBuilder()
	days_class_metrics_df = DFBuilder()
	days_class_metrics_cdf = {c:DFBuilder() for c in dataset.class_names}
	days_predictions = {}
	days_cm = {}
	bar = ProgressBarMulti(len(days), 4)
	with torch.no_grad():
		can_be_in_loop = True
		for day in days:
			dataset.set_max_day(day) # very important!!
			dataset.calcule_precomputed() # very important!!
			try:
				if can_be_in_loop:
					tdicts = []
					for ki,in_tdict in enumerate(data_loader):
						_tdict = train_handler.model(TDictHolder(in_tdict).to(train_handler.device))
						tdicts += [_tdict]
					tdict = minibatch_dict_collate(tdicts)

					### mse
					mse_loss_bdict = {}
					for kb,b in enumerate(dataset.band_names):
						p_onehot = tdict[f'input/onehot.{b}'][...,0] # (b,t)
						#p_rtime = tdict[f'input/rtime.{b}'][...,0] # (b,t)
						#p_dtime = tdict[f'input/dtime.{b}'][...,0] # (b,t)
						#p_x = tdict[f'input/x.{b}'] # (b,t,f)
						p_rerror = tdict[f'target/rerror.{b}'] # (b,t,1)
						p_rx = tdict[f'target/recx.{b}'] # (b,t,1)

						p_rx_pred = tdict[f'model/decx.{b}'] # (b,t,1)
						mse_loss_b = (p_rx-p_rx_pred)**2/(C_.REC_LOSS_EPS+C_.REC_LOSS_K*(p_rerror**2)) # (b,t,1)
						mse_loss_b = seq_utils.seq_avg_pooling(mse_loss_b, p_onehot)[...,0] # (b,t,1) > (b,t) > (b)
						mse_loss_bdict[b] = mse_loss_b[...,0] # (b,1) > (b)

					mse_loss = torch.cat([mse_loss_bdict[b][...,None] for b in dataset.band_names], axis=-1).mean(dim=-1) # (b,d) > (b)
					mse_loss = mse_loss.mean()

					days_rec_metrics_df.append(day, {
						'_day':day,
						'mse':tensor_to_numpy(mse_loss),
						})

					### class prediction
					y_true = tdict[target_y_key] # (b)
					#y_pred_p = torch.nn.functional.softmax(tdict[pred_y_key], dim=-1) # (b,c)
					y_pred_p = torch.sigmoid(tdict[pred_y_key]) # (b,c)
					#print('y_pred_p',y_pred_p[0])

					if target_is_onehot:
						assert y_pred_.shape==y_true.shape
						y_true = torch.argmax(y_true, dim=-1)

					y_true = tensor_to_numpy(y_true)
					y_pred_p = tensor_to_numpy(y_pred_p)

					days_predictions[day] = {'y_true':y_true, 'y_pred_p':y_pred_p}
					metrics_cdict, metrics_dict, cm = fcm.get_multiclass_metrics(y_pred_p, y_true, dataset.class_names)
					for c in dataset.class_names:
						days_class_metrics_cdf[c].append(day, update_dicts([{'_day':day}, metrics_cdict[c]]))
					days_class_metrics_df.append(day, update_dicts([{'_day':day}, metrics_dict]))

					### cm
					days_cm[day] = cm

					### progress bar
					recall = {c:metrics_cdict[c]['recall'] for c in dataset.class_names}
					bmetrics_dict = {k:metrics_dict[k] for k in metrics_dict.keys() if 'b-' in k}
					bar([f'lcset_name={dataset.lcset_name} - day={day:.3f}/{days[-1]:.3f}', f'mse_loss={mse_loss}', f'bmetrics_dict={bmetrics_dict}', f'recall={recall}'])
					#break # dummy

			except KeyboardInterrupt:
				can_be_in_loop = False

	bar.done()
	results = {
		'model_name':train_handler.model.get_name(),
		'survey':dataset.survey,
		'band_names':dataset.band_names,
		'class_names':dataset.class_names,
		'lcobj_names':dataset.get_lcobj_names(),

		'days':days,
		'days_rec_metrics_df':days_rec_metrics_df.get_df(),
		'days_class_metrics_df':days_class_metrics_df.get_df(),
		'days_class_metrics_cdf':{c:days_class_metrics_cdf[c].get_df() for c in dataset.class_names},
		'days_predictions':days_predictions,
		'days_cm':days_cm,
	}

	### save file
	save_filedir = f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}.d'
	files.save_pickle(save_filedir, results) # save file
	dataset.reset_max_day() # very important!!
	dataset.calcule_precomputed() # very important!!
	return