from __future__ import print_function
from __future__ import division
from . import C_

import torch
from fuzzytorch.utils import get_model_name, TDictHolder, tensor_to_numpy, minibatch_dict_collate
from fuzzytorch.models.utils import count_parameters
import numpy as np
from fuzzytools.progress_bars import ProgressBar, ProgressBarMulti
import fuzzytools.files as files
import fuzzytools.datascience.metrics as fcm
from fuzzytools.cuteplots.utils import save_fig
from fuzzytools.dataframes import DFBuilder
from fuzzytools.dicts import update_dicts
import matplotlib.pyplot as plt
import fuzzytorch.models.seq_utils as seq_utils
import pandas as pd

###################################################################################################################################################

def save_performance(train_handler, data_loader, save_rootdir,
	target_is_onehot:bool=False,
	classifier_key='y_last_pt',
	days_n:int=C_.DEFAULT_DAYS_N,
	**kwargs):
	train_handler.load_model() # important, refresh to best model
	train_handler.model.eval() # model eval
	data_loader.eval() # set mode
	dataset = data_loader.dataset # get dataset
	dataset.reset_max_day() # always reset max day
	dataset.uses_precomputed_samples = False

	days = np.linspace(C_.DEFAULT_MIN_DAY, dataset.max_day, days_n)#[::-1]
	days_rec_metrics_df = DFBuilder()
	days_class_metrics_df = DFBuilder()
	days_class_metrics_cdf = {c:DFBuilder() for c in dataset.class_names}
	days_cm = {}
	days_wrongs_df = {}
	bar_rows = 4
	bar = ProgressBarMulti(len(days), bar_rows)
	with torch.no_grad():
		can_be_in_loop = True
		for day in days: # along days
			dataset.set_max_day(day)
			try:
				if can_be_in_loop:
					tdict = []
					for ki,in_tdict in enumerate(data_loader):
						_tdict = train_handler.model(TDictHolder(in_tdict).to(train_handler.device))
						_tdict = TDictHolder(_tdict).to('cpu') # cpu to save gpu memory
						tdict.append(_tdict)
					tdict = minibatch_dict_collate(tdict)

					### decoder
					mse_loss_bdict = {}
					for kb,b in enumerate(dataset.band_names):
						p_onehot = tdict['input'][f'onehot.{b}'][...,0] # (b,t)
						#p_time = tdict['input'][f'time.{b}'][...,0] # (b,t)
						#p_dtime = tdict['input'][f'dtime.{b}'][...,0] # (b,t)
						#p_x = tdict['input'][f'x.{b}'] # (b,t,f)
						p_error = tdict['target'][f'error.{b}'] # (b,t,1)
						p_rx = tdict['target'][f'recx.{b}'] # (b,t,1)

						p_rx_pred = tdict['model'][f'decx.{b}'] # (b,t,1)
						mse_loss_b = (p_rx-p_rx_pred)**2/(p_error**2+C_.REC_LOSS_EPS) # (b,t,1)
						#mse_loss_b = torch.abs(p_rx-p_rx_pred)/(p_error**2+C_.REC_LOSS_EPS) # (b,t,1)
						mse_loss_b = seq_utils.seq_avg_pooling(mse_loss_b, p_onehot)[...,0] # (b,t,1) > (b,t) > (b)
						mse_loss_bdict[b] = mse_loss_b[...,0] # (b,1) > (b)

					mse_loss = torch.cat([mse_loss_bdict[b][...,None] for b in dataset.band_names], axis=-1).mean(dim=-1) # (b,d) > (b)
					mse_loss = tensor_to_numpy(mse_loss)
					mse_loss = mse_loss.mean()

					days_rec_metrics_df.append(day, {
						'_day':day,
						'mse':mse_loss,
						})

					### class prediction
					y_target = tdict['target']['y']
					#y_pred_p = torch.nn.functional.softmax(tdict['model'][classifier_key], dim=-1)
					y_pred_p = torch.sigmoid(tdict['model'][classifier_key])
					#print('y_pred_p',y_pred_p[0])

					if target_is_onehot:
						assert y_pred_.shape==y_target.shape
						y_target = torch.argmax(y_target, dim=-1)

					y_target = tensor_to_numpy(y_target)
					y_pred_p = tensor_to_numpy(y_pred_p)

					metrics_cdict, metrics_dict, cm = fcm.get_multiclass_metrics(y_pred_p, y_target, dataset.class_names)
					for c in dataset.class_names:
						days_class_metrics_cdf[c].append(day, update_dicts([{'_day':day}, metrics_cdict[c]]))
					days_class_metrics_df.append(day, update_dicts([{'_day':day}, metrics_dict]))

					### cm
					days_cm[day] = cm

					### wrong samples
					y_pred = np.argmax(y_pred_p, axis=-1)
					lcobj_names = dataset.get_lcobj_names()
					wrong_classification = ~(y_target==y_pred)
					assert len(lcobj_names)==len(wrong_classification)
					wrongs_df = DFBuilder()
					for kwc,wc in enumerate(wrong_classification):
						if wc:
							wrongs_df.append(lcobj_names[kwc], {'y_target':dataset.class_names[y_target[kwc]], 'y_pred':dataset.class_names[y_pred[kwc]]})
					days_wrongs_df[day] = wrongs_df.get_df()

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


		'days':days,
		'days_rec_metrics_df':days_rec_metrics_df.get_df(),
		'days_class_metrics_df':days_class_metrics_df.get_df(),
		'days_class_metrics_cdf':{c:days_class_metrics_cdf[c].get_df() for c in dataset.class_names},
		'days_cm':days_cm,
		'days_wrongs_df':days_wrongs_df,
	}

	### save file
	save_filedir = f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}.d'
	files.save_pickle(save_filedir, results) # save file
	dataset.reset_max_day() # very important!!
	return