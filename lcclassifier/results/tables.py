from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import fuzzytools.files as ftfiles
import fuzzytools.strings as strings
from fuzzytools.dataframes import DFBuilder
from fuzzytools.datascience.xerror import XError
from . import utils as utils
import pandas as pd
from scipy.interpolate import interp1d

RANDOM_STATE = 0

###################################################################################################################################################

def get_ps_performance_df(rootdir, cfilename, kf, lcset_name, model_names, dmetrics,
	target_class=None,
	day=None,
	train_mode='fine-tuning',
	n=1e3,
	uses_avg=False,
	baseline_roodir=None,
	):
	info_df = DFBuilder()
	if not baseline_roodir is None:
		files, files_ids = ftfiles.gather_files_by_kfold(baseline_roodir, kf, lcset_name,
			fext='df', # fixme
			disbalanced_kf_mode='oversampling', # error oversampling
			random_state=RANDOM_STATE,
			)
		print(files[0])
		# d = ftfiles.load_pickle(baseline_filedir)['synthetic-method=no-method [r]']
		# if uses_avg:
			# d = {k:XError([-np.inf]) for k in d.keys()}
		# info_df.append('model=b-RF w/ FATS', d)

	return info_df

	for kmn,model_name in enumerate(utils.get_sorted_model_names(model_names)):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
		files, files_ids = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
			fext='d',
			disbalanced_kf_mode='oversampling', # error oversampling
			random_state=RANDOM_STATE,
			)
		print(f'{model_name} {files_ids}({len(files_ids)}#); model={model_name}')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		days = files[0]()['days']

		d = {}
		for km,metric_name in enumerate(dmetrics.keys()):
			if target_class is None:
				metric_curve = np.concatenate([f()['days_class_metrics_df'][f'b-{metric_name}'].values[None] for f in files], axis=0)
			else:
				metric_curve = np.concatenate([f()['days_class_metrics_cdf'][target_class][f'{metric_name}'].values[None] for f in files], axis=0)
			interp_metric_curve = interp1d(days, metric_curve)(np.linspace(days.min(), day, int(n)))
			xe_metric_curve = XError(interp_metric_curve[:,-1])
			xe_metric_curve_avg = XError(np.mean(interp_metric_curve, axis=-1))

			new_metric_name = f'{target_class}-{metric_name if dmetrics[metric_name]["mn"] is None else dmetrics[metric_name]["mn"]}'
			d[new_metric_name] = xe_metric_curve_avg if uses_avg else xe_metric_curve

		index = f'model={utils.get_fmodel_name(model_name)}'
		info_df.append(index, d)

	return info_df

###################################################################################################################################################

def get_ps_times_df(rootdir, cfilename, kf, method, model_names,
	train_mode='pre-training',
	):
	info_df = DFBuilder()
	new_model_names = utils.get_sorted_model_names(model_names)
	for kmn,model_name in enumerate(new_model_names):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/model_info/{cfilename}'
		files, files_ids = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
			fext='d',
			disbalanced_kf_mode='oversampling', # error oversampling
			random_state=RANDOM_STATE,
			)
		print(f'{model_name} {files_ids}({len(files_ids)}#); model={model_name}')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		is_parallel = 'Parallel' in model_name

		loss_name = 'wmse-xentropy'
		print(files[0]()['monitors'][loss_name].keys())

		#th = 1 # bug?
		d = {}
		parameters = [f()['parameters'] for f in files][0]
		d['params'] = parameters
		#d['best_epoch'] = XError([f()['monitors']['wmse-xentropy']['best_epoch'] for f in files])
		d['time-per-iteration [segs]'] = sum([f()['monitors'][loss_name]['time_per_iteration'] for f in files])
		#print(d['time-per-iteration [segs]'].max())
		#d['time-per-iteration/params $1e6\\cdot$[segs]'] = sum([f()['monitors'][loss_name]['time_per_iteration']/parameters*1e6 for f in files])
		#d['time_per_epoch'] = sum([f()['monitors']['wmse-xentropy']['time_per_epoch'] for f in files])
		

		print(files[0]()['monitors'][loss_name]['time_per_epoch'])

		#d['time_per_epoch [segs]'] = sum([f()['monitors'][loss_name]['time_per_epoch'] for f in files])
		d['time-per-epoch [segs]'] = XError([f()['monitors'][loss_name]['total_time']/1500 for f in files])

		d['total-time [mins]'] = XError([f()['monitors'][loss_name]['total_time']/60 for f in files])
		

		index = f'model={utils.get_fmodel_name(model_name)}'
		info_df.append(index, d)

	return info_df