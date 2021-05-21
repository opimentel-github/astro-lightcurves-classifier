from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import fuzzytools.files as fcfiles
import fuzzytools.strings as strings
from fuzzytools.dataframes import DFBuilder
from fuzzytools.datascience.xerror import XError
from . import utils as utils
import pandas as pd
from scipy.interpolate import interp1d

###################################################################################################################################################

def get_parallel_serial_df(rootdir, cfilename, kf, lcset_name, model_names, dmetrics,
	day=None,
	train_mode='fine-tuning',
	n=1e3,
	#override_model_name=True,
	label_keys=[],
	uses_avg=False,
	baseline_filedir=None,
	):
	info_df = DFBuilder()
	if not baseline_filedir is None:
		d = fcfiles.load_pickle(baseline_filedir)['synthetic-method=no-method [r]']
		if uses_avg:
			d = {k:XError([-9999]) for k in d.keys()}
		info_df.append('model=b-RF w/ FATS', d)

	for kmn,model_name in enumerate(utils.get_sorted_model_names(model_names)):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
		files, files_ids = fcfiles.gather_files_by_kfold(load_roodir, kf, lcset_name, fext='d')
		#print(f'ids={files_ids}(n={len(files_ids)}#) - model={model_name}')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		days = files[0]()['days']

		d = {}
		for km,metric_name in enumerate(dmetrics.keys()):
			metric_curve = np.concatenate([f()['days_class_metrics_df'][metric_name].values[None] for f in files], axis=0)
			interp_metric_curve = interp1d(days, metric_curve)(np.linspace(days.min(), day, int(n)))
			xe_metric_curve = XError(interp_metric_curve[:,-1])
			xe_metric_curve_avg = XError(np.mean(interp_metric_curve, axis=-1))

			mn = metric_name if dmetrics[metric_name]['mn'] is None else dmetrics[metric_name]['mn']
			d[mn] = xe_metric_curve_avg if uses_avg else xe_metric_curve

		index = f'model={utils.get_fmodel_name(model_name)}'
		info_df.append(index, d)

	return info_df