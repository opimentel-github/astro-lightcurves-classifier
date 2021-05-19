from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import flamingchoripan.files as fcfiles
import flamingchoripan.strings as strings
from flamingchoripan.dataframes import DFBuilder
from flamingchoripan.datascience.xerror import XError
from . import utils as utils
import pandas as pd
from scipy.interpolate import interp1d

###################################################################################################################################################

def get_parallel_serial_df(rootdir, cfilename, kf, lcset_name, model_names, dmetrics,
	day=None,
	train_mode='fine-tuning',
	arch_modes=['Parallel', 'Serial'],
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

	p_model_names = []
	s_model_names = []
	for model_name in model_names:
		is_parallel = 'Parallel' in model_name
		if is_parallel:
			p_model_names += [model_name]
		else:
			s_model_names += [model_name]

	for kmn,model_name in enumerate(sorted(p_model_names)+sorted(s_model_names)):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
		files, files_ids = fcfiles.gather_files_by_kfold(load_roodir, kf, lcset_name, fext='d')
		#print(f'ids={files_ids}(n={len(files_ids)}#) - model={model_name}')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		days = files[0]()['days']
		mn_dict = strings.get_dict_from_string(model_name)
		rsc = mn_dict['rsc']
		mdl = mn_dict['mdl']
		is_parallel = 'Parallel' in mdl

		d = {}
		for km,metric_name in enumerate(dmetrics.keys()):
			metric_curve = np.concatenate([f()['days_class_metrics_df'][metric_name].values[None] for f in files], axis=0)
			interp_metric_curve = interp1d(days, metric_curve)(np.linspace(days.min(), day, int(n)))
			xe_metric_curve = XError(interp_metric_curve[:,-1])
			xe_metric_curve_avg = XError(np.mean(interp_metric_curve, axis=-1))

			mn = metric_name if dmetrics[metric_name]['mn'] is None else dmetrics[metric_name]['mn']
			#_d_key = strings.get_string_from_dict({k:mn_dict[k] for k in mn_dict.keys() if k in label_keys}, key_key_separator=' - ')
			#d_key = f'{mdl} ({_d_key})'
			#d_key = f'{mdl} [{arch_mode}]' if override_model_name else f'{label} [{arch_mode}]'
			d[mn] = xe_metric_curve_avg if uses_avg else xe_metric_curve

		#index = f'model={model_name}'
		te_dims = int(mn_dict.get('te-dims', 0))
		cell = mn_dict.get('cell', None)
		new_model_name = mdl+(f' w/ M***{te_dims//2}' if te_dims>0 else '')+(f' w/ cell***{cell}' if not cell is None else '')
		#print(new_model_name)
		index = f'model={new_model_name}'
		#index = f'metric={utils.get_mday_avg_str(mn, day) if uses_avg else utils.get_mday_str(metric_name, day)}' 
		#print(index, d)
		#assert 0
		info_df.append(index, d)

	#print(info_df.indexs)
	return info_df