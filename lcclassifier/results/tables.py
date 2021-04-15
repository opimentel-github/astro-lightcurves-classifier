from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import flamingchoripan.files as fcfiles
import flamingchoripan.strings as strings
from flamingchoripan.dataframes import DFBuilder
from flamingchoripan.datascience.statistics import XError
from . import utils as utils
import pandas as pd
from scipy.interpolate import interp1d

###################################################################################################################################################

def get_parallel_serial_df(rootdir, cfilename, kf, lcset_name, model_names, metric_names,
	day_to_metric=None,
	train_mode='fine-tuning',
	arch_modes=['Parallel', 'Serial'],
	n=1e3,
	#override_model_name=True,
	label_keys=[],
	):
	info_df = DFBuilder()
	for km,metric_name in enumerate(metric_names):
		for uses_avg in [False, True]:
			d = {}
			for arch_mode in arch_modes:
				for kmn,model_name in enumerate(model_names):
					_model_name = model_name.replace('Serial', arch_mode).replace('Parallel', arch_mode)
					load_roodir = f'{rootdir}/{_model_name}/{train_mode}/exp=performance/{cfilename}/{kf}@{lcset_name}'
					files, files_ids = fcfiles.gather_files_by_id(load_roodir, fext='d')
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

					metric_curve = np.concatenate([f()['days_class_metrics_df'][metric_name].values[None] for f in files], axis=0)
					interp_days = np.linspace(days.min(), day_to_metric, int(n))
					interp_metric_curve = interp1d(days, metric_curve)(interp_days)
					xe_metric_curve = XError(interp_metric_curve[:,-1])
					xe_metric_curve_avg = XError(np.mean(interp_metric_curve, axis=-1))

					_d_key = strings.get_string_from_dict({k:mn_dict[k] for k in mn_dict.keys() if k in label_keys}, key_key_separator=' - ')
					d_key = f'{mdl} ({_d_key})'
					#d_key = f'{mdl} [{arch_mode}]' if override_model_name else f'{label} [{arch_mode}]'
					d[d_key] = xe_metric_curve_avg if uses_avg else xe_metric_curve

			index = f'metric={utils.get_mday_avg_str(metric_name, day_to_metric) if uses_avg else utils.get_mday_str(metric_name, day_to_metric)}' 
			info_df.append(index, d)

	return info_df()