from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from flamingchoripan.files import search_for_filedirs, load_pickle
import flamingchoripan.strings as strings
import flamingchoripan.datascience.statistics as dstats
from . import utils as utils
import pandas as pd

###################################################################################################################################################

def get_df_table(rootdir, metric_names, model_names, day_to_metric, query_key, query_values,
	fext='metrics',
	arch_modes=['Parallel', 'Serial'],
	):
	index_df = []
	info_df = {}
	for arch_mode in arch_modes:
		for y in query_values:
			info_df[f'{str(arch_mode)}-{str(y)}'] = []

	for kax,mode in enumerate(['pre-training', 'fine-tuning']):
		for kmn,model_name in enumerate(model_names):
			new_rootdir = f'{rootdir}/{mode}/{model_name}'
			new_rootdir = new_rootdir.replace('mode=pre-training', f'mode={mode}') # patch
			new_rootdir = new_rootdir.replace('mode=fine-tuning', f'mode={mode}') # patch
			filedirs = search_for_filedirs(new_rootdir, fext=fext, verbose=0)
			print(f'[{kmn}][{len(filedirs)}#] {model_name}')
			mn_dict = strings.get_dict_from_string(model_name)
			rsc = mn_dict['rsc']
			mdl = mn_dict['mdl']
			is_parallel = 'Parallel' in mdl

			if is_parallel: # fix
				for km,metric_name in enumerate(metric_names):
					day_metric = []
					day_metric_avg = []
					for filedir in filedirs:
						rdict = load_pickle(filedir, verbose=0)
						#model_name = rdict['model_name']
						days = rdict['days']
						survey = rdict['survey']
						band_names = ''.join(rdict['band_names'])
						class_names = rdict['class_names']
						v, vs, _ = utils.get_metric_along_day(days, rdict, metric_name, day_to_metric)
						day_metric += [v]
						day_metric_avg += [vs.mean()]

					xe_day_metric = dstats.XError(day_metric, 0)
					xe_day_metric_avg = dstats.XError(day_metric_avg, 0)
					arch_mode = 'Parallel' if is_parallel else 'Serial'
					key = f'{arch_mode}-{mn_dict[query_key]}'
					info_df[key] += [xe_day_metric]
					info_df[key] += [xe_day_metric_avg]

					if kmn==0:
						index_df += [f'metric={utils.get_mday_str(metric_name, day_to_metric)}°training-mode={mode}']
						index_df += [f'metric={utils.get_mday_avg_str(metric_name, day_to_metric)}°training-mode={mode}']

	info_df = pd.DataFrame.from_dict(info_df)
	info_df.index = index_df
	return info_df