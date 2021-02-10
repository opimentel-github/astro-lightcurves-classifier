from __future__ import print_function
from __future__ import division
from . import C_

import warnings
import numpy as np
from flamingchoripan.files import search_for_filedirs, load_pickle, get_dict_from_filedir
import flamingchoripan.strings as strings
import flamingchoripan.datascience.statistics as dstats
from scipy.interpolate import interp1d

###################################################################################################################################################
	
def get_day_metric(days, metric_curves, target_day):
	return interp1d(days, metric_curves)(target_day)

def formating_model_name(model_name):
	dname = strings.get_dict_from_string(model_name)
	keys_to_remove = ['bi', 'bn', 'rnnU', 'rnnL', 'inD']
	for key in keys_to_remove:
		dname.pop(key, None) # remove
	#print(dname)
	new_name = ' - '.join([f'{key}: {dname[key]}' for key in dname.keys()])
	new_name = strings.string_replacement(new_name, C_.FNAME_REPLACE_DICT)
	return new_name

def latex_mean_days(metric_name, iday, fday):
	return '$|'+metric_name+'|_{'+f'{iday:.1f}'+'d}^{'+f'{fday:.1f}'+'d}$'

def latex_day(metric_name, day):
	return '$'+metric_name+'|^{'+f'{day:.1f}'+'d}$'

###################################################################################################################################################

def get_day_metrics_from_models(root_folder, model_names, metric_name, fext,
	return_xerror=True,
	):
	metrics_dict = {}
	samples = []
	for kmn,model_name in enumerate(model_names):
		print(f'model_name: {model_name}')
		filedirs = search_for_filedirs(f'{root_folder}/{model_name}', fext=fext, verbose=0)
		metric_curves = []
		for filedir in filedirs:
			rdict = load_pickle(filedir, verbose=0)
			days = rdict['days']
			mdict_name = ''
			metric = rdict['days_class_metrics_df'][metric_name].values[:]
			print(metric)

			assert 0
			

			if metric_name=='*accu*':
				for d in days:
					rdict[d]['*accu*'] = np.sum(np.diagonal(rdict[d]['cm']))/np.sum(rdict[d]['cm'])

			if metric_name in ['*precision*', '*recall*', '*f1score*']:
				class_names = rdict['class_names']
				metric_curves.append(np.array([np.mean([rdict[d][metric_name.replace('*','')][c] for c in class_names]) for d in days])[None,...])
				#metric_curves.append(np.array([rdict[d][metric_name] for d in days])[None,...]) # fix
			else:
				metric_curves.append(np.array([rdict[d]['metrics_dict'][metric_name] for d in days])[None,...])

			#print([rdict[d]['*baccu*'] for d in days])
			#print([rdict[d]['*recall*'] for d in days])
			#print('*'*40)

		metric_curves = np.concatenate(metric_curves, axis=0)
		samples.append(len(metric_curves))
		metrics_dict[model_name] = dstats.XError(metric_curves, 0) if return_xerror else metric_curves

	if not np.all(np.array(samples)==samples[0]):
		warnings.warn(f'not same samples: {samples}')

	return metrics_dict

def get_metrics_from_models(root_folder, model_names, metric_name, fext, # fix
	error_scale=1,
	):
	metrics_dict = {}
	samples = []
	for kmn,model_name in enumerate(model_names):
		print(f'model_name: {model_name}')
		filedirs = search_for_filedirs(f'{root_folder}/{model_name}', fext=fext, verbose=0)
		metric_curves = []
		for filedir in filedirs:
			rdict = load_pickle(filedir, verbose=0)
			#print(rdict.keys())
			metric_curves.append(np.array(rdict[metric_name])[None,...])

		metric_curves = np.concatenate(metric_curves, axis=0)
		samples.append(len(metric_curves))
		xe = dstats.XError(metric_curves, 0, error_scale=error_scale)
		metrics_dict[model_name] = xe

	if not np.all(np.array(samples)==samples[0]):
		warnings.warn(f'not same samples: {samples}')

	return metrics_dict

def get_info_from_models(root_folder, model_names, fext,
	):
	for kmn,model_name in enumerate(model_names):
		filedirs = search_for_filedirs(f'{root_folder}/{model_name}', fext=fext, verbose=0)
		for filedir in filedirs:
			rdict = load_pickle(filedir, verbose=0)
			survey = rdict['survey']
			band_names = ''.join(rdict['band_names'])
			class_names = rdict['class_names']
			return survey, band_names, class_names