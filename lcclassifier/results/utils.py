from __future__ import print_function
from __future__ import division
from . import C_

import fuzzytools.strings as strings
import fuzzytools.cuteplots.colors as cc
import numpy as np

###################################################################################################################################################

def get_fmodel_name(model_name):
	mn_dict = strings.get_dict_from_string(model_name)
	mdl = mn_dict['mdl']
	te_dims = int(mn_dict.get('te-dims', 0))
	cell = mn_dict.get('cell', None)
	model_name = []
	model_name += [mdl]
	model_name += [f'w/ {cell}' if not cell is None else '']
	model_name += [f'w/ M={te_dims//2}' if te_dims>0 else '']
	return ' '.join(model_name)

###################################################################################################################################################

def get_sorted_model_names(model_names,
	merged=True,
	):
	p_model_names = []
	s_model_names = []
	for model_name in model_names:
		is_parallel = 'Parallel' in model_name
		if is_parallel:
			p_model_names += [model_name]
		else:
			s_model_names += [model_name]
	p_model_names = sorted(p_model_names)
	s_model_names = sorted(s_model_names)
	return p_model_names+s_model_names if merged else p_model_names, s_model_names

###################################################################################################################################################

def get_cmetric_name(metric_name):
	#metric_name = metric_name.replace('accuracy', 'acc')
	#metric_name = metric_name.replace('f1score', 'f1s')
	return metric_name

def get_mday_str(metric_name, day_to_metric):
	new_metric_name = get_cmetric_name(metric_name)
	return new_metric_name+'$|^{'+str(day_to_metric)+'}$'

def get_mday_avg_str(metric_name, day_to_metric,
	first_day=2,
	):
	new_metric_name = get_cmetric_name(metric_name)
	return new_metric_name+'$|_{'+str(first_day)+'}'+'^{'+str(day_to_metric)+'}$'
	
###################################################################################################################################################

def filter_models(model_names, condition_dict):
	new_model_names = []
	for model_name in model_names:
		mn_dict = strings.get_dict_from_string(model_name)
		conds = []
		for c in condition_dict.keys():
			value = mn_dict.get(c, None)
			acceptable_values = condition_dict[c]
			conds += [value in acceptable_values]
		
		if all(conds):
			new_model_names += [model_name]

	return new_model_names

def get_color_dict(model_names):
	cmodel_names = []
	for kmn,model_name in enumerate(model_names):
		#if 'rsc=0' in model_name:
		cmodel_names += [get_fmodel_name(model_name)]

	cmodel_names = list(set(cmodel_names))
	colors = cc.colors()
	#colors = cc.get_colorlist('seaborn', len(cmodel_names))
	color_dict = {}
	for kmn,cmodel_name in enumerate(cmodel_names):
		color_dict[cmodel_name] = colors[kmn]
		#print(f'model_name: {model_name}')

	#print(color_dict)
	return color_dict