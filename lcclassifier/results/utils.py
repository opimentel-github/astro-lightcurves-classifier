from __future__ import print_function
from __future__ import division
from . import C_

from flamingchoripan.files import search_for_filedirs
import flamingchoripan.strings as strings
import flamingchoripan.cuteplots.colors as cc

###################################################################################################################################################
	
def get_cmodel_name(model_name):
	mn_dict = strings.get_dict_from_string(model_name)
	mn_dict.pop('enc-emb')
	cmodel_name = '°'.join([f'{k}={mn_dict[k]}' for k in mn_dict.keys()])
	cmodel_name = cmodel_name.replace('Parallel', '').replace('Serial', '')
	return cmodel_name

def get_models_from_rootdir(rootdir,
	fext='metrics',
	):
	filedirs = search_for_filedirs(rootdir, fext=fext, verbose=0)
	model_names = sorted(list(set([f.split('/')[-2] for f in filedirs])))
	return model_names

def filter_models(model_names, condition_dict):
	new_model_names = []
	for model_name in model_names:
		mn_dict = strings.get_dict_from_string(model_name)
		for c in condition_dict.keys():
			value = mn_dict.get(c, None)
			acceptable_values = condition_dict[c]
			if value in acceptable_values:
				new_model_names += [model_name]

	return new_model_names

def get_color_dict(model_names):
	cmodel_names = []
	for kmn,model_name in enumerate(model_names):
		cmodel_names += [get_cmodel_name(model_name)]

	cmodel_names = list(set(cmodel_names))
	colors = cc.colors()
	#colors = cc.get_colorlist('seaborn', len(cmodel_names))
	color_dict = {}
	for kmn,cmodel_name in enumerate(cmodel_names):
		color_dict[cmodel_name] = colors[kmn]
		#print(f'model_name: {model_name}')

	return color_dict