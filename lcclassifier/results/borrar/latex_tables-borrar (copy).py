from __future__ import print_function
from __future__ import division
from . import C_

from . import utils as utils
import numpy as np
from fuzzytools.myUtils.files import search_for_filedirs, load_pickle, get_dict_from_filedir
import fuzzytools.myUtils.strings as strings
import fuzzytools.dataScience.statistics as stats
from scipy.interpolate import interp1d
from fuzzytools.latexHelp.latex import LatexTable

###################################################################################################################################################

def latex_table_metrics_days(root_folder, target_days,
	fext='expmet',
	error_scale=1,
	):
	filedirs = search_for_filedirs(root_folder, fext=fext, verbose=0)
	model_names = sorted(list(set([f.split('/')[-2] for f in filedirs])))
	results_dict = {mn:{} for mn in model_names}
	for kax,metric_name in enumerate(['*baccu*', '*f1score*']):
		cmetric_name = metric_name.replace('*','')
		cmetric_name = cmetric_name.replace('baccu','baccu')
		metrics_dict = utils.get_day_metrics_from_models(root_folder, model_names, metric_name, fext, error_scale, return_xerror=False)
		days, survey, band_names, class_names = utils.get_info_from_models(root_folder, model_names, fext)
		for kmn,model_name in enumerate(model_names):
			for target_day in target_days:
				results_dict[model_name].update({
					f'{utils.latex_day(cmetric_name, target_day)}':stats.XError(utils.get_day_metric(days, metrics_dict[model_name], target_day)*100, 0),
				})

			### me > tuple
			for key in results_dict[model_name].keys():
				if isinstance(results_dict[model_name][key], stats.XError):
					results_dict[model_name][key] = (results_dict[model_name][key].mean, results_dict[model_name][key].std)

	latex_kwargs = {
		'delete_redundant_model_keys':False,
		'colored_criterium_row':True,
		'bold_criteriums':'max',
	}
	latexTable = LatexTable(results_dict, **latex_kwargs)
	label = 'results_columns'
	latexTable.print(label=label, centered=1)

def latex_table_metrics_mean(root_folder,
	fext='expmet',
	error_scale=1,
	):
	filedirs = search_for_filedirs(root_folder, fext=fext, verbose=0)
	model_names = sorted(list(set([f.split('/')[-2] for f in filedirs])))
	results_dict = {mn:{} for mn in model_names}
	for kax,metric_name in enumerate(['*precision*', '*recall*', '*f1score*', '*baccu*']):
		cmetric_name = metric_name.replace('*','')
		cmetric_name = cmetric_name.replace('baccu','baccu')
		metrics_dict = utils.get_day_metrics_from_models(root_folder, model_names, metric_name, fext, error_scale, return_xerror=False)
		days, survey, band_names, class_names = utils.get_info_from_models(root_folder, model_names, fext)
		for kmn,model_name in enumerate(model_names):
			nmodel_name = model_name
			results_dict[nmodel_name].update({
				f'{utils.latex_mean_days(cmetric_name, days[0], days[-1])}':stats.XError(np.mean(metrics_dict[nmodel_name], axis=1)*100, 0),
			})

			### me > tuple
			for key in results_dict[nmodel_name].keys():
				if isinstance(results_dict[nmodel_name][key], stats.XError):
					results_dict[nmodel_name][key] = (results_dict[nmodel_name][key].mean, results_dict[nmodel_name][key].std)

	latex_kwargs = {
		'delete_redundant_model_keys':False,
		'colored_criterium_row':True,
		'bold_criteriums':'max',
	}
	latexTable = LatexTable(results_dict, **latex_kwargs)
	label = 'results_columns'
	latexTable.print(label=label, centered=1)

def latex_table_parameters(root_folder,
	fext='exprec',
	error_scale=1,
	):
	filedirs = search_for_filedirs(root_folder, fext=fext, verbose=0)
	model_names = sorted(list(set([f.split('/')[-2] for f in filedirs])))
	results_dict = {mn:{} for mn in model_names}
	for kax,metric_name in enumerate(['parameters']):
		cmetric_name = metric_name.replace('','')
		metrics_dict = utils.get_metrics_from_models(root_folder, model_names, metric_name, fext, error_scale)
		days, survey, band_names, class_names = utils.get_info_from_models(root_folder, model_names, fext)
		for kmn,model_name in enumerate(model_names):
			results_dict[model_name].update({
				f'{cmetric_name}':metrics_dict[model_name],
			})

			### me > tuple
			for key in results_dict[model_name].keys():
				if isinstance(results_dict[model_name][key], stats.XError):
					results_dict[model_name][key] = (results_dict[model_name][key].mean, results_dict[model_name][key].std)

	for kax,metric_name in enumerate(['epoch_mins', 'convergence_mins']):
		cmetric_name = metric_name.replace('_',' ')
		metrics_dict = utils.get_metrics_from_models(root_folder, model_names, metric_name, fext, error_scale)
		days, survey, band_names, class_names = utils.get_info_from_models(root_folder, model_names, fext)
		for kmn,model_name in enumerate(model_names):
			results_dict[model_name].update({
				f'{cmetric_name} [mins]':metrics_dict[model_name],
			})

			### me > tuple
			for key in results_dict[model_name].keys():
				if isinstance(results_dict[model_name][key], stats.XError):
					results_dict[model_name][key] = (results_dict[model_name][key].mean, results_dict[model_name][key].std)

	latex_kwargs = {
		'delete_redundant_model_keys':False,
		'colored_criterium_row':True,
		'bold_criteriums':None,
	}
	latexTable = LatexTable(results_dict, **latex_kwargs)
	label = 'results_columns'
	latexTable.print(label=label, centered=1)