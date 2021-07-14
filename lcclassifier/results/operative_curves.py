from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import fuzzytools.files as ftfiles
import fuzzytools.strings as strings
from fuzzytools.matplotlib.lines import fill_beetween
import matplotlib.pyplot as plt
from fuzzytools.datascience.xerror import XError
from . import utils as utils
from matplotlib import cm
from lchandler.C_ import CLASSES_STYLES

PERCENTILE_PLOT = 95
RECT_PLOT_2X1 = (16, 8)
SHADOW_ALPHA = 0.25

XLABEL_DICT = {
	'rocc':'fpr',
	'prc':'recall',
	}

YLABEL_DICT = {
	'rocc':'tpr',
	'prc':'precision',
	}

GUIDE_CURVE_DICT = {
	'rocc':[0,1],
	'prc':[1,0],
	}

RANDOM_STATE = 0

################################################################################################################

def plot_ocurve_classes(model_names, target_classes, rootdir, cfilename, kf, lcset_name, target_day,
	baselines_dict={},
	figsize=RECT_PLOT_2X1,
	train_mode='fine-tuning',
	percentile=PERCENTILE_PLOT,
	shadow_alpha=SHADOW_ALPHA,
	ocurve_name='rocc',
	):
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	for target_class in target_classes:
		ps_model_names = utils.get_sorted_model_names(model_names, merged=False)
		for kax,ax in enumerate(axs):
			if len(ps_model_names[kax])==0:
				continue
			color_dict = utils.get_color_dict(ps_model_names[kax])
			for kmn,model_name in enumerate(ps_model_names[kax]):
				load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
				files, files_ids = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
					fext='d',
					disbalanced_kf_mode='oversampling', # error oversampling
					random_state=RANDOM_STATE,
					)
				print(f'{model_name} {files_ids}({len(files_ids)}#)')
				if len(files)==0:
					continue

				survey = files[0]()['survey']
				band_names = files[0]()['band_names']
				class_names = files[0]()['class_names']
				days = files[0]()['days']

				xe_aucroc = XError([f()['days_class_metrics_cdf'][target_class].loc[f()['days_class_metrics_cdf'][target_class]['_day']==target_day]['aucroc'].item() for f in files])
				label = f'{target_class} | AUC={xe_aucroc}'
				# label = f'{utils.get_fmodel_name(model_name)} | AUC={xe_aucroc}'
				# color = color_dict[utils.get_fmodel_name(model_name)]
				color = CLASSES_STYLES[target_class]['c']

				ocurves = [f()['days_class_metrics_cdf'][target_class].loc[f()['days_class_metrics_cdf'][target_class]['_day']==target_day][ocurve_name].item() for f in files]
				fill_beetween(ax, [ocurve[XLABEL_DICT[ocurve_name]] for ocurve in ocurves], [ocurve[YLABEL_DICT[ocurve_name]] for ocurve in ocurves],
					fill_kwargs={'color':color, 'alpha':shadow_alpha, 'lw':0,},
					median_kwargs={'color':color, 'alpha':1,},
					percentile=percentile,
					)
				ax.plot([None], [None], color=color, label=label)

		title = ''
		title += f'{target_class}-{ocurve_name.upper()} curve ({target_day:.3f} [days])'+'\n'
		title += f'train-mode={train_mode} - survey={survey}-{"".join(band_names)} [{kf}@{lcset_name}]'+'\n'
		fig.suptitle(title[:-1], va='bottom')

	for kax,ax in enumerate(axs):
		ax.plot([0, 1], GUIDE_CURVE_DICT[ocurve_name], '--', color='k', alpha=1, lw=1)
		ax.set_xlabel(XLABEL_DICT[ocurve_name])
		if kax==0:
			ax.set_ylabel(YLABEL_DICT[ocurve_name])
			ax.set_title('parallel models')
		else:
			ax.set_yticklabels([])
			ax.set_title('serial models')

		ax.set_xlim(0.0, 1.0)
		ax.set_ylim(0.0, 1.0)
		ax.grid(alpha=.5);ax.set_axisbelow(True)
		ax.legend(loc='lower right')

	fig.tight_layout()
	plt.show()

def plot_ocurve_models(rootdir, cfilename, kf, lcset_name, model_names, target_class, target_day,
	baselines_dict={},
	figsize=RECT_PLOT_2X1,
	train_mode='fine-tuning',
	percentile=PERCENTILE_PLOT,
	shadow_alpha=SHADOW_ALPHA,
	ocurve_name='rocc',
	):
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	ps_model_names = utils.get_sorted_model_names(model_names, merged=False)
	for kax,ax in enumerate(axs):
		if len(ps_model_names[kax])==0:
			continue
		color_dict = utils.get_color_dict(ps_model_names[kax])
		for kmn,model_name in enumerate(ps_model_names[kax]):
			load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
			files, files_ids = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
				fext='d',
				disbalanced_kf_mode='oversampling', # error oversampling
				random_state=RANDOM_STATE,
				)
			print(f'{model_name} {files_ids}({len(files_ids)}#)')
			if len(files)==0:
				continue

			survey = files[0]()['survey']
			band_names = files[0]()['band_names']
			class_names = files[0]()['class_names']
			days = files[0]()['days']

			xe_aucroc = XError([f()['days_class_metrics_cdf'][target_class].loc[f()['days_class_metrics_cdf'][target_class]['_day']==target_day]['aucroc'].item() for f in files])
			label = f'{utils.get_fmodel_name(model_name)} | AUC={xe_aucroc}'
			color = color_dict[utils.get_fmodel_name(model_name)]

			roccs = [f()['days_class_metrics_cdf'][target_class].loc[f()['days_class_metrics_cdf'][target_class]['_day']==target_day][ocurve_name].item() for f in files]
			fill_beetween(ax, [rocc['fpr'] for rocc in roccs], [rocc['tpr'] for rocc in roccs],
				fill_kwargs={'color':color, 'alpha':shadow_alpha, 'lw':0,},
				median_kwargs={'color':color, 'alpha':1,},
				percentile=percentile,
				)
			ax.plot([None], [None], color=color, label=label)

		title = ''
		title += f'{target_class}-ROC curve ({target_day:.3f} [days])'+'\n'
		title += f'train-mode={train_mode} - survey={survey}-{"".join(band_names)} [{kf}@{lcset_name}]'+'\n'
		fig.suptitle(title[:-1], va='bottom')

	for kax,ax in enumerate(axs):
		ax.plot([0, 1], [0, 1], '--', color='k', alpha=1, lw=1)
		ax.set_xlabel('FPR')
		if kax==0:
			ax.set_ylabel('TPR')
			ax.set_title('parallel models')
		else:
			ax.set_yticklabels([])
			ax.set_title('serial models')

		ax.set_xlim(0.0, 1.0)
		ax.set_ylim(0.0, 1.0)
		ax.grid(alpha=0.5)
		ax.legend(loc='lower right')

	fig.tight_layout()
	plt.show()