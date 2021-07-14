from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import fuzzytools.files as fcfiles
import fuzzytools.strings as strings
from fuzzytools.matplotlib.lines import fill_beetween
import matplotlib.pyplot as plt
from fuzzytools.datascience.xerror import XError
from . import utils as utils
from matplotlib import cm

PERCENTILE_PLOT = 95
RECT_PLOT_2X1 = (16, 8)
SHADOW_ALPHA = 0.25

###################################################################################################################################################

def plot_metric(rootdir, cfilename, kf, lcset_name, model_names, dmetrics,
	target_class=None,
	baselines_dict={},
	figsize=RECT_PLOT_2X1,
	train_mode='fine-tuning',
	percentile=PERCENTILE_PLOT,
	shadow_alpha=SHADOW_ALPHA,
	):
	for metric_name in dmetrics.keys():
		fig, axs = plt.subplots(1, 2, figsize=figsize)
		ps_model_names = utils.get_sorted_model_names(model_names, merged=False)
		for kax,ax in enumerate(axs):
			if len(ps_model_names[kax])==0:
				continue
			color_dict = utils.get_color_dict(ps_model_names[kax])
			ylims = [[],[]]
			for kmn,model_name in enumerate(ps_model_names[kax]):
				load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
				files, files_ids = fcfiles.gather_files_by_kfold(load_roodir, kf, lcset_name, fext='d')
				print(f'{model_name} {files_ids}({len(files_ids)}#)')
				if len(files)==0:
					continue

				survey = files[0]()['survey']
				band_names = files[0]()['band_names']
				class_names = files[0]()['class_names']
				days = files[0]()['days']

				if target_class is None:
					metric_curves = [f()['days_class_metrics_df'][metric_name].values for f in files]
				else:
					metric_curves = [f()['days_class_metrics_cdf'][target_class][metric_name.replace('b-', '')].values for f in files] 
				xe_metric_curve_avg = XError(np.mean(np.concatenate([metric_curve[None] for metric_curve in metric_curves], axis=0), axis=-1))

				label = f'{utils.get_fmodel_name(model_name)} | AUC={xe_metric_curve_avg}'
				color = color_dict[utils.get_fmodel_name(model_name)]
				fill_beetween(ax, [days for metric_curve in metric_curves], [metric_curve for metric_curve in metric_curves],
					fill_kwargs={'color':color, 'alpha':shadow_alpha, 'lw':0,},
					median_kwargs={'color':color, 'alpha':1,},
					percentile=percentile,
					)
				ax.plot([None], [None], color=color, label=label)
				ylims[0] += [ax.get_ylim()[0]]
				ylims[1] += [ax.get_ylim()[1]]

			mn = metric_name if dmetrics[metric_name]['mn'] is None else dmetrics[metric_name]['mn']
			mn = mn if target_class is None else mn.replace('b-', f'{target_class}-')
			title = ''
			title += f'{mn} v/s days'+'\n'
			title += f'train-mode={train_mode} - survey={survey}-{"".join(band_names)} [{kf}@{lcset_name}]'+'\n'
			fig.suptitle(title[:-1], va='bottom')

		for kax,ax in enumerate(axs):
			if f'{kf}@{lcset_name}' in baselines_dict.keys():
				# ax.plot(days, np.full_like(days, baselines_dict[f'{kf}@{lcset_name}'][metric_name]), ':', c='k', label=f'FATS & b-RF Baseline (day={days[-1]:.3f})')
				pass

			ax.set_xlabel('time [days]')
			if kax==1:
				ax.set_yticklabels([])
				ax.set_title('serial models')
			else:
				ax.set_ylabel(mn)
				ax.set_title('parallel models')

			ax.set_xlim([days.min(), days.max()])
			ax.set_ylim(min(ylims[0]), max(ylims[1])*1.05)
			ax.grid(alpha=0.5)
			ax.legend(loc='lower right')

		fig.tight_layout()
		plt.show()

###################################################################################################################################################

def plot_rocc(rootdir, cfilename, kf, lcset_name, model_names, target_class, target_day,
	baselines_dict={},
	figsize=RECT_PLOT_2X1,
	train_mode='fine-tuning',
	percentile=PERCENTILE_PLOT,
	shadow_alpha=SHADOW_ALPHA,
	):
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	ps_model_names = utils.get_sorted_model_names(model_names, merged=False)
	for kax,ax in enumerate(axs):
		if len(ps_model_names[kax])==0:
			continue
		color_dict = utils.get_color_dict(ps_model_names[kax])
		for kmn,model_name in enumerate(ps_model_names[kax]):
			load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
			files, files_ids = fcfiles.gather_files_by_kfold(load_roodir, kf, lcset_name, fext='d')
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

			roccs = [f()['days_class_metrics_cdf'][target_class].loc[f()['days_class_metrics_cdf'][target_class]['_day']==target_day]['rocc'].item() for f in files]
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