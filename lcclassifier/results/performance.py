from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import fuzzytools.files as fcfiles
import fuzzytools.strings as strings
from fuzzytools.matplotlib.lines import fill_beetween
from fuzzytools.matplotlib.lims import AxisLims
import matplotlib.pyplot as plt
from fuzzytools.datascience.xerror import XError
from . import utils as utils
from matplotlib import cm

PERCENTILE_PLOT = 95
RECT_PLOT_2X1 = (16, 8)
SHADOW_ALPHA = 0.25
RANDOM_STATE = 0

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
		axis_lims = AxisLims({'x':(None, None), 'y':(0, 1)}, {'x':.0, 'y':.1})
		ps_model_names = utils.get_sorted_model_names(model_names, merged=False)
		for kax,ax in enumerate(axs):
			if len(ps_model_names[kax])==0:
				continue
			color_dict = utils.get_color_dict(ps_model_names[kax])
			for kmn,model_name in enumerate(ps_model_names[kax]):
				load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
				files, files_ids = fcfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
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
				axis_lims.append('x', days)
				axis_lims.append('y', np.concatenate([metric_curve for metric_curve in metric_curves], axis=0))

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
			if kax==0:
				ax.set_ylabel(mn)
				ax.set_title('(a) parallel models')
			else:
				ax.set_yticklabels([])
				ax.set_title('(b) serial models')

			axis_lims.set_ax_axis_lims(ax)
			ax.grid(alpha=0.5)
			ax.legend(loc='lower right')

		fig.tight_layout()
		plt.show()