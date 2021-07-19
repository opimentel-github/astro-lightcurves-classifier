from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import fuzzytools.files as ftfiles
import fuzzytools.matplotlib.lines as lines
from fuzzytools.matplotlib.lims import AxisLims
from matplotlib import cm
import matplotlib.pyplot as plt
from fuzzytools.datascience.xerror import XError
from . import utils as utils
from fuzzytools.strings import bf_alphabet_count

RANDOM_STATE = 0
PERCENTILE = 75
SHADOW_ALPHA = .25
FIGSIZE_2X1 = (16, 8)

###################################################################################################################################################

def plot_metric(rootdir, cfilename, kf, lcset_name, model_names, dmetrics,
	target_class=None,
	baselines_dict={},
	figsize=FIGSIZE_2X1,
	train_mode='fine-tuning',
	percentile=PERCENTILE,
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
				files, files_ids = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
					fext='d',
					disbalanced_kf_mode='oversampling', # error oversampling
					random_state=RANDOM_STATE,
					)
				print(f'{model_name} {files_ids}({len(files_ids)}#); model={model_name}')
				if len(files)==0:
					continue

				survey = files[0]()['survey']
				band_names = files[0]()['band_names']
				class_names = files[0]()['class_names']
				days = files[0]()['days']

				if target_class is None:
					metric_curves = [f()['days_class_metrics_df'][f'b-{metric_name}'].values for f in files]
				else:
					metric_curves = [f()['days_class_metrics_cdf'][target_class][f'{metric_name}'].values for f in files] 
				xe_metric_curve_avg = XError(np.mean(np.concatenate([metric_curve[None] for metric_curve in metric_curves], axis=0), axis=-1))

				model_label = utils.get_fmodel_name(model_name)
				label = f'{model_label}; AUC={xe_metric_curve_avg}'
				color = color_dict[utils.get_fmodel_name(model_name)]
				lines.fill_beetween(ax, [days for _ in metric_curves], metric_curves,
					fill_kwargs={'color':color, 'alpha':shadow_alpha, 'lw':0,},
					median_kwargs={'color':color, 'alpha':1,},
					percentile=percentile,
					)
				ax.plot([None], [None], color=color, label=label)
				axis_lims.append('x', days)
				axis_lims.append('y', np.concatenate([metric_curve for metric_curve in metric_curves], axis=0))

			new_metric_name = f'{target_class}-{metric_name if dmetrics[metric_name]["mn"] is None else dmetrics[metric_name]["mn"]}'
			suptitle = ''
			suptitle += f'{new_metric_name} v/s days using moving th-day'+'\n'
			suptitle += f'set={survey} [{lcset_name.replace(".@", "")}]'+'\n'
			fig.suptitle(suptitle[:-1], va='bottom')

		for kax,ax in enumerate(axs):
			if f'{kf}@{lcset_name}' in baselines_dict.keys():
				# ax.plot(days, np.full_like(days, baselines_dict[f'{kf}@{lcset_name}'][metric_name]), ':', c='k', label=f'FATS & b-RF Baseline (day={days[-1]:.3f})')
				pass

			ax.set_xlabel('time [days]')
			if kax==0:
				ax.set_ylabel(new_metric_name)
				ax.set_title(f'{bf_alphabet_count(0)} Parallel models')
			else:
				ax.set_yticklabels([])
				ax.set_title(f'{bf_alphabet_count(1)} Serial models')

			axis_lims.set_ax_axis_lims(ax)
			ax.grid(alpha=0.5)
			ax.legend(loc='lower right')

		fig.tight_layout()
		plt.show()