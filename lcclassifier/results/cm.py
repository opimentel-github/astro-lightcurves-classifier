from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import fuzzytools.files as ftfiles
import fuzzytools.strings as strings
import matplotlib.pyplot as plt
from fuzzytools.datascience.xerror import XError
from . import utils as utils
from matplotlib import cm
from fuzzytools.cuteplots.cm_plots import plot_custom_confusion_matrix
from fuzzytools.cuteplots.animators import PlotAnimator
from fuzzytools.progress_bars import ProgressBar

FIGSIZE = (6,5)
RANDOM_STATE = 0
PERCENTILE = 75

###################################################################################################################################################

def plot_cm(rootdir, cfilename, kf, lcset_name, model_names,
	figsize=FIGSIZE,
	train_mode='fine-tuning',
	export_animation=False,
	animation_duration=12,
	new_order_classes=['SNIa', 'SNIbc', 'SNII-b-n', 'SLSN'],
	percentile=PERCENTILE,
	):
	for kmn,model_name in enumerate(model_names):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
		files, files_ids = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
			fext='d',
			disbalanced_kf_mode='oversampling', # error oversampling
			random_state=RANDOM_STATE,
			)
		print(f'ids={files_ids}(n={len(files_ids)}#); model={model_name}')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		is_parallel = 'Parallel' in model_name
		days = files[0]()['days']

		plot_animation = PlotAnimator(animation_duration,
			is_dummy=not export_animation,
			#save_init_frame=True,
			save_end_frame=True,
			)

		target_days = days if export_animation else [days[-1]]
		bar = ProgressBar(len(target_days), bar_format='{l_bar}{bar}{postfix}')
		for kd,target_day in enumerate(target_days):
			bar(f'{target_day:.1f}/{target_days[-1]:.1f} [days]')
			xe_dict = {}
			for metric_name in ['b-precision', 'b-recall', 'b-f1score']:
				xe_metric = XError([f()['days_class_metrics_df'].loc[f()['days_class_metrics_df']['_day']==target_day][metric_name].item() for f in files])
				xe_dict[metric_name] = xe_metric

			bprecision_xe = xe_dict['b-precision']
			brecall_xe = xe_dict['b-recall']
			bf1score_xe = xe_dict['b-f1score']

			title = ''
			title += f'{utils.get_fmodel_name(model_name)}'+'\n'
			#title += f'survey={survey}-{"".join(band_names)} [{kf}@{lcset_name}]'+'\n'
			#title += f'train-mode={train_mode}; eval-set={kf}@{lcset_name}'+'\n'
			title += f'b-recall={brecall_xe}; b-f1score={bf1score_xe}'+'\n'
			title += f'th-day={target_day:.3f} [days]'+'\n'
			#title += f'b-p/r={bprecision_xe} / {brecall_xe}'+'\n'
			#title += f'b-f1score={bf1score_xe}'+'\n'
			if export_animation:
				#title += str(bar)+'\n'
				title += f'target_day={target_day:.3f}/{days[-1]:.3f} [days]'+'\n'
				pass
			cms = np.concatenate([f()['days_cm'][target_day][None] for f in files], axis=0)
			fig, ax, cm_norm = plot_custom_confusion_matrix(cms, class_names,
				#fig=fig,
				#ax=ax,
				title=title[:-1],
				figsize=figsize,
				new_order_classes=new_order_classes,
				percentile=percentile,
				)
			uses_close_fig = kd<len(days)-1
			plot_animation.append(fig, uses_close_fig)

		bar.done()
		plt.show()
		plot_animation.save(f'../temp/{model_name}.gif') # gif mp4