from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import flamingchoripan.files as fcfiles
import flamingchoripan.strings as strings
from flamingchoripan.cuteplots.cm_plots import plot_custom_confusion_matrix
from flamingchoripan.cuteplots.animations import PlotAnimation
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from flamingchoripan.datascience.statistics import XError
from . import utils as utils

###################################################################################################################################################

def plot_metric(rootdir, cfilename, kf, lcset_name, model_names, metric_name,
	baselines_dict={},
	label_keys=[],
	figsize=C_.PLOT_FIGZISE_RECT,
	train_mode='fine-tuning',
	p=C_.P_PLOT,
	alpha=0.2,
	):
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	color_dict = utils.get_color_dict(model_names)

	for kmn,model_name in enumerate(model_names):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/exp=performance/{cfilename}/{kf}@{lcset_name}'
		files, files_ids = fcfiles.gather_files_by_id(load_roodir, fext='d')
		print(f'ids={files_ids}(n={len(files_ids)}#) - model={model_name}')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		days = files[0]()['days']
		mn_dict = strings.get_dict_from_string(model_name)
		rsc = mn_dict['rsc']
		mdl = mn_dict['mdl']
		is_parallel = 'Parallel' in mdl

		metric_curve = np.concatenate([f()['days_class_metrics_df'][metric_name].values[None] for f in files], axis=0)
		xe_metric_curve = XError(metric_curve)
		xe_metric_curve_avg = XError(np.mean(metric_curve, axis=-1))

		ax = axs[int(not is_parallel)]
		_label = strings.get_string_from_dict({k:mn_dict[k] for k in mn_dict.keys() if k in label_keys}, key_key_separator=' - ')
		label = f'{mdl} ({_label}) | {xe_metric_curve_avg}*'
		color = color_dict[utils.get_cmodel_name(model_name)] if rsc=='0' else 'k'
		ax.plot(days, xe_metric_curve.median, '--' if is_parallel else '-', label=label, c=color)
		ax.fill_between(days, getattr(xe_metric_curve, f'p{p}'), getattr(xe_metric_curve, f'p{100-p}'), alpha=alpha, fc=color)

	title = ''
	title += f'{metric_name} v/s days'+'\n'
	title += f'train-mode={train_mode} - survey={survey} [{kf}@{lcset_name}] - bands={"".join(band_names)}'+'\n'
	fig.suptitle(title[:-1], va='bottom')

	for kax,ax in enumerate(axs):
		is_accuracy = 'accuracy' in metric_name
		random_guess = 100./len(class_names)
		if is_accuracy:
			ax.plot(days, np.full_like(days, random_guess), ':', c='k', label=f'RandomGuess ({random_guess:.3f}%)', alpha=.5)

		if f'{kf}@{lcset_name}' in baselines_dict.keys():
			ax.plot(days, np.full_like(days, baselines_dict[f'{kf}@{lcset_name}'][metric_name]), ':', c='k', label=f'FATS & b-RF Baseline (day={days[-1]:.3f})')

		ax.set_xlabel('time [days]')
		if kax==1:
			ax.set_ylabel(None)
			ax.set_yticklabels([])
			ax.set_title('serial models')
		else:
			ax.set_ylabel(metric_name)
			ax.set_title('parallel models')

		ax.set_xlim([days.min(), days.max()])
		ax.set_ylim([random_guess*.95, 100] if is_accuracy else [0, 1])
		ax.grid(alpha=0.5)
		ax.legend(loc='lower right')

	fig.tight_layout()
	plt.show()

###################################################################################################################################################

def plot_cm(rootdir, cfilename, kf, lcset_name, model_names,
	label_keys=[],
	figsize=C_.PLOT_FIGZISE_RECT,
	train_mode='fine-tuning',
	export_animation=False,
	animation_duration=10,
	new_order_classes=['SNIa', 'SNIbc', 'allSNII', 'SLSN'],
	):
	for kmn,model_name in enumerate(model_names):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/exp=performance/{cfilename}/{kf}@{lcset_name}'
		files, files_ids = fcfiles.gather_files_by_id(load_roodir, fext='d')
		print(f'ids={files_ids}(n={len(files_ids)}#) - model={model_name}')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		days = files[0]()['days']
		mn_dict = strings.get_dict_from_string(model_name)
		rsc = mn_dict['rsc']
		mdl = mn_dict['mdl']
		is_parallel = 'Parallel' in mdl

		xe_dict = {}
		for metric_name in ['b-accuracy', 'b-f1score']:
			metric_curve = np.concatenate([f()['days_class_metrics_df'][metric_name].values[None] for f in files], axis=0)
			xe_metric_curve = XError(metric_curve)
			xe_metric_curve_avg = XError(np.mean(metric_curve, axis=-1))
			xe_dict[metric_name] = (xe_metric_curve, xe_metric_curve_avg)

		_label = strings.get_string_from_dict({k:mn_dict[k] for k in mn_dict.keys() if k in label_keys}, key_key_separator=' - ')
		label = f'{mdl} ({_label})'

		plot_animation = PlotAnimation(len(days), animation_duration, dummy=not export_animation)
		for kd,day in enumerate(days):
			f1score_xe = xe_dict['b-f1score'][0]
			accuracy_xe = xe_dict['b-accuracy'][0]
			title = ''
			title += f'{label}'+'\n'
			title += f'train-mode={train_mode} - survey={survey} [{kf}@{lcset_name}] - bands={"".join(band_names)}'+'\n'
			title += f'b-f1score={f1score_xe}'+'\n'
			title += f'b-accuracy={accuracy_xe}'+'\n'
			title += f'day={target_day:.3f}/{day_to_metric:.3f}'+'\n'
			cm_kwargs = {
				#'fig':fig,
				#'ax':ax,
				'title':title[:-1],
				'figsize':(6,5),
				'new_order_classes':new_order_classes,
			}
			cms = np.concatenate([f()['days_cm'][day][None] for f in files], axis=0)
			print(cms.shape)
			assert 0
			fig, ax = plot_custom_confusion_matrix(cms, class_names, **cm_kwargs)
			plot_animation.add_frame(fig)
			if kd<len(days)-1:
				plt.close(fig)
			else:
				plt.show()

		plot_animation.save(f'../temp/{model_name}.gif')