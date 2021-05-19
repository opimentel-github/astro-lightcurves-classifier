from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import fuzzytools.files as fcfiles
import fuzzytools.strings as strings
from fuzzytools.cuteplots.cm_plots import plot_custom_confusion_matrix
from fuzzytools.cuteplots.animations import PlotAnimation
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from fuzzytools.datascience.statistics import XError
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
			f1score_xe = xe_dict['b-accuracy'][0]
			accuracy_xe = xe_dict['b-f1score'][0]
			title = ''
			title = f'{label}'
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

'''

###################################################################################################################################################

def plot_mse(rootdir, model_names,
	figsize=C_.PLOT_FIGZISE_RECT,
	fext='metrics',
	set_name='???',
	):
	fig, ax = plt.subplots(1, 1, figsize=figsize)
	color_dict = utils.get_color_dict(model_names)
	for kmn,model_name in enumerate(model_names):
		new_rootdir = f'{rootdir}/{mode}/{model_name}'
		new_rootdir = new_rootdir.replace('mode=pre-training', f'mode={mode}') # patch
		new_rootdir = new_rootdir.replace('mode=fine-tuning', f'mode={mode}') # patch
		filedirs = search_for_filedirs(new_rootdir, fext=fext, verbose=0)
		print(f'[{kmn}] {model_name} (iters: {len(filedirs)})')
		mn_dict = strings.get_dict_from_string(model_name)
		rsc = mn_dict['rsc']
		mdl = mn_dict['mdl']
		is_parallel = 'Parallel' in mdl

		metric_curve = []
		for filedir in filedirs:
			rdict = load_pickle(filedir, verbose=0)
			#model_name = rdict['model_name']
			days = rdict['days']
			survey = rdict['survey']
			band_names = ''.join(rdict['band_names'])
			class_names = rdict['class_names']
			metric_curve += [rdict['days_rec_metrics_df']['mse'].values[:][None,:]]

		metric_curve = np.concatenate(metric_curve, axis=0)
		xe_metric_curve = XError(np.log(metric_curve), 0)
		label = f'{mdl} {rsc}'
		color = color_dict[utils.get_cmodel_name(model_name)]
		ax.plot(days, xe_metric_curve.median, '--' if is_parallel else '-', label=label, c=color)
		ax.fill_between(days, xe_metric_curve.p15, xe_metric_curve.p85, alpha=0.25, fc=color)

	title = 'log-reconstruction-wmse v/s days\n'
	title += f'survey: {survey} - bands: {band_names}\n'
	ax.set_title(title[:-1])
	ax.set_xlabel('days')
	ax.set_ylabel('mse')
	ax.set_xlim([days.min(), days.max()])
	ax.grid(alpha=0.5)
	ax.legend(loc='upper right')
	plt.show()


def plot_f1score_mse(rootdir,
	figsize=(10,6),
	error_scale=1,
	):
	filedirs = search_for_filedirs(root_folder, fext='exprec', verbose=0)
	model_names = sorted(list(set([f.split('/')[-2] for f in filedirs])))
	colors = cc.colors()

	fig, ax = plt.subplots(1, 1, figsize=figsize)
	metrics_dict_mse = utils.get_day_metrics_from_models(root_folder, model_names, 'mse', 'exprec', error_scale, return_xerror=False)
	metrics_dict_f1 = utils.get_day_metrics_from_models(root_folder, model_names, '*f1score*', 'expmet', error_scale, return_xerror=False)
	days, survey, band_names, class_names = utils.get_info_from_models(root_folder, model_names, 'expmet')
	xy = []
	for kmn,model_name in enumerate(model_names):
		mse_xe = dstats.XError(metrics_dict_mse[model_name].mean(axis=1), 0)
		f1_xe = dstats.XError(metrics_dict_f1[model_name].mean(axis=1), 0)
		for i in range(len(mse_me)):
			try:
				x = mse_xe.x[i]
				y = f1_xe.x[i]
				xy.append([x, y])
				ax.scatter(x, y,
					facecolors=[colors[kmn]],
					edgecolors='k',
					s=45,
					alpha=1,
					marker='o' if 'teD-0' in model_name else 'o',
					lw=1.5,
					linewidth=1 if 'teD-0' in model_name else 0,
					#linewidth=0.0,
					label=f'{utils.formating_model_name(model_name)}' if i==0 else None,
				)
			except:
				pass

	### regression
	xy = np.array(xy)
	new_x = np.linspace(xy[:,0].min(), xy[:,0].max(), 100)

	p = np.polyfit(xy[:,0], xy[:,1], deg=1)
	#ax.plot(new_x, p[0]*new_x+p[1], 'k', label='Linear Regression1', lw=1, alpha=0.75)

	p = np.polyfit(xy[:,0], xy[:,1], deg=2)
	ax.plot(new_x, p[0]*new_x**2+p[1]*new_x+p[2], 'k', label='Linear Regression2', lw=1, alpha=0.75)

	title = 'f1score v/s reconstruction mse'
	title += f'\nsurvey: {survey} - bands: {band_names}'
	ax.set_title(title)
	ax.set_xlabel(utils.latex_mean_days('mse', days[0], days[-1]), fontsize=C_.FONTSIZE)
	ax.set_ylabel(utils.latex_mean_days('f1score', days[0], days[-1]), fontsize=C_.FONTSIZE)
	ax.set_xlim(ax.get_xlim()[::-1])
	ax.set_ylim([0, 1])
	ax.grid(alpha=0.5)
	ax.legend(loc='upper left')
	plt.show()

###################################################################################################################################################

def plot_te_scores(root_folder,
	figsize=C_.PLOT_FIGZISE_RECT,
	error_scale=1,
	):
	filedirs = search_for_filedirs(root_folder, fext='exprec', verbose=0)
	model_names = sorted(list(set([f.split('/')[-2] for f in filedirs])))
	colors = ccolors.colors()

	fig, axs = plt.subplots(1, 2, figsize=figsize)
	for kax,(metric_name,fext) in enumerate(zip(['mse', '*f1score*'], ['exprec', 'expmet'])):
		ax = axs[kax]
		metrics_dict = utils.get_day_metrics_from_models(root_folder, model_names, metric_name, fext, error_scale, return_xerror=False)
		days, survey, band_names, class_names = utils.get_info_from_models(root_folder, model_names, 'expmet')
		tes = ['0', '2', '4', '8']
		rdict = {m:{t:0 for t in tes} for m in ['ParallelRNN', 'SerialRNN']}
		for kmn,model_name in enumerate(model_names):
			mn_dict = strings.get_dict_from_string(model_name)
			if mn_dict['teD'] in tes:
				me = dstats.XError(metrics_dict[model_name].mean(axis=-1), 0)
				rdict[mn_dict['mdl']][mn_dict['teD']] = me

		for k,mdl in enumerate(rdict.keys()):
			arr_tes = np.array([int(t) for t in tes])
			mean = np.array([rdict[mdl][t].mean for t in tes])
			std = np.array([rdict[mdl][t].std for t in tes])
			ax.plot(arr_tes, mean, '-o', c=colors[k], label=f'{mdl}')
			ax.fill_between(arr_tes, mean-std, mean+std, alpha=0.25, fc=colors[k])

		title = 'f1score v/s reconstruction mse'
		title += f'\nsurvey: {survey} - bands: {band_names}'
		ax.set_title(title)
		ax.set_xlabel('temporal encoding dims')
		ax.set_ylabel(utils.latex_mean_days(metric_name.replace('__',''), days[0], days[-1]), fontsize=C_.FONTSIZE)
		ax.grid(alpha=0.5)
		ax.legend(loc='upper left')

	plt.show()

###################################################################################################################################################
	
def plot_precision_recall_classes(root_folder, model_name,
	figsize=C_.PLOT_FIGZISE_RECT,
	fext='expmet',
	error_scale=1,
	ylims=(0,1),
	):
	print(f'model_name: {model_name}')
	mn_dict = strings.get_dict_from_string(model_name)
	filedirs = search_for_filedirs(f'{root_folder}/{model_name}', fext=fext, verbose=0)
	colors = ccolors.colors()
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	for kax,metric_name in enumerate(['precision', 'recall']):
		ax = axs[kax]
		samples = []
		class_names = load_pickle(filedirs[0], verbose=0)['class_names']
		for kc,c in enumerate(class_names):
			metric_curves = []
			for filedir in filedirs:
				filedict = get_dict_from_filedir(filedir)
				rdict = load_pickle(filedir, verbose=0)
				survey = rdict['survey']
				band_names = ''.join(rdict['band_names'])
				class_names = rdict['class_names']
				days = np.array(rdict['days'])
				metric_curves.append(np.array([rdict[d][metric_name][c] for d in days])[None,...])
				
			metric_curves = np.concatenate(metric_curves, axis=0)
			xe = dstats.XError(metric_curves, 0, error_scale=error_scale)
			samples.append(len(me))
			class_population = rdict[days[-1]]['cm'].sum(axis=-1)[kc]
			ax.plot(days, xe.median, '--' if int(mn_dict['teD'])==0 else '-', label=f'{c} - population: {class_population:,}', c=colors[kc])
			ax.fill_between(days, xe.p5, xe.p95, alpha=0.25, fc=colors[kc])

		if not np.all(np.array(samples)==samples[0]):
			warnings.warn(f'not same samples: {samples}')

		cmetric_name = metric_name.replace('__','')
		title = f'{cmetric_name} v/s days'
		title += f'\n{utils.formating_model_name(model_name)}'
		title += f'\nsurvey: {survey} - bands: {band_names}'
		title += f'\nshadow region: {xe.get_symbol("std")} ({len(xe)} itrs)'
		ax.set_title(title)
		ax.set_xlabel('days')
		ax.set_ylabel(cmetric_name)
		ax.set_xlim([days.min(), days.max()])
		ax.set_ylim(ylims)
		ax.grid(alpha=0.5)
		ax.legend(loc='lower right')

	fig.tight_layout()
	plt.show()

###################################################################################################################################################

def plot_training_losses(root_folder,
	figsize=(14,6),
	fext='exprec',
	error_scale=1,
	):
	filedirs = search_for_filedirs(root_folder, fext=fext, verbose=0)
	model_names = sorted(list(set([f.split('/')[-2] for f in filedirs])))
	colors = ccolors.colors()

	fig, ax = plt.subplots(1, 1, figsize=figsize)
	samples = []
	lengths = []
	for kmn,model_name in enumerate(model_names):
		mn_dict = strings.get_dict_from_string(model_name)
		if not int(mn_dict['teD']) in [0,4]:
			continue
		print(f'model_name: {model_name}')
		filedirs = search_for_filedirs(f'{root_folder}/{model_name}', fext=fext, verbose=0)
		metric_curves = []
		for filedir in filedirs:
			filedict = get_dict_from_filedir(filedir)
			rdict = load_pickle(filedir, verbose=0)
			survey = rdict['survey']
			band_names = ''.join(rdict['band_names'])
			loss = np.array(rdict['history_dict']['finalloss_evolution_k']['train'])[200:-1000][::5]
			lengths.append(len(loss))
			metric_curves.append(loss)
			
		metric_curves = np.concatenate([mc[:min(lengths)][None] for mc in metric_curves], axis=0)
		xe = dstats.XError(metric_curves, 0, error_scale=error_scale)
		samples.append(len(me))
		iters = np.arange(0, min(lengths))
		ax.plot(iters, gaussian_filter1d(xe.median, sigma=3), '--' if int(mn_dict['teD'])==0 else '-', label=f'{utils.formating_model_name(model_name)}', c=colors[kmn], lw=2)
		ax.plot(iters, xe.median, '-', c=colors[kmn], alpha=0.15)
		#ax.fill_between(iters, xe.p5, xe.p95, alpha=0.8, fc=colors[kmn])
		
	if not np.all(np.array(samples)==samples[0]):
		warnings.warn(f'not same samples: {samples}')
		
	title = 'reconstruction mse loss (autoencoder)'
	title += f'\nsurvey: {survey} - bands: {band_names}'
	title += f'\nshadow region: {xe.get_symbol("std")} ({len(xe)} itrs)'
	ax.set_title(title)
	ax.set_xlabel('training iterations')
	ax.set_ylabel('reconstruction mse error')
	ax.set_xlim([0, None])
	ax.grid(alpha=0.5)
	ax.legend(loc='upper right')
	plt.show()

###################################################################################################################################################

def get_new_order_classes(class_names):
	new_classes = []
	for nc in ['SNIa', 'SNIbc', 'SNII', 'SLSN']:
		for c in class_names:
			if nc in c:
				new_classes.append(c)
				break
	return new_classes

def plot_cm(root_folder, model_name, target_day,
	fext='expmet',
	return_fig=False,
	verbose=1,
	):
	if verbose==1:
		print(f'model_name: {model_name}')
	filedirs = search_for_filedirs(f'{root_folder}/{model_name}', fext=fext, verbose=0)

	cms = []
	for filedir in filedirs:
		rdict = load_pickle(filedir, verbose=0)
		class_names = rdict['class_names']
		band_names = ''.join(rdict['band_names'])
		survey = rdict['survey']
		cm = rdict[target_day]['cm']
		#print('cm',cm)
		cms.append(cm[None])

	cms = np.concatenate(cms, axis=0)
	title = f'confusion matrix - day: {target_day:.2f}'
	title += f'\n{utils.formating_model_name(model_name)}'
	title += f'\nsurvey: {survey} - bands: {band_names}'
	cm_kwargs = {
		'figsize':C_.PLOT_FIGZISE_CM,
		'title':title,
		'add_accuracy_in_title':0,
		'new_order_classes':get_new_order_classes(class_names),
		'add_accuracy_in_title':1,
	}
	fig, ax = plot_custom_confusion_matrix(cms, class_names, **cm_kwargs)
	if return_fig:
		return fig
	return None

def animation_cm(root_folder, model_name, target_days,
	animation_time:float=10,
	):
	print(f'model_name: {model_name}')
	fps = len(target_days)/animation_time
	animation = PlotAnimation(len(target_days), fps)
	for target_day in target_days:
		fig = plot_cm(root_folder, model_name, target_day, return_fig=True, verbose=0)
		animation.add_frame(fig)
		plt.close()

	video_save_dir = f'{root_folder}/{model_name}'
	video_save_cfilename = f'cm'
	animation.save(video_save_dir, video_save_cfilename)

'''