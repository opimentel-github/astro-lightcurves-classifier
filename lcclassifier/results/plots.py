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
from fuzzytools.datascience.xerror import XError
from . import utils as utils
from fuzzytools.progress_bars import ProgressBar
import math
from matplotlib import cm

###################################################################################################################################################

def plot_metric(rootdir, cfilename, kf, lcset_name, model_names, dmetrics,
	baselines_dict={},
	figsize=(14,8),
	train_mode='fine-tuning',
	p=C_.P_PLOT,
	alpha=0.2,
	):
	for metric_name in dmetrics.keys():
		fig, axs = plt.subplots(1, 2, figsize=figsize)
		ps_model_names = utils.get_sorted_model_names(model_names, merged=False)
		for kax,ax in enumerate(axs):
			color_dict = utils.get_color_dict(ps_model_names[kax])
			ylims = [[],[]]
			for kmn,model_name in enumerate(ps_model_names[kax]):
				load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
				files, files_ids = fcfiles.gather_files_by_kfold(load_roodir, kf, lcset_name, fext='d')
				print(f'ids={files_ids}(n={len(files_ids)}#) - model={model_name}')
				if len(files)==0:
					continue

				survey = files[0]()['survey']
				band_names = files[0]()['band_names']
				class_names = files[0]()['class_names']
				days = files[0]()['days']

				metric_curve = np.concatenate([f()['days_class_metrics_df'][metric_name].values[None] for f in files], axis=0)
				#metric_curve = np.concatenate([f()['days_class_metrics_cdf']['SNII-b-n'][metric_name].values[None] for f in files], axis=0) # SNIa SNIbc
				xe_metric_curve = XError(metric_curve)
				xe_metric_curve_avg = XError(np.mean(metric_curve, axis=-1))

				label = f'{utils.get_fmodel_name(model_name)} | AUC={xe_metric_curve_avg}'
				color = color_dict[utils.get_fmodel_name(model_name)]
				if p is None:
					for i in range(0, len(xe_metric_curve.x)):
						ax.plot(days, xe_metric_curve.x[i], '-', label=label, c=color)
				else:
					ax.plot(days, xe_metric_curve.median, '-', label=label, c=color)
					ax.fill_between(days, xe_metric_curve.get_percentile(p), xe_metric_curve.get_percentile(100-p), alpha=alpha, fc=color)
				ylims[0] += [ax.get_ylim()[0]]
				ylims[1] += [ax.get_ylim()[1]]

			mn = metric_name if dmetrics[metric_name]['mn'] is None else dmetrics[metric_name]['mn']
			title = ''
			title += f'{mn} v/s days'+'\n'
			title += f'train-mode={train_mode} - survey={survey}-{"".join(band_names)} [{kf}@{lcset_name}]'+'\n'
			fig.suptitle(title[:-1], va='bottom')

		for kax,ax in enumerate(axs):
			if f'{kf}@{lcset_name}' in baselines_dict.keys():
				ax.plot(days, np.full_like(days, baselines_dict[f'{kf}@{lcset_name}'][metric_name]), ':', c='k', label=f'FATS & b-RF Baseline (day={days[-1]:.3f})')

			ax.set_xlabel('time [days]')
			if kax==1:
				#ax.set_ylabel(None)
				#ax.set_yticklabels([])
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

def plot_cm(rootdir, cfilename, kf, lcset_name, model_names,
	figsize=(6,5),
	train_mode='fine-tuning',
	export_animation=False,
	animation_duration=12,
	new_order_classes=['SNIa', 'SNIbc', 'SNII-b-n', 'SLSN'],
	n=1e3,
	):
	for kmn,model_name in enumerate(model_names):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
		files, files_ids = fcfiles.gather_files_by_kfold(load_roodir, kf, lcset_name, fext='d')
		print(f'ids={files_ids}(n={len(files_ids)}#) - model={model_name}')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		is_parallel = 'Parallel' in model_name
		days = files[0]()['days']

		plot_animation = PlotAnimation(animation_duration,
			is_dummy=not export_animation,
			#save_init_frame=True,
			save_end_frame=True,
			)

		target_days = days if export_animation else [days[-1]]
		bar = ProgressBar(len(target_days), bar_format='{l_bar}{bar}{postfix}')
		for kd,day in enumerate(target_days):
			bar(f'{day:.1f}/{target_days[-1]:.1f} [days]')
			xe_dict = {}
			for metric_name in ['b-precision', 'b-recall', 'b-f1score']:
				metric_curve = np.concatenate([f()['days_class_metrics_df'][metric_name].values[None] for f in files], axis=0)
				interp_metric_curve = interp1d(days, metric_curve)(np.linspace(days.min(), day, int(n)))
				xe_metric_curve = XError(interp_metric_curve[:,-1])
				xe_dict[metric_name] = xe_metric_curve

			bprecision_xe = xe_dict['b-precision']
			brecall_xe = xe_dict['b-recall']
			bf1score_xe = xe_dict['b-f1score']

			title = ''
			title += f'{utils.get_fmodel_name(model_name)}'+'\n'
			#title += f'survey={survey}-{"".join(band_names)} [{kf}@{lcset_name}]'+'\n'
			#title += f'train-mode={train_mode} - eval-set={kf}@{lcset_name}'+'\n'
			title += f'b-recall={brecall_xe} - b-f1score={bf1score_xe}'+'\n'
			#title += f'b-p/r={bprecision_xe} / {brecall_xe}'+'\n'
			#title += f'b-f1score={bf1score_xe}'+'\n'
			if export_animation:
				#title += str(bar)+'\n'
				title += f'day-threshold={day:.3f}/{days[-1]:.3f} [days]'+'\n'
			cm_kwargs = {
				#'fig':fig,
				#'ax':ax,
				'title':title[:-1],
				'figsize':figsize,
				'new_order_classes':new_order_classes,
				}
			cms = np.concatenate([f()['days_cm'][day][None] for f in files], axis=0)
			fig, ax, cm_norm = plot_custom_confusion_matrix(cms, class_names, **cm_kwargs)
			uses_close_fig = kd<len(days)-1
			plot_animation.append(fig, uses_close_fig)

		bar.done()
		plt.show()
		plot_animation.save(f'../temp/{model_name}.mp4')

###################################################################################################################################################

def plot_temporal_encoding(rootdir, cfilename, kf, lcset_name, model_names,
	train_mode='pre-training',
	layers=3,
	figsize=(12,10),
	n=1e3,
	):
	for kmn,model_name in enumerate(model_names):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/temporal_encoding/{cfilename}/{kf}@{lcset_name}'
		files, files_ids = fcfiles.gather_files_by_id(load_roodir, fext='d')
		print(f'ids={files_ids}({len(files_ids)}#) - model={model_name}')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		mn_dict = strings.get_dict_from_string(model_name)
		mdl = mn_dict['mdl']
		is_parallel = 'Parallel' in mdl
		days = files[0]()['days']
		#days = np.linspace(days[0], days[-1], int(n))
		#days = np.linspace(0, 40, int(n))
		

		def get_fourier(t, weights, te_periods, te_phases):
			x = np.zeros_like(t)
			for kw,w in enumerate(weights):
				x += w*np.sin(2*np.pi*t/te_periods[kw]+te_phases[kw])
			return x

		###
		b = 'r'
		fig, axs = plt.subplots(layers, 1, figsize=figsize)
		cmap = cm.get_cmap('viridis', 100)

		for layer in range(0, layers):
			ax = axs[layer]
			#for kfile,file in enumerate(files):
			kfile = 1
			d = files[kfile]()['temporal_encoding_info']['encoder'][f'ml_attn.{b}' if is_parallel else f'ml_attn']['te_film'][layer]
			weight = d['weight']
			alpha_weights, beta_weights = np.split(weight.T, 2, axis=-1)
			#kfu = 1
			dalphas = []
			alphas = []
			dbetas = []
			betas = []
			for kfu in range(0, len(weight)//2):
				te_ws = d['te_ws']
				te_periods = d['te_periods']
				#print(te_periods)
				te_phases = d['te_phases']
				#print(alpha_weight.shape, beta_weight.shape, te_ws.shape, te_phases.shape)

				alpha = get_fourier(days, alpha_weights[:,kfu], te_periods, te_phases)
				alphas += [(alpha[None]*1e0)**2] # fixme luego usar (alpha+1)
				dalphas += [(np.diff(alpha, prepend=alpha[0])[None]*1e0)**2]

				beta = get_fourier(days, beta_weights[:,kfu], te_periods, te_phases)
				betas += [(beta[None]*1e0)**2]
				dbetas += [(np.diff(beta, prepend=beta[0])[None]*1e0)**2]
				#ax.plot(days, alpha, 'r', lw=1, label=f'scale learned curves' if kfu==0 else None)#, c=cmap(k/len(te_ws)))
				#ax.plot(days, beta, 'g', lw=1, label='bias learned curves' if kfu==0 else None)#, c=cmap(k/len(te_ws)))
				ax.grid(alpha=0.5)
				ax.legend()
				label = f'encoder-layer={layer} - band={b}' if is_parallel else f'encoder-layer={layer}'
				ax.set_ylabel(label)
				#ax.set_yticklabels([])
				#ax.set_xlim([0,1000])

			ax.plot(days, np.concatenate(alphas, axis=0).mean(axis=0), '-r', lw=1, label=f'scale learned curves' if kfu==0 else None)#, c=cmap(k/len(te_ws)))
			ax.plot(days, np.concatenate(betas, axis=0).mean(axis=0), '-g', lw=1, label='bias learned curves' if kfu==0 else None)#, c=cmap(k/len(te_ws)))
			ax.plot(days, np.concatenate(dalphas, axis=0).mean(axis=0), '--r', lw=1, label=f'scale learned curves' if kfu==0 else None)#, c=cmap(k/len(te_ws)))
			ax.plot(days, np.concatenate(dbetas, axis=0).mean(axis=0), '--g', lw=1, label='bias learned curves' if kfu==0 else None)#, c=cmap(k/len(te_ws)))
			ax.set_xlabel('time [days]')

		_label = strings.get_string_from_dict({k:mn_dict[k] for k in mn_dict.keys() if k in label_keys}, key_key_separator=' - ')
		label = f'{mdl} ({_label})'
		title = ''
		title += f'{label}'+'\n'
		title += f'survey={survey} [{kf}@{lcset_name}] - bands={"".join(band_names)}'+'\n'
		title += f'train-mode={train_mode}'+'\n'
		axs[0].set_title(title[:-1])
		plt.show()

###################################################################################################################################################

def xxx(rootdir, cfilename, kf, lcset_name, model_names,
	train_mode='pre-training',
	layers=2,
	figsize=(12,10),
	n=1e3,
	):
	for kmn,model_name in enumerate(model_names):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/temporal_encoding/{cfilename}/{kf}@{lcset_name}'
		files, files_ids = fcfiles.gather_files_by_id(load_roodir, fext='d')
		print(f'ids={files_ids}({len(files_ids)}#) - model={model_name}')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		mn_dict = strings.get_dict_from_string(model_name)
		mdl = mn_dict['mdl']
		is_parallel = 'Parallel' in mdl
		days = files[0]()['days']
		#days = np.linspace(days[0], days[-1], int(n))
		days = np.linspace(0, 40, int(n))
		
		###
		b = 'r'
		fig, axs = plt.subplots(1+layers, 1, figsize=figsize)
		cmap = cm.get_cmap('viridis', 100)

		ax = axs[0]
		d = files[0]()['temporal_encoding_info']['encoder'][f'ml_attn.{b}' if is_parallel else f'ml_attn']['te_film'][0]
		print(d.keys())
		te_ws = d['initial_ws']
		te_phases = np.zeros_like(te_ws)

		for k in range(0, len(te_ws)):
			w = te_ws[k]
			p = 2*math.pi/w
			phase = te_phases[k]
			ntime = days*d['ktime']
			ax.plot(p, 0, 'o', c=cmap(k/len(te_ws)))
			ax.grid(alpha=0.5, axis='x')
			ax.set_ylabel(f'initial condition')
			ax.set_yticklabels([])

		for layer in range(0, layers):
			ax = axs[1+layer]
			for kfile,file in enumerate(files):
				d = file()['temporal_encoding_info']['encoder'][f'ml_attn.{b}' if is_parallel else f'ml_attn']['te_film'][layer]
				te_ws = d['te_ws']
				te_phases = d['te_phases']
				ktime = d['ktime']

				for k in range(0, len(te_ws)):
					w = te_ws[k]
					p = 2*math.pi/w
					phase = te_phases[k]
					ntime = days*d['ktime']
					ax.plot(p, kfile+np.random.uniform(0, 0.1), 'o', c=cmap(k/len(te_ws)))
					ax.grid(alpha=0.5, axis='x')
					label = f'layer={layer} - b={b}' if is_parallel else f'layer={layer}'
					ax.set_ylabel(label)
					ax.set_yticklabels([])
					#ax.set_xlim([0,1000])

		ax.set_xlabel('temporal-encoding periods [days]')

		_label = strings.get_string_from_dict({k:mn_dict[k] for k in mn_dict.keys() if k in label_keys}, key_key_separator=' - ')
		label = f'{mdl} ({_label})'
		title = ''
		title += f'{label}'+'\n'
		title += f'survey={survey}-{"".join(band_names)} [{kf}@{lcset_name}]'+'\n'
		title += f'train-mode={train_mode}'+'\n'
		axs[0].set_title(title[:-1])
		plt.show()