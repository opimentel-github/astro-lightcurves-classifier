from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import fuzzytools.files as ftfiles
import fuzzytools.strings as strings
from matplotlib import cm
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import fuzzytools.matplotlib.ax_styles as ax_styles
from . import utils as utils

RANDOM_STATE = C_.RANDOM_STATE
PERCENTILE_PLOT = 95
SHADOW_ALPHA = .25
FIGSIZE = (20, 10)
ALPHABET = C_.ALPHABET

###################################################################################################################################################

def get_fourier(t, weights, te_periods, te_phases):
	x = np.zeros_like(t)
	for kw,w in enumerate(weights):
		x += w*np.sin(2*np.pi*t/te_periods[kw]+te_phases[kw])
	return x

def get_diff(x, k):
	assert len(x.shape)==1
	new_x = copy(x)
	for _ in range(0, k):
		new_x = np.diff(new_x)
		new_x = np.concatenate([[new_x[0]], new_x], axis=0)
	return new_x

###################################################################################################################################################

def plot_temporal_encoding(rootdir, cfilename, kf, lcset_name, model_names,
	train_mode='pre-training',
	layers=1,
	figsize=FIGSIZE,
	n=1e3,
	percentile=PERCENTILE_PLOT,
	shadow_alpha=SHADOW_ALPHA,
	):
	for kmn,model_name in enumerate(model_names):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/temporal_encoding/{cfilename}'
		if not ftfiles.path_exists(load_roodir):
			continue
		files, files_ids = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
			fext='d',
			disbalanced_kf_mode='ignore', # error oversampling ignore
			random_state=RANDOM_STATE,
			)
		print(f'{model_name} {files_ids}({len(files_ids)}#)')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		mn_dict = strings.get_dict_from_string(model_name)
		mdl = mn_dict['mdl']
		is_parallel = 'Parallel' in mdl
		if not is_parallel:
			continue

		days = files[0]()['days']
		days = np.linspace(days[0], days[-1], int(n))

		global_median_curves_d = {}
		fig, axs = plt.subplots(2, len(band_names), figsize=figsize)
		for kfile,file in enumerate(files):
			for kb,b in enumerate(band_names):
				d = file()['temporal_encoding_info']['encoder'][f'ml_attn.{b}']['te_film']
				weight = d['weight'] # (f,2m)
				alpha_weights, beta_weights = np.split(weight.T, 2, axis=-1) # (f,2m)>(2m,f/2),(2m,f/2)
				scales = []
				biases = []
				for kfu in range(0, alpha_weights.shape[-1]):
					te_ws = d['te_ws']
					te_periods = d['te_periods']
					te_phases = d['te_phases']
					alpha = get_fourier(days, alpha_weights[:,kfu], te_periods, te_phases)
					dalpha = get_diff(alpha, 1)**2
					beta = get_fourier(days, beta_weights[:,kfu], te_periods, te_phases)
					dbeta = get_diff(beta, 1)**2
					scales += [dalpha]
					biases += [dbeta]

				d = {
					'scale':{'curve':scales, 'c':'r'},
					'bias':{'curve':biases, 'c':'g'},
					}
				for kax,curve_name in enumerate(['scale', 'bias']):
					ax = axs[kax,kb]
					curves = d[curve_name]['curve']
					c = 'k'
					median_curve = np.median(np.concatenate([curve[None]*1e6 for curve in curves], axis=0), axis=0)
					if not f'{kax}/{kb}' in global_median_curves_d.keys():
						global_median_curves_d[f'{kax}/{kb}'] = []
					global_median_curves_d[f'{kax}/{kb}'] += [median_curve]
					ax.plot(days, median_curve,
						c=c,
						alpha=1,
						lw=.5,
						)
					ax.plot([None], [None], c=c, label=f'variation power continuous-time function' if kfile==0 else None)
					ax.legend(loc='upper right')
					ax.grid(alpha=0.5)
					ax.set_xlim((days[0], days[-1]))
					ax.set_title('$\\bf{('+f'{ALPHABET[kb]}.{kax}'+')}$ '+f'variation power for {curve_name}; band={b}')
					ax_styles.set_color_borders(ax, C_.COLOR_DICT[b])
					if kb==0:
						ax.set_ylabel(f'variation power')
					else:
						pass
					if kax==0:
						ax.set_xticklabels([])
					else:
						ax.set_xlabel(f'time [days]')
			model_label = utils.get_fmodel_name(model_name)
			suptitle = ''
			suptitle = f'{model_label}'+'\n'
			# suptitle += f'set={survey} [{lcset_name.replace(".@", "")}]'+'\n'
			fig.suptitle(suptitle[:-1], va='bottom')

		for k in global_median_curves_d.keys():
			kax,kb = k.split('/')
			median_curves = global_median_curves_d[k]
			ax = axs[int(kax),int(kb)]
			ax.plot(days, np.median(np.concatenate([median_curve[None] for median_curve in median_curves], axis=0), axis=0), '-',
				# c=['r', 'g'][int(kax)],
				c='r',
				label=f'median variation power continuous-time function',
				)
			ax.legend(loc='upper right')

		fig.tight_layout()
		plt.show()