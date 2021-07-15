from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import fuzzytools.files as ftfiles
import fuzzytools.strings as strings
import matplotlib.pyplot as plt
from . import utils as utils
from fuzzytools.lists import flat_list
import fuzzytools.matplotlib.ax_styles as ax_styles
from fuzzytools.strings import bf_alphabet_count

RANDOM_STATE = 0
FIGSIZE = (16,10)

###################################################################################################################################################

def plot_slope_distance_attnstats(rootdir, cfilename, kf, lcset_name, model_names,
	train_mode='pre-training',
	figsize=FIGSIZE,
	attn_th=0.5,
	len_th=5,
	n_bins=50,
	bins_xrange=[None, None],
	bins_yrange=[None, None],
	cmap_name='inferno', # plasma viridis inferno
	dj=3,
	distance_mode='mean',
	):
	for kmn,model_name in enumerate(model_names):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/attn_stats/{cfilename}'
		if not ftfiles.path_exists(load_roodir):
			continue
		files, files_ids = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
			fext='d',
			disbalanced_kf_mode='ignore', # error oversampling ignore
			random_state=RANDOM_STATE,
			)
		print(f'{model_name} {files_ids}({len(files_ids)}#)')
		assert len(files)>0

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		#days = files[0]()['days']

		target_class_names = class_names
		x_key = f'peak_distance.j~dj={dj}~mode={distance_mode}'
		y_key = f'local_slope_m.j~dj={dj}'
		label_dict = {
			x_key:f'peak-distance [days]',
			y_key:f'local-slope using $\\Delta j={dj}$',
			}

		fig, axs = plt.subplots(2, len(band_names), figsize=figsize)
		for kb,b in enumerate(band_names):
			xy_marginal = []
			xy_attn = []
			attn_scores_collection = flat_list([f()['attn_scores_collection'][b] for f in files])
			for d in attn_scores_collection:
				#print(d.keys())
				xy_marginal += [[d[x_key], d[y_key]]]
				if d['attn_scores_min_max_k.j']>=attn_th and d['b_len']>=len_th and d['c'] in target_class_names:
					xy_attn += [[d[x_key], d[y_key]]]
		
			xy_marginal = np.array(xy_marginal)
			xy_attn = np.array(xy_attn)
			print('xy_marginal', xy_marginal.shape, 'xy_attn', xy_attn.shape)

			xrange0 = xy_attn[:,0].min() if bins_xrange[0] is None else bins_xrange[0]
			xrange1 = xy_attn[:,0].max() if bins_xrange[1] is None else bins_xrange[1]
			yrange0 = xy_attn[:,1].min() if bins_yrange[0] is None else bins_yrange[0]
			yrange1 = xy_attn[:,1].max() if bins_yrange[1] is None else bins_yrange[1]

			d = {
				'xy_marginal':{'xy':xy_marginal, 'title':'joint distribution'},
				'xy_attn':{'xy':xy_attn, 'title':f'conditional joint distribution using '+'$\\bar{s}_{th}='+str(attn_th)+'$'},
				}
			for kax,xy_name in enumerate(['xy_marginal', 'xy_attn']):
				ax = axs[kax,kb]
				xy = d[xy_name]['xy']
				H, xedges, yedges = np.histogram2d(xy[:,0], xy[:,1], bins=(np.linspace(xrange0, xrange1, n_bins), np.linspace(yrange0, yrange1, n_bins)))
				H = H.T  # Let each row list bins with common y range.
				extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
				ax.imshow(H, interpolation='nearest', origin='lower', aspect='auto',
					cmap=cmap_name, 
					extent=extent,
					)
				ax.axvline(0, linewidth=.5, color='w')
				ax.axhline(0, linewidth=.5, color='w')
				title = ''
				title += f'{bf_alphabet_count(kb, kax)} {d[xy_name]["title"]}; band={b}'+'\n'
				ax.set_title(title[:-1])

				txt_y = yedges[0]
				ax.text(0, txt_y, 'pre SNe-peak < ', fontsize=12, c='w', ha='right', va='bottom')
				ax.text(0, txt_y, ' > post SNe-peak', fontsize=12, c='w', ha='left', va='bottom')
				ax_styles.set_color_borders(ax, C_.COLOR_DICT[b])

				xlabel = label_dict[x_key]
				ylabel = label_dict[y_key]
				if kb==0:
					ax.set_ylabel(ylabel)
				else:
					ax.set_yticklabels([])
				if kax==0:
					ax.set_xticklabels([])
				else:
					ax.set_xlabel(xlabel)

		model_label = utils.get_fmodel_name(model_name)
		suptitle = ''
		suptitle += f'local-slope v/s peak-distance'+'\n'
		# suptitle += f'survey={survey}-{"".join(band_names)} [{kf}@{lcset_name}]'+'\n'
		suptitle += f'{model_label}'+'\n'
		fig.suptitle(suptitle[:-1], va='bottom')

		fig.tight_layout()
		plt.show()