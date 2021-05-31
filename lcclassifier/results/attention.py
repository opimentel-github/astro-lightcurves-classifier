from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import fuzzytools.files as fcfiles
import fuzzytools.strings as strings
import matplotlib.pyplot as plt
from . import utils as utils
from fuzzytools.lists import flat_list

###################################################################################################################################################

def plot_attention_statistics(rootdir, cfilename, kf, lcset_name, model_name,
	train_mode='pre-training',
	figsize=(14,8),
	#attn_key='attn_scores_min_max',
	#attn_entropy_key='attn_entropy',
	attn_th=0.95,
	len_th=10,

	n_bins=50,
	bins_xrange=[None, None],
	bins_yrange=[None, None],
	):
	load_roodir = f'{rootdir}/{model_name}/{train_mode}/attn_stats/{cfilename}/{kf}@{lcset_name}'
	print(load_roodir)
	files, files_ids = fcfiles.gather_files_by_id(load_roodir, fext='d')
	print(f'{model_name} {files_ids}({len(files_ids)}#)')
	assert len(files)>0

	survey = files[0]()['survey']
	band_names = files[0]()['band_names']
	class_names = files[0]()['class_names']
	#days = files[0]()['days']

	fig, axs = plt.subplots(1, 2, figsize=figsize)

	label = f'{utils.get_fmodel_name(model_name)}'

	title = ''
	title += f'linear-trend v/s days-from-peak'+'\n'
	title += f'train-mode={train_mode} - survey={survey}-{"".join(band_names)} [{kf}@{lcset_name}]'+'\n'
	title += f'{label}'+'\n'
	fig.suptitle(title[:-1], va='bottom')

	target_class_names = class_names
	#target_class_names = ['SLSN']
	x_key = 'days_from_peak1.j'
	y_key = 'linear_trend_m.j'

	xy = []
	xy_marginal = []
	for b in band_names:
		attn_scores_collection = flat_list([f()['attn_scores_collection'][b] for f in files])
		for d in attn_scores_collection:
			#print(d.keys())
			xy_marginal += [[d[x_key], d[y_key]]]
			if d['attn_scores_min_max_k.j']>=attn_th and d['b_len']>=len_th and d['c'] in target_class_names:
				xy += [[d[x_key], d[y_key]]]
	
	xy_marginal = np.array(xy_marginal)
	xy = np.array(xy)
	print('xy', xy.shape, 'xy_marginal', xy_marginal.shape)
	#assert 0

	xrange0 = xy[:,0].min() if bins_xrange[0] is None else bins_xrange[0]
	xrange1 = xy[:,0].max() if bins_xrange[1] is None else bins_xrange[1]
	yrange0 = xy[:,1].min() if bins_yrange[0] is None else bins_yrange[0]
	yrange1 = xy[:,1].max() if bins_yrange[1] is None else bins_yrange[1]

	### marginals
	ax = axs[0]
	H, xedges, yedges = np.histogram2d(xy_marginal[:,0], xy_marginal[:,1], bins=(np.linspace(xrange0, xrange1, n_bins), np.linspace(yrange0, yrange1, n_bins)))
	H = H.T  # Let each row list bins with common y range.
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	ax.imshow(H, interpolation='nearest', origin='lower', aspect='auto', extent=extent)
	ax.axvline(0, linewidth=.5, color='w')
	ax.axhline(0, linewidth=.5, color='w')
	#title = f'{metric_name} v/s days - mode: {mode}'
	#title += f'\nsurvey: {survey} - bands: {band_names}'
	#title += f'\nshadow region: {xe.get_symbol("std")} ({len(xe)} itrs)'
	title = ''
	title += 'joint distribution'+'\n'
	ax.set_title(title[:-1])

	label_dict = {
		'linear_trend_m.j':'linear-trend',
		'days_from_peak1.j':'days-from-peak',
	}

	xlabel = label_dict[x_key]
	ylabel = label_dict[y_key]
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	txt_y = yedges[0]
	ax.text(0, txt_y, ' pre-peak ', fontsize=12, c='w', ha='right', va='bottom')
	ax.text(0, txt_y, ' post-peak ', fontsize=12, c='w', ha='left', va='bottom')

	### attn stats
	ax = axs[1]
	H, xedges, yedges = np.histogram2d(xy[:,0], xy[:,1], bins=(np.linspace(xrange0, xrange1, n_bins), np.linspace(yrange0, yrange1, n_bins)))
	H = H.T  # Let each row list bins with common y range.
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	ax.imshow(H, interpolation='nearest', origin='lower', aspect='auto', extent=extent)
	ax.axvline(0, linewidth=.5, color='w')
	ax.axhline(0, linewidth=.5, color='w')
	#title = f'{metric_name} v/s days - mode: {mode}'
	#title += f'\nsurvey: {survey} - bands: {band_names}'
	#title += f'\nshadow region: {xe.get_symbol("std")} ({len(xe)} itrs)'
	title = ''
	title += f'conditional joint distribution w/ '+'$\\bar{s}_{ij}\\geq'+str(attn_th)+'$'+'\n'
	ax.set_title(title[:-1])
	ax.set_xlabel(xlabel)
	ax.set_yticks([])
	ax.text(0, txt_y, ' pre-peak ', fontsize=12, c='w', ha='right', va='bottom')
	ax.text(0, txt_y, ' post-peak ', fontsize=12, c='w', ha='left', va='bottom')

	fig.tight_layout()
	plt.show()
	return