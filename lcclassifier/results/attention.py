from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import warnings
from flamingchoripan.files import search_for_filedirs, load_pickle
import flamingchoripan.strings as strings
from flamingchoripan.cuteplots.cm_plots import plot_custom_confusion_matrix
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import flamingchoripan.datascience.statistics as dstats
from . import utils as utils

###################################################################################################################################################

def plot_attention_statistics(rootdir, model_name, x_var, y_var,
	figsize=(6*2,6),
	fext='attnscores',
	attn_key='attn_scores_min_max',
	attn_entropy_key='attn_entropy/len',
	attn_th=0.8,
	attn_entropy_th_p=0.1,

	N=50,
	p=1,
	bins_xrange=[None, None],
	bins_yrange=[None, None],
	extent=None,
	):
	mode = 'fine-tuning'
	new_rootdir = f'{rootdir}/{mode}/{model_name}'
	filedirs = search_for_filedirs(new_rootdir, fext=fext, verbose=0)
	print(f'[{0}][{len(filedirs)}#] {model_name}')
	mn_dict = strings.get_dict_from_string(model_name)
	rsc = mn_dict['rsc']
	mdl = mn_dict['mdl']
	is_parallel = 'Parallel' in mdl

	filedir = filedirs[0]
	rdict = load_pickle(filedir, verbose=0)
	model_name = rdict['model_name']
	survey = rdict['survey']
	band_names = ''.join(rdict['band_names'])
	class_names = rdict['class_names']
	attn_scores_collection = rdict['attn_scores_collection']
	print(attn_scores_collection[0])

	attn_entropies = [r[attn_entropy_key] for r in attn_scores_collection]
	attn_entropies = list(set(attn_entropies))
	#print('attn_entropies',np.min(attn_entropies), np.mean(attn_entropies), attn_entropies[:100])
	attn_entropy_th = np.sort(attn_entropies)[int(len(attn_entropies)*attn_entropy_th_p)]
		
	#x = np.array([np.random.uniform(.25,.75) for r in attn_scores_collection if r[attn_entropy_key]<=attn_entropy_th and r[attn_key]>=attn_th])
	x = np.array([r[x_var] for r in attn_scores_collection if r[attn_entropy_key]<=attn_entropy_th and r[attn_key]>=attn_th])
	y = np.array([r[y_var] for r in attn_scores_collection if r[attn_entropy_key]<=attn_entropy_th and r[attn_key]>=attn_th])
	x_m = np.array([r[x_var] for r in attn_scores_collection])
	y_m = np.array([r[y_var] for r in attn_scores_collection])

	xrange0 = x.min()+abs(x.max()-x.min())*p if bins_xrange[0] is None else bins_xrange[0]
	xrange1 = x.max()-abs(x.max()-x.min())*p if bins_xrange[1] is None else bins_xrange[1]
	yrange0 = y.min()+abs(y.max()-y.min())*p if bins_yrange[0] is None else bins_yrange[0]
	yrange1 = y.max()-abs(y.max()-y.min())*p if bins_yrange[1] is None else bins_yrange[1]

	fig, axs = plt.subplots(1, 2, figsize=figsize)

	### marginals
	ax = axs[0]
	H, xedges, yedges = np.histogram2d(x_m, y_m, bins=(np.linspace(xrange0, xrange1, N), np.linspace(yrange0, yrange1, N)))
	H = H.T  # Let each row list bins with common y range.
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]] if extent is None else extent
	ax.imshow(H, interpolation='nearest', origin='lower', aspect='auto', extent=extent)
	ax.axvline(0, linewidth=.5, color='w')
	ax.axhline(0, linewidth=.5, color='w')
	#title = f'{metric_name} v/s days - mode: {mode}'
	#title += f'\nsurvey: {survey} - bands: {band_names}'
	#title += f'\nshadow region: {xe.get_symbol("std")} ({len(xe)} itrs)'
	#ax.set_title(title)
	ax.set_xlabel(x_var)
	ax.set_ylabel(y_var)

	### attn
	ax = axs[1]
	H, xedges, yedges = np.histogram2d(x, y, bins=(np.linspace(xrange0, xrange1, N), np.linspace(yrange0, yrange1, N)))
	H = H.T  # Let each row list bins with common y range.
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]] if extent is None else extent
	ax.imshow(H, interpolation='nearest', origin='lower', aspect='auto', extent=extent)
	ax.axvline(0, linewidth=.5, color='w')
	ax.axhline(0, linewidth=.5, color='w')
	#title = f'{metric_name} v/s days - mode: {mode}'
	#title += f'\nsurvey: {survey} - bands: {band_names}'
	#title += f'\nshadow region: {xe.get_symbol("std")} ({len(xe)} itrs)'
	#ax.set_title(title)
	ax.set_xlabel(x_var)
	ax.set_yticks([])

	fig.tight_layout()
	plt.show()