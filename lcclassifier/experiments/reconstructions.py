from __future__ import print_function
from __future__ import division
from . import C_

import torch
from fuzzytorch.utils import TDictHolder, tensor_to_numpy
import numpy as np
from lchandler import C_ as C_lchandler
from lchandler.plots.lc import plot_lightcurve
import fuzzytools.prints as prints
from fuzzytools.cuteplots.utils import save_fig
import matplotlib.pyplot as plt
import random
from fuzzytools.strings import bf_alphabet_count

FIGSIZE = (10,10)

###################################################################################################################################################

def save_reconstructions(train_handler, data_loader, save_rootdir,
	m:int=2,
	figsize:tuple=FIGSIZE,
	nc:int=1,
	**kwargs):
	results = []
	for experiment_id in range(0, m):
		random.seed(experiment_id)
		np.random.seed(experiment_id)
		r = _save_reconstructions(train_handler, data_loader, save_rootdir, str(experiment_id),
			figsize,
			nc,
			**kwargs)
		results.append(r)
	return results

def _save_reconstructions(train_handler, data_loader, save_rootdir, experiment_id,
	figsize:tuple=FIGSIZE,
	nc:int=1,
	**kwargs):
	train_handler.load_model() # important, refresh to best model
	train_handler.model.eval() # important, model eval mode
	dataset = data_loader.dataset # get dataset
	
	with torch.no_grad():
		lcobj_names = dataset.get_random_stratified_lcobj_names(nc)
		fig, axs = plt.subplots(len(lcobj_names), 1, figsize=figsize)
		for k,lcobj_name in enumerate(lcobj_names):
			ax = axs[k]
			in_tdict, lcobj = dataset.get_item(lcobj_name)
			in_tdict = dataset.fix_tdict(in_tdict)
			tdict = train_handler.model(TDictHolder(in_tdict).to(train_handler.device, add_dummy_dim=True))
			for kb,b in enumerate(dataset.band_names):
				p_onehot = tdict[f'input/onehot.{b}'][...,0] # (b,t)
				p_rtime = tdict[f'input/rtime.{b}'][...,0] # (b,t)
				#p_dtime = tdict[f'input/dtime.{b}'][...,0] # (b,t)
				#p_x = tdict[f'input/x.{b}'] # (b,t,f)
				#p_rerror = tdict[f'target/rerror.{b}'] # (b,t,1)
				#p_rx = tdict[f'target/rec_x.{b}'] # (b,t,1)

				b_len = p_onehot.sum().item()
				lcobjb = lcobj.get_b(b)
				plot_lightcurve(ax, lcobj, b, label=f'{b} obs', max_day=dataset.max_day)

				### rec plot
				p_rtime = tensor_to_numpy(p_rtime[0,:]) # (b,t) > (t)
				p_rx_pred = tdict[f'model/decx.{b}'][0,:,0] # (b,t,1) > (t)
				p_rx_pred = dataset.get_rec_inverse_transform(tensor_to_numpy(p_rx_pred), b)
				ax.plot(p_rtime[:b_len], p_rx_pred[:b_len], '--', c=C_lchandler.COLOR_DICT[b], label=f'{b} obs reconstruction')

			title = ''
			if k==0:
				title += f'multi-band light curve reconstructions'+'\n'
				title += f'set={dataset.survey} [{dataset.lcset_name.replace(".@", "")}]'+'\n'
			title += f'{bf_alphabet_count(k)} lcobj={lcobj_names[k]} [{dataset.class_names[lcobj.y]}]'+'\n'
			ax.set_title(title[:-1])
			ax.set_ylabel('observation [flux]')
			ax.legend(loc='upper right')
			ax.grid(alpha=0.5)

		ax.set_xlabel('time [days]')
		fig.tight_layout()

	### save file
	image_save_filedir = f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}~exp_id={experiment_id}.png'
	save_fig(image_save_filedir, fig)
	dataset.reset_max_day() # very important!!
	return
