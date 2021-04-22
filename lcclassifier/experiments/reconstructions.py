from __future__ import print_function
from __future__ import division
from . import C_

import torch
from fuzzytorch.utils import get_model_name, TDictHolder, tensor_to_numpy
import numpy as np
from lchandler import C_ as C_lchandler
from lchandler.plots.lc import plot_lightcurve
import flamingchoripan.prints as prints
from flamingchoripan.cuteplots.utils import save_fig
import matplotlib.pyplot as plt

###################################################################################################################################################

def save_reconstructions(train_handler, data_loader, save_rootdir,
	m:int=2,
	figsize:tuple=C_.DEFAULT_FIGSIZE_BOX,
	nc:int=1,
	**kwargs):
	results = []
	for experiment_id in range(0, m):
		r = _save_reconstructions(train_handler, data_loader, save_rootdir, str(experiment_id),
			figsize,
			nc,
			**kwargs)
		results.append(r)
	return results

def _save_reconstructions(train_handler, data_loader, save_rootdir, experiment_id,
	figsize:tuple=C_.DEFAULT_FIGSIZE_BOX,
	nc:int=1,
	**kwargs):
	### dataloader and extract dataset - important
	train_handler.load_model() # important, refresh to best model
	train_handler.model.eval() # important, model eval mode
	data_loader.eval() # set mode
	dataset = data_loader.dataset # get dataset
	
	with torch.no_grad():
		lcobj_names = dataset.get_random_stratified_lcobj_names(nc)
		fig, axs = plt.subplots(len(lcobj_names), 1, figsize=figsize)
		for k,lcobj_name in enumerate(lcobj_names):
			ax = axs[k]
			tdict, lcobj = dataset.get_item(lcobj_name, return_lcobjs=True)
			out_tdict = train_handler.model(TDictHolder(tdict).to(train_handler.device, add_dummy_dim=True))
			onehot = out_tdict['input']['onehot']
			
			for kb,b in enumerate(dataset.band_names):
				b_len = onehot[...,kb].sum().item()
				lcobjb = lcobj.get_b(b)
				plot_lightcurve(ax, lcobj, b, label=f'{b} obs', max_day=dataset.max_day)

				### rec plot
				days = tensor_to_numpy(out_tdict['input']['time'][0,onehot[0,:,kb]])
				p_rx_pred = tensor_to_numpy(out_tdict['model'][f'rec_x.{b}'][0,:,0])
				p_rx_pred = dataset.get_rec_inverse_transform(p_rx_pred, b)
				ax.plot(days[:b_len], p_rx_pred[:b_len], '--', c=C_lchandler.COLOR_DICT[b], label=f'{b} obs reconstruction')

			title = ''
			title += f'survey={dataset.survey} [{dataset.lcset_name}] - lcobj={lcobj_names[k]} [{dataset.class_names[lcobj.y]}]'+'\n'
			ax.set_title(title[:-1])
			ax.set_ylabel('observations [flux]')
			ax.legend(loc='upper right')
			ax.grid(alpha=0.5)

		ax.set_xlabel('time [days]')
		fig.tight_layout()

	### save file
	image_save_filedir = f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}~exp_id={experiment_id}.png'
	save_fig(image_save_filedir, fig)
	return
