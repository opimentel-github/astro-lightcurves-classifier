from __future__ import print_function
from __future__ import division
from . import C_

import torch
from fuzzytorch.utils import get_model_name, TDictHolder
import numpy as np
from lchandler import C_ as C_lchandler
from lchandler.plots.lc import plot_lightcurve
import flamingchoripan.prints as prints
from flamingchoripan.cuteplots.utils import save_fig
import matplotlib.pyplot as plt

###################################################################################################################################################

def reconstructions_m(train_handler, data_loader,
	m:int=2,
	figsize:tuple=C_.DEFAULT_FIGSIZE_BOX,
	nc:int=1,
	save_rootdir:str='results',
	experiment_id:int=0,
	**kwargs):
	results = []
	for experiment_id in range(m):
		r = reconstructions(train_handler, data_loader,
			figsize,
			nc,
			save_rootdir,
			experiment_id,
			**kwargs)
		results.append(r)

	return results

def reconstructions(train_handler, data_loader,
	figsize:tuple=C_.DEFAULT_FIGSIZE_BOX,
	nc:int=1,
	save_rootdir:str='results',
	experiment_id:int=0,
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
				b_len = onehot[...,kb].sum()
				lcobjb = lcobj.get_b(b)
				plot_lightcurve(ax, lcobj, b, label=f'{b} observation', max_day=dataset.max_day)

				### rec plot
				days = out_tdict['input']['time'][0,onehot[0,:,kb]].cpu().numpy()
				p_rx_pred = out_tdict['model'][f'rec-x.{b}'][0,:,0].cpu().numpy()
				p_rx_pred = dataset.get_rec_inverse_transform(p_rx_pred, b)
				ax.plot(days[:b_len], p_rx_pred[:b_len], '--', c=C_lchandler.COLOR_DICT[b], label=f'{b} reconstruction')

			title = f'survey: {dataset.survey} - set: {dataset.lcset_name} - lcobj: {lcobj_names[k]} - class: {dataset.class_names[lcobj.y]}'
			ax.set_title(title)
			ax.set_ylabel('flux')
			ax.legend(loc='upper right')
			ax.grid(alpha=0.5)

		ax.set_xlabel('days')
		fig.tight_layout()

	### save file
	complete_save_roodir = train_handler.complete_save_roodir.split('/')[-1] # train_handler.get_complete_save_roodir().split('/')[-1]
	image_save_dir = f'{save_rootdir}/{complete_save_roodir}'
	image_save_filedir = f'{image_save_dir}/exp_id={experiment_id}°id={train_handler.id}°set={dataset.lcset_name}.png'
	#prints.print_green(f'> saving: {image_save_filedir}')
	save_fig(image_save_filedir, fig)
	return image_save_filedir
