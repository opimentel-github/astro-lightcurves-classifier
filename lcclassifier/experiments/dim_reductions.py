from __future__ import print_function
from __future__ import division
from . import C_

import torch
from fuzzytorch.utils import TDictHolder, tensor_to_numpy, minibatch_dict_collate
import numpy as np
from fuzzytools.progress_bars import ProgressBar, ProgressBarMulti
import fuzzytools.files as files
from fuzzytools.dataframes import DFBuilder
from fuzzytools.dicts import update_dicts
import fuzzytorch.models.seq_utils as seq_utils
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import TSNE
from umap import UMAP
from fuzzytools.datascience.dim_reductors import DimReductor

DEFAULT_DAYS_N = C_.DEFAULT_DAYS_N

###################################################################################################################################################

def save_dim_reductions(train_handler, data_loader, save_rootdir,
	target_is_onehot:bool=False,
	target_y_key='target/y',
	pred_y_key='model/y',
	days_n:int=DEFAULT_DAYS_N,
	random_state=0,
	**kwargs):
	train_handler.load_model() # important, refresh to best model
	train_handler.model.eval() # important, model eval mode
	dataset = data_loader.dataset # get dataset

	dataset.reset_max_day() # always reset max day
	days_embeddings = {}
	days_y_true = {}
	days = np.linspace(C_.DEFAULT_MIN_DAY, dataset.max_day, days_n)#[::-1]
	bar = ProgressBar(len(days))
	with torch.no_grad():
		for day in days:
			dataset.set_max_day(day) # very important!!
			dataset.calcule_precomputed() # very important!!
			
			tdicts = []
			for ki,in_tdict in enumerate(data_loader):
				_tdict = train_handler.model(TDictHolder(in_tdict).to(train_handler.device))
				tdicts += [_tdict]
			tdict = minibatch_dict_collate(tdicts)

			### class prediction
			y_true = tdict[target_y_key] # (b)
			#y_pred_p = torch.nn.functional.softmax(tdict[pred_y_key], dim=-1) # (b,c)
			y_pred_p = torch.sigmoid(tdict[pred_y_key]) # (b,c)
			#print('y_pred_p',y_pred_p[0])

			if target_is_onehot:
				assert y_pred_.shape==y_true.shape
				y_true = torch.argmax(y_true, dim=-1)

			y_true = tensor_to_numpy(y_true)
			y_pred_p = tensor_to_numpy(y_pred_p)

			days_y_true[day] = y_true

			### embeddings
			encz_last = tdict[f'model/encz_last']
			days_embeddings[day] = tensor_to_numpy(encz_last)
			bar(f'day={day:.3f}/{days[-1]:.3f}')
	bar.done()

	### train map
	scaler = StandardScaler()
	reduction_map = UMAP(
		n_components=2,
		metric='euclidean',
		n_neighbors=int(10),
		min_dist=.1,
		random_state=random_state,
		transform_seed=random_state,
		)
	dim_reductor = DimReductor(scaler, reduction_map,
		inter_pca_dims=10,
		)
	dim_reductor.fit([days_embeddings[day] for day in days])

	### compute maps
	days_dim_reductions = {}
	bar = ProgressBar(len(days))
	for day in days:
		x = days_embeddings[day]
		new_x = dim_reductor.transform(x)
		days_dim_reductions[day] = new_x
		bar(f'day={day:.3f}/{days[-1]:.3f} - x.shape={x.shape} - new_x.shape={new_x.shape}')
	bar.done()

	results = {
		'model_name':train_handler.model.get_name(),
		'survey':dataset.survey,
		'band_names':dataset.band_names,
		'class_names':dataset.class_names,

		'days':days,
		'days_dim_reductions':days_dim_reductions,
		'days_y_true':days_y_true,
	}

	### save file
	save_filedir = f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}.d'
	files.save_pickle(save_filedir, results) # save file
	dataset.reset_max_day() # very important!!
	dataset.calcule_precomputed() # very important!!
	return