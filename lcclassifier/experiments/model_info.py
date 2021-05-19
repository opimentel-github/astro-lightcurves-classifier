from __future__ import print_function
from __future__ import division
from . import C_

import torch
from fuzzytorch.utils import get_model_name, TDictHolder, minibatch_dict_collate
from fuzzytorch.models.utils import count_parameters
import numpy as np
from fuzzytools.progress_bars import ProgressBar, ProgressBarMulti
import fuzzytools.files as files
import fuzzytools.datascience.metrics as fcm
from fuzzytools.cuteplots.utils import save_fig
from fuzzytools.dataframes import DFBuilder
from fuzzytools.dicts import update_dicts
import matplotlib.pyplot as plt
import fuzzytorch.models.seq_utils as seq_utils
import pandas as pd

###################################################################################################################################################

def save_model_info(train_handler, data_loader, save_rootdir,
	**kwargs):
	train_handler.load_model() # important, refresh to best model
	train_handler.model.eval() # model eval
	dataset = data_loader.dataset # get dataset
	
	results = {
		'model_name':train_handler.model.get_name(),
		'survey':dataset.survey,
		'band_names':dataset.band_names,
		'class_names':dataset.class_names,

		'parameters':count_parameters(train_handler.model),
		'monitors':{},
	}
	for lmonitor in train_handler.lmonitors:
		results['monitors'][lmonitor.name] = {
			'save_dict':lmonitor.get_save_dict(),
			'best_epoch':lmonitor.get_best_epoch(),
			'time_per_iteration':lmonitor.get_time_per_iteration(),
			#'time_per_epoch_set':{set_name:lmonitor.get_time_per_epoch_set(set_name) for set_name in ['train', 'val']},
			'time_per_epoch':lmonitor.get_time_per_epoch(),
			'total_time':lmonitor.get_total_time(),
		}

	### save file
	save_filedir = f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}.d'
	files.save_pickle(save_filedir, results) # save file
	return