#!/usr/bin/env python3
import sys
sys.path.append('../') # or just install the module
sys.path.append('../../fuzzy-torch') # or just install the module
sys.path.append('../../flaming-choripan') # or just install the module
sys.path.append('../../astro-lightcurves-handler') # or just install the module
sys.path.append('../../sne-lightcurves-synthetic') # or just install the module

if __name__== '__main__':
	### parser arguments
	import argparse
	from flamingchoripan.prints import print_big_bar

	parser = argparse.ArgumentParser('usage description')
	parser.add_argument('-gpu',  type=int, default=-1, help='gpu_index')
	parser.add_argument('-mc',  type=str, default='parallel_rnn_models', help='model_collections method')
	parser.add_argument('-email',  type=bool, default=False, help='send_email')
	parser.add_argument('-batch_size',  type=int, default=512, help='batch_size')
	parser.add_argument('-load_model',  type=bool, default=False, help='load_model')
	parser.add_argument('-epochs_max',  type=int, default=int(1e4), help='epochs_max')
	parser.add_argument('-save_rootdir',  type=str, default='SAVE_TESIS1', help='save_rootdir')
	parser.add_argument('-iid',  type=int, default=1, help='initial id')
	parser.add_argument('-fid',  type=int, default=5, help='final id')
	main_args = parser.parse_args()
	print_big_bar()

	###################################################################################################################################################
	from flamingchoripan.files import search_for_filedirs
	from lchandler import C_ as C_

	surveys_rootdir = '../../surveys-save/'
	filedirs = search_for_filedirs(surveys_rootdir, fext=C_.EXT_SPLIT_LIGHTCURVE)

	###################################################################################################################################################
	import numpy as np
	from flamingchoripan.files import load_pickle, save_pickle
	from flamingchoripan.files import get_dict_from_filedir
	from lchandler import C_ as C_

	def load_lcdataset(filename):
		assert filename.split('.')[-1]==C_.EXT_SPLIT_LIGHTCURVE
		return load_pickle(filename)

	filedir = '../../surveys-save/alerceZTFv7.1/survey=alerceZTFv7.1°bands=gr°mode=onlySNe°method=mcmc.splcds'

	filedict = get_dict_from_filedir(filedir)
	root_folder = filedict['*rootdir*']
	cfilename = filedict['*cfilename*']
	survey = filedict['survey']
	lcdataset = load_lcdataset(filedir)
	print(lcdataset['raw'].keys())
	print(lcdataset['raw'].get_random_lcobj(False).keys())
	print(lcdataset)

	###################################################################################################################################################
	from lcclassifier.models.model_collections import ModelCollections

	model_collections = ModelCollections(lcdataset)
	#getattr(model_collections, 'parallel_rnn_models_dt')()
	#getattr(model_collections, 'parallel_rnn_models_te')()
	#getattr(model_collections, 'serial_rnn_models_dt')()
	#getattr(model_collections, 'serial_rnn_models_te')()
	#getattr(model_collections, 'parallel_tcn_models_dt')()
	#getattr(model_collections, 'parallel_tcn_models_te')()
	#getattr(model_collections, 'serial_tcn_models_te')()
	getattr(model_collections, 'parallel_atcn_models_te')()
	#getattr(model_collections, 'serial_atcn_models_te')()
	print(model_collections)

	###################################################################################################################################################
	### LOSS & METRICS
	from lcclassifier.losses import LCMSEReconstruction, LCXEntropy, LCCompleteLoss
	from lcclassifier.metrics import LCXEntropyMetric, LCAccuracy

	loss_kwargs = {
		'model_output_is_with_softmax':False,
		'target_is_onehot':False,
	}
	#pre_loss = LCCompleteLoss('wmse', lcdataset['raw'].band_names)
	#pre_loss = LCCrossEntropy('wxentropy', **loss_kwargs)
	pre_loss = LCCompleteLoss('wmse-wxentropy', lcdataset['raw'].band_names, **loss_kwargs)

	pre_metrics = [
		LCXEntropyMetric('wxentropy', **loss_kwargs),
		LCAccuracy('b-accuracy', balanced=True, **loss_kwargs),
		LCAccuracy('accuracy', **loss_kwargs),
	]

	###################################################################################################################################################
	import os
	if main_args.gpu>=0:
		os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # see issue #152
		os.environ['CUDA_VISIBLE_DEVICES'] = str(main_args.gpu) # CUDA-GPU

	###################################################################################################################################################
	### GRID
	from lcclassifier.datasets import CustomDataset
	from lcclassifier.dataloaders import CustomDataLoader

	for mp_grid in model_collections.pms:
		### DATASETS
		dataset_kwargs = mp_grid['dataset_kwargs']
		
		s_train_dataset = CustomDataset(lcdataset, 'train+val.mcmc', **dataset_kwargs)
		s_val_dataset = CustomDataset(lcdataset, 'val', **dataset_kwargs)
		r_train_dataset = CustomDataset(lcdataset, 'train', **dataset_kwargs)
		r_val_dataset = CustomDataset(lcdataset, 'val', **dataset_kwargs)
		
		mp_grid['mdl_kwargs']['curvelength_max'] = s_train_dataset.get_max_len()
		mp_grid['dec_mdl_kwargs']['curvelength_max'] = s_train_dataset.get_max_len()
		s_train_dataset.transfer_to(s_val_dataset) # transfer information to val/test
		s_train_dataset.transfer_to(r_train_dataset) # transfer information to val/test
		s_train_dataset.transfer_to(r_val_dataset) # transfer information to val/test

		print('s_train_dataset:', s_train_dataset)
		print('s_val_dataset:', s_val_dataset)
		print('r_train_dataset:', r_train_dataset)
		print('r_val_dataset:', r_val_dataset)
		
		s_train_dataset.generate_daugm_samples(1)
		s_val_dataset.generate_daugm_samples(1)
		r_train_dataset.generate_daugm_samples(100)
		r_val_dataset.generate_daugm_samples(100)
		
		### DATALOADERS
		loader_kwargs = {
			'batch_size':512,
			#'num_workers':2, # bug?
		}
		random_subcrops = 3
		s_train_loader = CustomDataLoader(s_train_dataset, random_subcrops=random_subcrops, shuffle=True, **loader_kwargs)
		s_val_loader = CustomDataLoader(s_val_dataset, random_subcrops=random_subcrops, shuffle=True, **loader_kwargs)
		r_train_loader = CustomDataLoader(r_train_dataset, random_subcrops=random_subcrops, shuffle=True, **loader_kwargs)
		r_val_loader = CustomDataLoader(r_val_dataset, random_subcrops=0, **loader_kwargs)
		
		### IDS
		model_ids = range(0, 5)
		for ki,model_id in enumerate(model_ids): # IDS
			### GET MODEL
			mdl_kwargs = mp_grid['mdl_kwargs']
			for k_ in ['mdl_kwargs', 'dec_mdl_kwargs']:
				mp_grid[k_]['input_dims'] = s_train_loader.dataset.get_output_dims()
				mp_grid[k_]['te_features'] = s_train_loader.dataset.get_te_features_dims()
				mp_grid[k_]['curvelength_max'] = s_train_dataset.get_max_len()
				
			model = mdl_kwargs['C'](**mp_grid)
			
			### OPTIMIZER
			import torch.optim as optims
			from fuzzytorch.optimizers import LossOptimizer

			pre_optimizer_kwargs = {
				'opt_kwargs':{
					'lr':1e-3,
				},
				'decay_kwargs':{
					'lr':0.9,
				}
			}
			pre_optimizer = LossOptimizer(model, optims.Adam, **pre_optimizer_kwargs)

			### MONITORS
			from flamingchoripan.prints import print_bar
			from fuzzytorch.handlers import ModelTrainHandler
			from fuzzytorch.monitors import LossMonitor
			from fuzzytorch import C_
			import math

			monitor_config = {
				'val_epoch_counter_duration':1, # every k epochs check
				'earlystop_epoch_duration':20,
				#'save_mode':C_.SM_NO_SAVE,
				#'save_mode':C_.SM_ALL,
				#'save_mode':C_.SM_ONLY_ALL,
				'save_mode':C_.SM_ONLY_INF_METRIC,
				#'save_mode':C_.SM_ONLY_INF_LOSS,
				#'save_mode':C_.SM_ONLY_SUP_METRIC,
			}
			pre_loss_monitors = LossMonitor(pre_loss, pre_optimizer, pre_metrics, **monitor_config)
			
			### TRAIN
			mtrain_config = {
				'id':model_id,
				'epochs_max':1e5,
				'save_rootdir':f'../save/training',
				'extra_model_name_dict':{
					'mode':'s',
					#'ef-be':f'1e{math.log10(s_train_loader.dataset.effective_beta_eps)}',
					'ef-be':s_train_loader.dataset.effective_beta_eps,
				},
				'uses_train_eval_loader_methods':True,
			}
			model_train_handler = ModelTrainHandler(model, pre_loss_monitors, **mtrain_config)
			model_train_handler.build_gpu(0 if main_args.gpu>=0 else None)
			if ki==0:
				print(model_train_handler)
			model_train_handler.fit_loader(s_train_loader, s_val_loader)

			###################################################################################################################################################
			import fuzzytorch
			import fuzzytorch.plots
			import fuzzytorch.plots.training as ffplots

			### training plots
			plot_kwargs = {
				'save_rootdir':f'../save/train_plots',
			}
			ffplots.plot_loss(model_train_handler, **plot_kwargs)
			#ffplots.plot_evaluation_loss(train_handler, **plot_kwargs)
			#ffplots.plot_evaluation_metrics(train_handler, **plot_kwargs)

			###################################################################################################################################################
			import lcclassifier.experiments.images as exp_img

			### perform the experiments
			exp_kwargs = {
				'save_rootdir':f'../save/experiments',
				'm':4,
				'send_email':0,
			}
			#exp_img.reconstructions(model_train_handler, s_train_loader, **exp_kwargs)
			exp_img.reconstructions(model_train_handler, s_val_loader, **exp_kwargs)

			###################################################################################################################################################
			import lcclassifier.experiments.performance as exp_perf

			### perform the experiments
			exp_kwargs = {
				'save_rootdir':f'../save/experiments',
				'target_is_onehot':False,
				'send_email':0,
			}
			#exp_perf.reconstruction_along_days(model_train_handler, s_train_loader, **exp_kwargs) # over real
			exp_perf.reconstruction_along_days(model_train_handler, s_val_loader, **exp_kwargs) # over real

			###################################################################################################################################################
			import lcclassifier.experiments.performance as exp_perf

			### perform the experiments
			exp_kwargs = {
				'save_rootdir':f'../save/experiments',
				'target_is_onehot':False,
				'send_email':0,
			}
			#exp_perf.metrics_along_days(model_train_handler, s_val_loader, **exp_kwargs) # over real
			exp_perf.metrics_along_days(model_train_handler, s_val_loader, **exp_kwargs) # over real