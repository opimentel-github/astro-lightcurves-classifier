#!/usr/bin/env python3
# -*- coding: utf-8 -*
import sys
sys.path.append('../') # or just install the module
sys.path.append('../../fuzzy-torch') # or just install the module
sys.path.append('../../flaming-choripan') # or just install the module
sys.path.append('../../astro-lightcurves-handler') # or just install the module

if __name__== '__main__':
	### parser arguments
	import argparse
	from flamingchoripan.prints import print_big_bar

	parser = argparse.ArgumentParser('usage description')
	parser.add_argument('-method',  type=str, default='spm-mcmc-estw', help='method')
	parser.add_argument('-gpu',  type=int, default=-1, help='gpu')
	parser.add_argument('-mc',  type=str, default='parallel_rnn_models', help='model_collections method')
	parser.add_argument('-batch_size',  type=int, default=512, help='batch_size')
	parser.add_argument('-load_model',  type=bool, default=False, help='load_model')
	parser.add_argument('-epochs_max',  type=int, default=1e4, help='epochs_max')
	parser.add_argument('-save_rootdir',  type=str, default='../save', help='save_rootdir')
	parser.add_argument('-iid',  type=int, default=0, help='initial id')
	parser.add_argument('-fid',  type=int, default=5, help='final id')
	parser.add_argument('-kf',  type=str, default='0', help='kf')
	parser.add_argument('-rsc',  type=int, default=3, help='random_subcrops')
	parser.add_argument('-upc',  type=int, default=True, help='precompute')
	#main_args = parser.parse_args([])
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

	filedir = f'../../surveys-save/alerceZTFv7.1/survey=alerceZTFv7.1°bands=gr°mode=onlySNe°method={main_args.method}.splcds'
	filedict = get_dict_from_filedir(filedir)
	root_folder = filedict['*rootdir*']
	cfilename = filedict['*cfilename*']
	survey = filedict['survey']
	lcdataset = load_pickle(filedir)
	print(lcdataset)

	###################################################################################################################################################
	from lcclassifier.models.model_collections import ModelCollections

	model_collections = ModelCollections(lcdataset)
	getattr(model_collections, main_args.mc)()
	#getattr(model_collections, 'parallel_rnn_models_dt')()
	#getattr(model_collections, 'parallel_rnn_models_te')()
	#getattr(model_collections, 'serial_rnn_models_dt')()
	#getattr(model_collections, 'serial_rnn_models_te')()
	#getattr(model_collections, 'parallel_tcnn_models_dt')()
	#getattr(model_collections, 'parallel_tcnn_models_te')()
	#getattr(model_collections, 'serial_tcnn_models_dt')()
	#getattr(model_collections, 'serial_tcnn_models_te')()
	#getattr(model_collections, 'parallel_atcnn_models_te')()
	#getattr(model_collections, 'serial_atcnn_models_te')()

	###################################################################################################################################################
	### LOSS & METRICS
	from lcclassifier.losses import LCMSEReconstruction, LCXEntropy, LCCompleteLoss
	from lcclassifier.metrics import LCXEntropyMetric, LCAccuracy

	loss_kwargs = {
		'model_output_is_with_softmax':False,
		'target_is_onehot':False,
		'uses_poblation_weights':True,
	}
	#pt_loss = LCCompleteLoss('wmse', lcdataset['raw'].band_names)
	#pt_loss = LCXEntropy('wxentropy', **loss_kwargs)
	pt_loss = LCCompleteLoss('wmse-xentropy', lcdataset['raw'].band_names, **loss_kwargs)
	pt_metrics = [
		LCXEntropyMetric('xentropy', **loss_kwargs),
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

		s_train_dataset = CustomDataset(lcdataset, f'{main_args.kf}@train.{main_args.method}', **dataset_kwargs)
		s_val_dataset = CustomDataset(lcdataset, f'{main_args.kf}@val.{main_args.method}', **dataset_kwargs)
		r_train_dataset = CustomDataset(lcdataset, f'{main_args.kf}@train', **dataset_kwargs)
		r_val_dataset = CustomDataset(lcdataset, f'{main_args.kf}@val', **dataset_kwargs)

		mp_grid['mdl_kwargs']['curvelength_max'] = s_train_dataset.get_max_len()
		mp_grid['dec_mdl_kwargs']['curvelength_max'] = s_train_dataset.get_max_len()
		s_train_dataset.transfer_metadata_to(s_val_dataset) # transfer metadata to val/test
		s_train_dataset.transfer_metadata_to(r_train_dataset) # transfer metadata to val/test
		s_train_dataset.transfer_metadata_to(r_val_dataset) # transfer metadata to val/test

		print('s_train_dataset:', s_train_dataset)
		print('s_val_dataset:', s_val_dataset)
		print('r_train_dataset:', r_train_dataset)
		print('r_val_dataset:', r_val_dataset)
		
		if main_args.upc:
			synth_precomputed_samples = 5
			real_precomputed_samples = synth_precomputed_samples*1
			s_train_dataset.precompute_samples(synth_precomputed_samples)
			s_val_dataset.precompute_samples(synth_precomputed_samples)
			r_train_dataset.precompute_samples(real_precomputed_samples)
			r_val_dataset.precompute_samples(real_precomputed_samples)

		### DATALOADERS
		loader_kwargs = {
			#'num_workers':2, # bug?
			'batch_size':main_args.batch_size,
			'random_subcrops':main_args.rsc,
		}
		s_train_loader = CustomDataLoader(s_train_dataset, shuffle=True, **loader_kwargs)
		s_val_loader = CustomDataLoader(s_val_dataset, shuffle=False, **loader_kwargs)
		r_train_loader = CustomDataLoader(r_train_dataset, shuffle=True, **loader_kwargs)
		r_val_loader = CustomDataLoader(r_val_dataset, shuffle=False, **loader_kwargs)

		### IDS
		model_ids = list(range(main_args.iid, main_args.fid+1))
		for ki,model_id in enumerate(model_ids): # IDS
			### GET MODEL
			mdl_kwargs = mp_grid['mdl_kwargs']
			for k_ in ['mdl_kwargs', 'dec_mdl_kwargs']:
				mp_grid[k_]['input_dims'] = s_train_loader.dataset.get_output_dims()
				mp_grid[k_]['te_features'] = s_train_loader.dataset.get_te_features_dims()
				#mp_grid[k_]['curvelength_max'] = s_train_dataset.get_max_len()

			model = mdl_kwargs['C'](**mp_grid)

			### OPTIMIZER
			import torch.optim as optims
			from fuzzytorch.optimizers import LossOptimizer

			pt_optimizer_kwargs = {
				'opt_kwargs':{
					'lr':.5e-3,
				},
				#'decay_kwargs':{
				#	'lr':.95,
				#}
			}
			pt_optimizer = LossOptimizer(model, optims.Adam, **pt_optimizer_kwargs)

			### MONITORS
			from flamingchoripan.prints import print_bar
			from fuzzytorch.handlers import ModelTrainHandler
			from fuzzytorch.monitors import LossMonitor
			from fuzzytorch import C_
			import math

			monitor_config = {
				'val_epoch_counter_duration':2, # every k epochs check
				'earlystop_epoch_duration':20,
				#'save_mode':C_.SM_NO_SAVE,
				#'save_mode':C_.SM_ALL,
				#'save_mode':C_.SM_ONLY_ALL,
				#'save_mode':C_.SM_ONLY_INF_METRIC,
				'save_mode':C_.SM_ONLY_INF_LOSS,
				#'save_mode':C_.SM_ONLY_SUP_METRIC,
			}
			pt_loss_monitors = LossMonitor(pt_loss, pt_optimizer, pt_metrics, **monitor_config)

			### TRAIN
			mtrain_config = {
				'id':model_id,
				'epochs_max':1e5,
				'save_rootdir':f'../save/training',
				'extra_model_name_dict':{
					'mode':'pt',
					#'ef-be':f'1e{math.log10(s_train_loader.dataset.effective_beta_eps)}',
					'ef-be':s_train_loader.dataset.effective_beta_eps,
					'rsc':main_args.rsc,
				},
				'uses_train_eval_loader_methods':True,
			}
			model_train_handler = ModelTrainHandler(model, pt_loss_monitors, **mtrain_config)
			model_train_handler.build_gpu(0 if main_args.gpu>=0 else None)
			if ki==0:
				print(model_train_handler)
			model_train_handler.fit_loader(s_train_loader, s_val_loader) # main fit


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

			exp_kwargs = {
				'm':4,
			}
			exp_img.reconstructions_m(model_train_handler, s_train_loader, save_rootdir=f'../save/experiments/train', **exp_kwargs)
			exp_img.reconstructions_m(model_train_handler, s_val_loader, save_rootdir=f'../save/experiments/val', **exp_kwargs)

			###################################################################################################################################################
			import lcclassifier.experiments.performance as exp_perf

			### perform the experiments
			exp_kwargs = {
				'target_is_onehot':False,
			}
			#exp_perf.reconstruction_along_days(model_train_handler, s_train_loader, **exp_kwargs)
			exp_perf.reconstruction_along_days(model_train_handler, s_val_loader, save_rootdir=f'../save/experiments/val', **exp_kwargs)

			###################################################################################################################################################
			import lcclassifier.experiments.performance as exp_perf

			exp_kwargs = {
				'target_is_onehot':False,
			}
			#exp_perf.metrics_along_days(model_train_handler, s_val_loader, **exp_kwargs)
			exp_perf.metrics_along_days(model_train_handler, s_val_loader, save_rootdir=f'../save/experiments/val', **exp_kwargs)