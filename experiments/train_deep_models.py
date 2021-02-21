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
	parser.add_argument('-batch_size',  type=int, default=128, help='batch_size')
	parser.add_argument('-load_model',  type=bool, default=False, help='load_model')
	parser.add_argument('-epochs_max',  type=int, default=1e4, help='epochs_max')
	parser.add_argument('-save_rootdir',  type=str, default='../save', help='save_rootdir')
	parser.add_argument('-mids',  type=str, default='0-5', help='initial_id-final_id')
	parser.add_argument('-kf',  type=str, default='0', help='kf')
	parser.add_argument('-rsc',  type=int, default=0, help='random_subcrops')
	parser.add_argument('-upc',  type=int, default=True, help='uses_precompute')
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
	#getattr(model_collections, 'parallel_rnn_models_te')() # not used
	#getattr(model_collections, 'serial_rnn_models_dt')()
	#getattr(model_collections, 'serial_rnn_models_te')() # not used

	#getattr(model_collections, 'parallel_tcnn_models_dt')()
	#getattr(model_collections, 'parallel_tcnn_models_te')() # not used
	#getattr(model_collections, 'serial_tcnn_models_dt')()
	#getattr(model_collections, 'serial_tcnn_models_te')() # not used

	#getattr(model_collections, 'parallel_atcnn_models_te')()
	#getattr(model_collections, 'serial_atcnn_models_te')()

	print(model_collections)

	###################################################################################################################################################
	### LOSS & METRICS
	from lcclassifier.losses import LCMSEReconstruction, LCXEntropy, LCCompleteLoss
	from lcclassifier.metrics import LCXEntropyMetric, LCAccuracy

	pt_loss_kwargs = {
		'model_output_is_with_softmax':False,
		'target_is_onehot':False,
		'uses_poblation_weights':False, # False True
		'classifier_key':'y.last',
	}
	pt_loss = LCCompleteLoss('wmse-xentropy', lcdataset['raw'].band_names, **pt_loss_kwargs)
	pt_metrics = [
		LCXEntropyMetric('xentropy', **pt_loss_kwargs),
		LCAccuracy('b-accuracy', balanced=True, **pt_loss_kwargs),
		LCAccuracy('accuracy', **pt_loss_kwargs),
	]

	ft_loss_kwargs = {
		'model_output_is_with_softmax':False,
		'target_is_onehot':False,
		'uses_poblation_weights':False, # False True
		'classifier_key':'y.last-ft',
	}
	ft_loss = LCXEntropy('xentropy', **ft_loss_kwargs)
	ft_metrics = [
		LCXEntropyMetric('xentropy', **ft_loss_kwargs),
		LCAccuracy('b-accuracy', balanced=True, **ft_loss_kwargs),
		LCAccuracy('accuracy', **ft_loss_kwargs),
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

	previous_dataset_kwargs = None
	for mp_grid in model_collections.mps:
		### DATASETS
		dataset_kwargs = mp_grid['dataset_kwargs']

		s_train_dataset = CustomDataset(lcdataset, f'{main_args.kf}@train.{main_args.method}', **dataset_kwargs)
		s_val_dataset = CustomDataset(lcdataset, f'{main_args.kf}@val.{main_args.method}', **dataset_kwargs)
		r_train_dataset = CustomDataset(lcdataset, f'{main_args.kf}@train', **dataset_kwargs)
		r_val_dataset = CustomDataset(lcdataset, f'{main_args.kf}@val', **dataset_kwargs)
		r_test_dataset = CustomDataset(lcdataset, f'{main_args.kf}@test', **dataset_kwargs)

		mp_grid['mdl_kwargs']['curvelength_max'] = s_train_dataset.get_max_len()
		mp_grid['dec_mdl_kwargs']['curvelength_max'] = s_train_dataset.get_max_len()
		s_train_dataset.transfer_metadata_to(s_val_dataset) # transfer metadata to val/test
		s_train_dataset.transfer_metadata_to(r_train_dataset) # transfer metadata to val/test
		s_train_dataset.transfer_metadata_to(r_val_dataset) # transfer metadata to val/test
		s_train_dataset.transfer_metadata_to(r_test_dataset) # transfer metadata to val/test

		print('s_train_dataset:', s_train_dataset)
		print('s_val_dataset:', s_val_dataset)
		print('r_train_dataset:', r_train_dataset)
		print('r_val_dataset:', r_val_dataset)
		print('r_test_dataset:', r_test_dataset)
		
		if main_args.upc and 0:
			if previous_dataset_kwargs is None or not dataset_kwargs==previous_dataset_kwargs:
				synth_precomputed_samples = 1
				real_precomputed_samples = 1
				s_train_dataset.precompute_samples(synth_precomputed_samples)
				s_val_dataset.precompute_samples(synth_precomputed_samples)
				r_train_dataset.precompute_samples(real_precomputed_samples)
				r_val_dataset.precompute_samples(real_precomputed_samples)
				#r_test_dataset.precompute_samples(real_precomputed_samples)

		### DATALOADERS
		loader_kwargs = {
			'num_workers':2,
			'pin_memory':False, # False True
			'prefetch_factor':1,
			'batch_size':main_args.batch_size,
			'random_subcrops':main_args.rsc,
		}
		s_train_loader = CustomDataLoader(s_train_dataset, shuffle=True, **loader_kwargs)
		s_val_loader = CustomDataLoader(s_val_dataset, shuffle=False, **loader_kwargs)
		r_train_loader = CustomDataLoader(r_train_dataset, shuffle=True, **loader_kwargs)
		r_val_loader = CustomDataLoader(r_val_dataset, shuffle=False, **loader_kwargs)
		r_test_loader = CustomDataLoader(r_test_dataset, shuffle=False, **loader_kwargs)
		previous_dataset_kwargs = dataset_kwargs.copy()

		### IDS
		ki, kf = [int(k) for k in main_args.mids.split('-')]
		model_ids = list(range(ki, kf))
		for ki,model_id in enumerate(model_ids): # IDS

			### GET MODEL
			mdl_kwargs = mp_grid['mdl_kwargs']
			for k_ in ['mdl_kwargs', 'dec_mdl_kwargs']:
				mp_grid[k_]['input_dims'] = s_train_loader.dataset.get_output_dims()
				mp_grid[k_]['te_features'] = s_train_loader.dataset.get_te_features_dims()
				#mp_grid[k_]['curvelength_max'] = s_train_dataset.get_max_len()
			model = mdl_kwargs['C'](**mp_grid)

			### pre-training
			### OPTIMIZER
			import torch.optim as optims
			from fuzzytorch.optimizers import LossOptimizer

			pt_optimizer_kwargs = {
				'opt_kwargs':{
					'lr':.75e-3,
					#'betas':(0.9999, 0.9999),
				},
				#'decay_kwargs':{
				#	'lr':.95,
				#}
			}
			pt_optimizer = LossOptimizer(model, optims.AdamW, **pt_optimizer_kwargs) # Adagrad Adadelta RMSprop Adam AdamW

			### MONITORS
			from flamingchoripan.prints import print_bar
			from fuzzytorch.handlers import ModelTrainHandler
			from fuzzytorch.monitors import LossMonitor
			from fuzzytorch import C_
			import math

			monitor_config = {
				'val_epoch_counter_duration':0, # every k epochs check
				'earlystop_epoch_duration':20, # 10 15 20 25 30
				'target_metric_crit':'b-accuracy',
				#'save_mode':C_.SM_NO_SAVE,
				#'save_mode':C_.SM_ALL,
				#'save_mode':C_.SM_ONLY_ALL,
				#'save_mode':C_.SM_ONLY_INF_METRIC,
				#'save_mode':C_.SM_ONLY_INF_LOSS,
				'save_mode':C_.SM_ONLY_SUP_METRIC,
			}
			pt_loss_monitors = LossMonitor(pt_loss, pt_optimizer, pt_metrics, **monitor_config)

			### TRAIN
			train_mode = 'pre-training'
			mtrain_config = {
				'id':model_id,
				'epochs_max':1e5,
				'save_rootdir':f'../save/training',
				'extra_model_name_dict':{
					'mode':train_mode,
					#'ef-be':f'1e{math.log10(s_train_loader.dataset.effective_beta_eps)}',
					#'ef-be':s_train_loader.dataset.effective_beta_eps,
					'rsc':main_args.rsc,
				},
				'uses_train_eval_loader_methods':True,
			}
			pt_model_train_handler = ModelTrainHandler(model, pt_loss_monitors, **mtrain_config)
			pt_model_train_handler.build_gpu(0 if main_args.gpu>=0 else None)
			if ki==0:
				print(pt_model_train_handler)
			pt_model_train_handler.fit_loader(s_train_loader, s_val_loader) # main fit

			###################################################################################################################################################
			import fuzzytorch
			import fuzzytorch.plots
			import fuzzytorch.plots.training as ffplots

			### training plots
			plot_kwargs = {
				'save_rootdir':f'../save/train_plots',
			}
			ffplots.plot_loss(pt_model_train_handler, **plot_kwargs)
			#ffplots.plot_evaluation_loss(train_handler, **plot_kwargs)
			#ffplots.plot_evaluation_metrics(train_handler, **plot_kwargs)

			###################################################################################################################################################
			from lcclassifier.experiments.images import reconstructions_m

			pt_exp_kwargs = {
				'm':15,
				'target_is_onehot':False,
				'classifier_key':'y.last',
			}
			reconstructions_m(pt_model_train_handler, s_train_loader, save_rootdir=f'../save/experiments/{main_args.kf}@s_train/{train_mode}', **pt_exp_kwargs) # sanity check / slow
			reconstructions_m(pt_model_train_handler, r_train_loader, save_rootdir=f'../save/experiments/{main_args.kf}@r_train/{train_mode}', **pt_exp_kwargs) # sanity check
			reconstructions_m(pt_model_train_handler, s_val_loader, save_rootdir=f'../save/experiments/{main_args.kf}@s_val/{train_mode}', **pt_exp_kwargs)
			reconstructions_m(pt_model_train_handler, r_val_loader, save_rootdir=f'../save/experiments/{main_args.kf}@r_val/{train_mode}', **pt_exp_kwargs)
			reconstructions_m(pt_model_train_handler, r_test_loader, save_rootdir=f'../save/experiments/{main_args.kf}@r_test/{train_mode}', **pt_exp_kwargs)

			###################################################################################################################################################
			from lcclassifier.experiments.performance import metrics_along_days
			from lcclassifier.experiments.attention import attention_statistics

			if model_id==model_ids[-1]:
				attention_statistics(pt_model_train_handler, s_train_loader, save_rootdir=f'../save/experiments/{main_args.kf}@s_train/{train_mode}', **pt_exp_kwargs) # slow
				attention_statistics(pt_model_train_handler, s_val_loader, save_rootdir=f'../save/experiments/{main_args.kf}@s_val/{train_mode}', **pt_exp_kwargs)
				pass

			#metrics_along_days(pt_model_train_handler, s_train_loader, save_rootdir=f'../save/experiments/{main_args.kf}@s_train/{train_mode}', **pt_exp_kwargs) # sanity check / slow
			metrics_along_days(pt_model_train_handler, r_train_loader, save_rootdir=f'../save/experiments/{main_args.kf}@r_train/{train_mode}', **pt_exp_kwargs) # sanity check
			metrics_along_days(pt_model_train_handler, s_val_loader, save_rootdir=f'../save/experiments/{main_args.kf}@s_val/{train_mode}', **pt_exp_kwargs)
			metrics_along_days(pt_model_train_handler, r_val_loader, save_rootdir=f'../save/experiments/{main_args.kf}@r_val/{train_mode}', **pt_exp_kwargs)
			metrics_along_days(pt_model_train_handler, r_test_loader, save_rootdir=f'../save/experiments/{main_args.kf}@r_test/{train_mode}', **pt_exp_kwargs)

			### fine-tuning
			### OPTIMIZER
			import torch.optim as optims
			from fuzzytorch.optimizers import LossOptimizer

			ft_optimizer_kwargs = {
				'opt_kwargs':{
					'lr':1.1e-2,
				},
				#'decay_kwargs':{
				#	'lr':.95,
				#}
			}
			ft_optimizer = LossOptimizer(model.get_classifier_model(), optims.SGD, **ft_optimizer_kwargs)

			### MONITORS
			from flamingchoripan.prints import print_bar
			from fuzzytorch.handlers import ModelTrainHandler
			from fuzzytorch.monitors import LossMonitor
			from fuzzytorch import C_
			import math

			monitor_config = {
				'val_epoch_counter_duration':0, # every k epochs check
				'earlystop_epoch_duration':150,
				'target_metric_crit':'b-accuracy',
				#'save_mode':C_.SM_NO_SAVE,
				#'save_mode':C_.SM_ALL,
				#'save_mode':C_.SM_ONLY_ALL,
				#'save_mode':C_.SM_ONLY_INF_METRIC,
				#'save_mode':C_.SM_ONLY_INF_LOSS,
				'save_mode':C_.SM_ONLY_SUP_METRIC,
			}
			ft_loss_monitors = LossMonitor(ft_loss, ft_optimizer, ft_metrics, **monitor_config)

			### TRAIN
			train_mode = 'fine-tuning'
			mtrain_config = {
				'id':model_id,
				'epochs_max':1e5,
				'save_rootdir':f'../save/training',
				'extra_model_name_dict':{
					'mode':train_mode,
					#'ef-be':f'1e{math.log10(s_train_loader.dataset.effective_beta_eps)}',
					#'ef-be':s_train_loader.dataset.effective_beta_eps,
					'rsc':main_args.rsc,
				},
				'uses_train_eval_loader_methods':True,
			}
			ft_model_train_handler = ModelTrainHandler(model, ft_loss_monitors, **mtrain_config)
			ft_model_train_handler.build_gpu(0 if main_args.gpu>=0 else None)
			if ki==0:
				print(ft_model_train_handler)
			ft_model_train_handler.fit_loader(r_train_loader, r_val_loader) # main fit

			###################################################################################################################################################
			from lcclassifier.experiments.performance import metrics_along_days
			from lcclassifier.experiments.attention import attention_statistics

			ft_exp_kwargs = {
				'm':15,
				'target_is_onehot':False,
				'classifier_key':'y.last-ft',
			}
			if model_id==model_ids[-1]:
				attention_statistics(ft_model_train_handler, s_train_loader, save_rootdir=f'../save/experiments/{main_args.kf}@s_train/{train_mode}', **ft_exp_kwargs) # slow
				attention_statistics(ft_model_train_handler, s_val_loader, save_rootdir=f'../save/experiments/{main_args.kf}@s_val/{train_mode}', **ft_exp_kwargs)
				pass
			
			#metrics_along_days(pt_model_train_handler, s_train_loader, save_rootdir=f'../save/experiments/{main_args.kf}@s_train/{train_mode}', **ft_exp_kwargs) # sanity check
			metrics_along_days(ft_model_train_handler, r_train_loader, save_rootdir=f'../save/experiments/{main_args.kf}@r_train/{train_mode}', **ft_exp_kwargs) # sanity check
			metrics_along_days(ft_model_train_handler, s_val_loader, save_rootdir=f'../save/experiments/{main_args.kf}@s_val/{train_mode}', **ft_exp_kwargs)
			metrics_along_days(ft_model_train_handler, r_val_loader, save_rootdir=f'../save/experiments/{main_args.kf}@r_val/{train_mode}', **ft_exp_kwargs)
			metrics_along_days(ft_model_train_handler, r_test_loader, save_rootdir=f'../save/experiments/{main_args.kf}@r_test/{train_mode}', **ft_exp_kwargs)