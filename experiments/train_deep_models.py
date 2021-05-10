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
	parser.add_argument('-batch_size',  type=int, default=120, help='batch_size') # *** 50 100 200
	parser.add_argument('-batch_size_c',  type=int, default=16, help='batch_size')
	parser.add_argument('-load_model',  type=bool, default=False, help='load_model')
	parser.add_argument('-epochs_max',  type=int, default=1e4, help='epochs_max')
	parser.add_argument('-save_rootdir',  type=str, default='../save', help='save_rootdir')
	parser.add_argument('-mids',  type=str, default='0-10', help='initial_id-final_id')
	parser.add_argument('-kf',  type=str, default='0', help='kf')
	parser.add_argument('-rsc',  type=int, default=0, help='random_subcrops')
	parser.add_argument('-bypass',  type=int, default=0, help='bypass')
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

	filedir = f'../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method={main_args.method}.splcds'
	filedict = get_dict_from_filedir(filedir)
	rootdir = filedict['_rootdir']
	cfilename = filedict['_cfilename']
	lcdataset = load_pickle(filedir)
	#print(lcdataset)

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
	from lcclassifier.losses import LCMSEReconstruction, LCXEntropy, LCCompleteLoss, LCBinXEntropy
	from lcclassifier.metrics import LCWMSE, LCXEntropyMetric, LCAccuracy

	pt_loss_kwargs = {
		'band_names':lcdataset['raw'].band_names,
		'model_output_is_with_softmax':False,
		'target_is_onehot':False,
		'classifier_key':'y_last_pt',
		}
	pt_loss = LCCompleteLoss('wmse-xentropy', **pt_loss_kwargs)
	pt_metrics = [
		LCWMSE('b-wmse', balanced=True, **pt_loss_kwargs),
		LCXEntropyMetric('b-xentropy', balanced=True, **pt_loss_kwargs),
		LCAccuracy('b-accuracy', balanced=True, **pt_loss_kwargs),
		]

	ft_loss_kwargs = {
		'class_names':lcdataset['raw'].class_names,
		'model_output_is_with_softmax':False,
		'model_output_is_with_sigmoid':False,
		'target_is_onehot':False,
		'classifier_key':'y_last_ft',
		}
	#ft_loss = LCXEntropy('xentropy', **ft_loss_kwargs)
	ft_loss = LCBinXEntropy('bin-xentropy', **ft_loss_kwargs)
	ft_metrics = [
		LCXEntropyMetric('b-xentropy', balanced=True, **ft_loss_kwargs),
		LCAccuracy('b-accuracy', balanced=True, **ft_loss_kwargs),
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
	from torch.utils.data import DataLoader
	import torch

	model_ids = list(range(*[int(mi) for mi in main_args.mids.split('-')])) # IDS
	for kmodel_id,model_id in enumerate(model_ids):

		for mp_grid in model_collections.mps: # MODEL CONFIGS
			### DATASETS
			dataset_kwargs = mp_grid['dataset_kwargs']
			s_balanced_repeats = 4
			r_balanced_repeats = s_balanced_repeats*12
			if main_args.bypass:
				s_train_dataset = CustomDataset(f'{main_args.kf}@train', lcdataset, **dataset_kwargs, balanced_repeats=r_balanced_repeats)
			else:
				s_train_dataset = CustomDataset(f'{main_args.kf}@train.{main_args.method}', lcdataset, **dataset_kwargs, balanced_repeats=s_balanced_repeats)
			r_train_dataset = CustomDataset(f'{main_args.kf}@train', lcdataset, **dataset_kwargs, balanced_repeats=r_balanced_repeats)
			r_val_dataset = CustomDataset(f'{main_args.kf}@val', lcdataset, **dataset_kwargs)
			r_test_dataset = CustomDataset(f'{main_args.kf}@test', lcdataset, **dataset_kwargs)

			mp_grid['mdl_kwargs']['curvelength_max'] = s_train_dataset.get_max_len()
			s_train_dataset.transfer_metadata_to(r_train_dataset) # transfer metadata to val/test
			s_train_dataset.transfer_metadata_to(r_val_dataset) # transfer metadata to val/test
			s_train_dataset.transfer_metadata_to(r_test_dataset) # transfer metadata to val/test

			s_precomputed_samples = 0 # *** 0* 5 10 15
			r_precomputed_samples = 0 # *** 0*
			s_train_dataset.precompute_samples(s_precomputed_samples)
			r_train_dataset.precompute_samples(r_precomputed_samples)

			print('s_train_dataset:', s_train_dataset)
			print('r_train_dataset:', r_train_dataset)
			print('r_val_dataset:', r_val_dataset)
			print('r_test_dataset:', r_test_dataset)

			### DATALOADERS
			worker_init_fn = lambda id:np.random.seed(torch.initial_seed() // 2**32+id) # num_workers-numpy bug
			loader_kwargs = {
				'num_workers':2, # 0 2*
				'pin_memory':True, # False True
				#'prefetch_factor':1, # only if num_workers>0
				'worker_init_fn':worker_init_fn,
				'batch_size':main_args.batch_size//(1+main_args.rsc),
				'random_subcrops':main_args.rsc,
				}
			s_train_loader = CustomDataLoader(s_train_dataset, shuffle=True, **loader_kwargs) # DataLoader CustomDataLoader
			loader_kwargs.update({
				'batch_size':main_args.batch_size_c//(1+main_args.rsc),
				'random_subcrops':main_args.rsc,
				})
			r_train_loader = CustomDataLoader(r_train_dataset, shuffle=True, **loader_kwargs) # DataLoader CustomDataLoader
			r_val_loader = CustomDataLoader(r_val_dataset, shuffle=False, **loader_kwargs) # DataLoader CustomDataLoader
			r_test_loader = CustomDataLoader(r_test_dataset, shuffle=False, **loader_kwargs) # DataLoader CustomDataLoader

			### GET MODEL
			mp_grid['mdl_kwargs']['input_dims'] = s_train_loader.dataset.get_output_dims()
			model = mp_grid['mdl_kwargs']['C'](**mp_grid)

			### pre-training
			### OPTIMIZER
			import torch.optim as optims
			from fuzzytorch.optimizers import LossOptimizer

			def pt_lr_f(epoch):
				initial_lr = 1e-6
				max_lr = 1*1e-3
				d_epochs = 50
				p = np.clip(epoch/d_epochs, 0, 1)
				return initial_lr+p*(max_lr-initial_lr)

			pt_opt_kwargs_f = {
				#'lr':lambda epoch:2.e-3, # ***
				'lr':pt_lr_f, # ***
				}
			pt_optimizer_kwargs = {
				'clip_grad':1.,
				}
			pt_optimizer = LossOptimizer(model, optims.Adam, pt_opt_kwargs_f, **pt_optimizer_kwargs) # SGD Adagrad Adadelta RMSprop Adam AdamW

			### MONITORS
			from flamingchoripan.prints import print_bar
			from fuzzytorch.handlers import ModelTrainHandler
			from fuzzytorch.monitors import LossMonitor
			from fuzzytorch import C_
			import math

			val_epoch_counter_duration = 2
			monitor_config = {
				'val_epoch_counter_duration':val_epoch_counter_duration, # every k epochs check
				'earlystop_epoch_duration':1e6,
				'target_metric_crit':'b-wmse',
				#'save_mode':C_.SM_NO_SAVE,
				#'save_mode':C_.SM_ALL,
				#'save_mode':C_.SM_ONLY_ALL,
				'save_mode':C_.SM_ONLY_INF_METRIC,
				#'save_mode':C_.SM_ONLY_INF_LOSS,
				#'save_mode':C_.SM_ONLY_SUP_METRIC,
				}
			pt_loss_monitors = LossMonitor(pt_loss, pt_optimizer, pt_metrics, **monitor_config)

			### TRAIN
			train_mode = 'pre-training'
			mtrain_config = {
				'id':model_id,
				'epochs_max':2000, # limit this as the pre-training is very time consuming
				'extra_model_name_dict':{
					#'mode':train_mode,
					#'ef-be':f'1e{math.log10(s_train_loader.dataset.effective_beta_eps)}',
					#'ef-be':s_train_loader.dataset.effective_beta_eps,
					'b':main_args.batch_size,
					'rsc':main_args.rsc,
					'bypass':main_args.bypass,
					},
				'uses_train_eval_loader_methods':True,
				'evaluate_train':False, # False to speed up training
				}
			pt_model_train_handler = ModelTrainHandler(model, pt_loss_monitors, **mtrain_config)
			complete_model_name = pt_model_train_handler.get_complete_model_name()
			pt_model_train_handler.set_complete_save_roodir(f'../save/{complete_model_name}/{train_mode}/_training/{cfilename}/{main_args.kf}@train')
			pt_model_train_handler.build_gpu(0 if main_args.gpu>=0 else None)
			if kmodel_id==0:
				print(pt_model_train_handler)
			pt_model_train_handler.fit_loader(s_train_loader, r_val_loader) # main fit
			pt_model_train_handler.load_model() # important, refresh to best model

			###################################################################################################################################################
			import fuzzytorch
			import fuzzytorch.plots
			import fuzzytorch.plots.training as ffplots

			### training plots
			plot_kwargs = {
				'save_rootdir':f'../save/train_plots',
				}
			#ffplots.plot_loss(pt_model_train_handler, **plot_kwargs) # use this
			#ffplots.plot_evaluation_loss(train_handler, **plot_kwargs)
			#ffplots.plot_evaluation_metrics(train_handler, **plot_kwargs)

			###################################################################################################################################################
			from lcclassifier.experiments.reconstructions import save_reconstructions
			from lcclassifier.experiments.model_info import save_model_info
			from lcclassifier.experiments.temporal_encoding import save_temporal_encoding
			from lcclassifier.experiments.performance import save_performance
			from lcclassifier.experiments.attention_scores import save_attn_scores_animation
			from lcclassifier.experiments.attention_stats import save_attention_statistics

			### attention experiments
			save_attn_exps = 0 # kmodel_id==0
			if save_attn_exps:
				pt_exp_kwargs = {
					'm':2,
					}
				save_attn_scores_animation(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/attn_scores/{cfilename}', **pt_exp_kwargs) # sanity check / slow
				save_attn_scores_animation(pt_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/attn_scores/{cfilename}', **pt_exp_kwargs) # sanity check
				save_attn_scores_animation(pt_model_train_handler, r_val_loader, f'../save/{complete_model_name}/{train_mode}/attn_scores/{cfilename}', **pt_exp_kwargs)
				save_attn_scores_animation(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/attn_scores/{cfilename}', **pt_exp_kwargs)

				#save_attention_statistics(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/attn_stats/{cfilename}', **pt_exp_kwargs)

			pt_exp_kwargs = {
				'm':20,
				'target_is_onehot':False,
				'classifier_key':'y_last_pt',
				}
			if 1:
				save_reconstructions(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs) # sanity check / slow
				#save_reconstructions(pt_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs) # sanity check
				#save_reconstructions(pt_model_train_handler, r_val_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs)
				save_reconstructions(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs)

				save_model_info(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/model_info/{cfilename}', **pt_exp_kwargs)
				save_temporal_encoding(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/temporal_encoding/{cfilename}', **pt_exp_kwargs)
			
			###################################################################################################################################################
			### fine-tuning
			### OPTIMIZER
			import torch.optim as optims
			from fuzzytorch.optimizers import LossOptimizer

			def ft_lr_f(epoch):
				initial_lr = 1e-6
				max_lr = 1*1e-3
				d_epochs = 50
				p = np.clip(epoch/d_epochs, 0, 1)
				return initial_lr+p*(max_lr-initial_lr)

			ft_opt_kwargs_f = {
				#'lr':lambda epoch:1*1e-3, # ***
				'lr':ft_lr_f, # ***
				}
			ft_optimizer_kwargs = {
				'clip_grad':1.,
				}
			ft_optimizer = LossOptimizer(model.get_classifier_model(), optims.Adam, ft_opt_kwargs_f, **ft_optimizer_kwargs) # SGD Adagrad Adadelta RMSprop Adam AdamW

			### MONITORS
			from flamingchoripan.prints import print_bar
			from fuzzytorch.handlers import ModelTrainHandler
			from fuzzytorch.monitors import LossMonitor
			from fuzzytorch import C_
			import math

			monitor_config = {
				'val_epoch_counter_duration':0, # every k epochs check
				'earlystop_epoch_duration':1e6,
				'target_metric_crit':'b-xentropy',
				#'save_mode':C_.SM_NO_SAVE,
				#'save_mode':C_.SM_ALL,
				#'save_mode':C_.SM_ONLY_ALL,
				'save_mode':C_.SM_ONLY_INF_METRIC,
				#'save_mode':C_.SM_ONLY_INF_LOSS,
				#'save_mode':C_.SM_ONLY_SUP_METRIC, 
				}
			ft_loss_monitors = LossMonitor(ft_loss, ft_optimizer, ft_metrics, **monitor_config)

			### TRAIN
			train_mode = 'fine-tuning'
			mtrain_config = {
				'id':model_id,
				'epochs_max':200, # limit this as the pre-training is very time consuming 5 10 15 20 25 30
				'save_rootdir':f'../save/{train_mode}/_training/{cfilename}',
				'extra_model_name_dict':{
					#'mode':train_mode,
					#'ef-be':f'1e{math.log10(s_train_loader.dataset.effective_beta_eps)}',
					#'ef-be':s_train_loader.dataset.effective_beta_eps,
					'b':main_args.batch_size,
					'rsc':main_args.rsc,
					'bypass':main_args.bypass,
					},
				'uses_train_eval_loader_methods':True,
				'evaluate_train':False, # False to speed up training
				}
			ft_model_train_handler = ModelTrainHandler(model, ft_loss_monitors, **mtrain_config)
			complete_model_name = ft_model_train_handler.get_complete_model_name()
			ft_model_train_handler.set_complete_save_roodir(f'../save/{complete_model_name}/{train_mode}/_training/{cfilename}/{main_args.kf}@train')
			ft_model_train_handler.build_gpu(0 if main_args.gpu>=0 else None)
			if kmodel_id==0:
				print(ft_model_train_handler)
			ft_model_train_handler.fit_loader(r_train_loader, r_val_loader) # main fit
			ft_model_train_handler.load_model() # important, refresh to best model

			###################################################################################################################################################
			from lcclassifier.experiments.performance import save_performance
		
			ft_exp_kwargs = {
				'm':15,
				'target_is_onehot':False,
				'classifier_key':'y_last_ft',
				}	
			#save_performance(ft_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs) # sanity check / slow
			save_performance(ft_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs) # sanity check
			#save_performance(ft_model_train_handler, r_val_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs)
			save_performance(ft_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs)

			save_model_info(ft_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/model_info/{cfilename}', **pt_exp_kwargs)