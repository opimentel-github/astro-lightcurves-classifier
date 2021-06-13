#!/usr/bin/env python3
# -*- coding: utf-8 -*
import sys
sys.path.append('../') # or just install the module
sys.path.append('../../fuzzy-torch') # or just install the module
sys.path.append('../../fuzzy-tools') # or just install the module
sys.path.append('../../astro-lightcurves-handler') # or just install the module

###################################################################################################################################################
import argparse
from fuzzytools.prints import print_big_bar

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--method',  type=str, default='spm-mcmc-estw')
parser.add_argument('--gpu',  type=int, default=-1)
parser.add_argument('--mc',  type=str, default='parallel_rnn_models')
parser.add_argument('--batch_size',  type=int, default=100)
parser.add_argument('--save_rootdir',  type=str, default='../save')
parser.add_argument('--mid',  type=str, default='0')
parser.add_argument('--kf',  type=str, default='0')
parser.add_argument('--bypass',  type=int, default=0) # 0 1
parser.add_argument('--only_attn_exp',  type=int, default=0) # 0 1
parser.add_argument('--invert_mpg',  type=int, default=0) # 0 1
parser.add_argument('--extra_model_name',  type=str, default='')
parser.add_argument('--classifier_mids',  type=int, default=10)
parser.add_argument('--num_workers',  type=int, default=5) # 2 3 4 5
parser.add_argument('--pin_memory',  type=int, default=1) # 0 1
parser.add_argument('--balanced_metrics',  type=int, default=0) # 0 1 # critical
#main_args = parser.parse_args([])
main_args = parser.parse_args()
print_big_bar()

###################################################################################################################################################
from fuzzytools.files import search_for_filedirs
from lchandler import C_ as C_

surveys_rootdir = '../../surveys-save/'
filedirs = search_for_filedirs(surveys_rootdir, fext=C_.EXT_SPLIT_LIGHTCURVE)

###################################################################################################################################################
import numpy as np
from fuzzytools.files import load_pickle, save_pickle
from fuzzytools.files import get_dict_from_filedir

filedir = f'../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method={main_args.method}.splcds'
filedict = get_dict_from_filedir(filedir)
rootdir = filedict['_rootdir']
cfilename = filedict['_cfilename']
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
from lcclassifier.losses import LCMSEReconstruction, LCXEntropy, LCCompleteLoss, LCBinXEntropy
from lcclassifier.metrics import LCWMSE, LCXEntropyMetric, LCAccuracy

metric_prefix = 'b-' if main_args.balanced_metrics else ''
pt_loss_kwargs = {
	'band_names':lcdataset['raw'].band_names,
	'model_output_is_with_softmax':False,
	'target_is_onehot':False,
	'classifier_key':'y_last_pt',
	# 'classifier_key':'y_last_ft',
	}
pt_loss = LCCompleteLoss('wmse-xentropy', **pt_loss_kwargs)
pt_metrics = [
	LCWMSE(f'{metric_prefix}wmse', balanced=main_args.balanced_metrics, **pt_loss_kwargs),
	LCXEntropyMetric(f'{metric_prefix}xentropy', balanced=main_args.balanced_metrics, **pt_loss_kwargs),
	LCAccuracy(f'{metric_prefix}accuracy', balanced=main_args.balanced_metrics, **pt_loss_kwargs),
	]

### ft
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
	LCXEntropyMetric(f'{metric_prefix}xentropy', balanced=main_args.balanced_metrics, **ft_loss_kwargs),
	LCAccuracy(f'{metric_prefix}accuracy', balanced=main_args.balanced_metrics, **ft_loss_kwargs),
	]

###################################################################################################################################################
import os

if main_args.gpu>=0:
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # see issue #152
	os.environ['CUDA_VISIBLE_DEVICES'] = str(main_args.gpu) # CUDA-GPU
	device = 'cuda:0'
else:
	device = 'cpu'

###################################################################################################################################################
mp_grids = model_collections.mps[::-1] if main_args.invert_mpg else model_collections.mps
for mp_grid in mp_grids: # MODEL CONFIGS
	from lcclassifier.datasets import CustomDataset
	from torch.utils.data import DataLoader
	from fuzzytools.strings import get_dict_from_string
	from fuzzytools.files import get_filedirs, copy_filedir
	import torch
	from copy import copy, deepcopy

	### DATASETS
	dataset_kwargs = mp_grid['dataset_kwargs']
	bypass_autoencoder = 0
	# bypass_autoencoder = 1
	# ds_mode = {'random':1.0, 'left':0.0, 'none':0.0} # avoid none
	ds_mode = {'random':.9, 'left':.1, 'none':.0}
	ds_p = 10/100
	std_scale = 1/2

	lcset_name = f'{main_args.kf}@train.{main_args.method}' if not main_args.bypass else f'{main_args.kf}@train'
	# s_train_dataset_da = CustomDataset(lcset_name, copy(lcdataset[lcset_name]), device,
	# 	uses_daugm=False,
	# 	uses_dynamic_balance=True,
	# 	**dataset_kwargs)
	# s_train_loader_da = DataLoader(s_train_dataset_da,
	# 	shuffle=True,
	# 	drop_last=True,
	# 	batch_size=main_args.batch_size,
	# 	)
	s_train_dataset_da = CustomDataset(lcset_name, copy(lcdataset[lcset_name]), 'cpu',
		precomputed_copies=0,
		uses_daugm=True,
		uses_dynamic_balance=True,
		ds_mode=ds_mode,
		ds_p=ds_p,
		std_scale=std_scale,
		**dataset_kwargs)
	s_train_loader_da = DataLoader(s_train_dataset_da,
		shuffle=True,
		drop_last=True,
		batch_size=main_args.batch_size,
		num_workers=main_args.num_workers,
		pin_memory=main_args.pin_memory,
		# prefetch_factor=5,
		worker_init_fn=lambda id:np.random.seed(torch.initial_seed() // 2**32+id), # num_workers-numpy bug
		persistent_workers=main_args.num_workers>0,
		)

	lcset_name = f'{main_args.kf}@train.{main_args.method}'
	s_train_dataset = CustomDataset(lcset_name, copy(lcdataset[lcset_name]), device, **dataset_kwargs)
	s_train_loader = DataLoader(s_train_dataset, shuffle=False, batch_size=main_args.batch_size)

	lcset_name = f'{main_args.kf}@train'
	r_train_dataset = CustomDataset(lcset_name, copy(lcdataset[lcset_name]), device, **dataset_kwargs)
	r_train_loader = DataLoader(r_train_dataset, shuffle=False, batch_size=main_args.batch_size)

	lcset_name = f'{main_args.kf}@val'
	r_val_dataset = CustomDataset(lcset_name, copy(lcdataset[lcset_name]), device, **dataset_kwargs)
	r_val_loader = DataLoader(r_val_dataset, shuffle=False, batch_size=main_args.batch_size)

	lcset_name = f'{main_args.kf}@test'
	r_test_dataset = CustomDataset(lcset_name, copy(lcdataset[lcset_name]), device, **dataset_kwargs)
	r_test_loader = DataLoader(r_test_dataset, shuffle=False, batch_size=main_args.batch_size)

	s_train_dataset_da.set_scalers_from(s_train_dataset_da).calcule_precomputed(verbose=1)
	s_train_dataset.set_scalers_from(s_train_dataset_da).calcule_precomputed(verbose=1)
	r_train_dataset.set_scalers_from(s_train_dataset_da).calcule_precomputed(verbose=1)
	r_val_dataset.set_scalers_from(s_train_dataset_da).calcule_precomputed(verbose=1)
	r_test_dataset.set_scalers_from(s_train_dataset_da).calcule_precomputed(verbose=1)

	### GET MODEL
	mp_grid['mdl_kwargs']['input_dims'] = s_train_loader.dataset.get_output_dims()
	model = mp_grid['mdl_kwargs']['C'](**mp_grid)

	### pre-training
	### OPTIMIZER
	import torch.optim as optims
	from fuzzytorch.optimizers import LossOptimizer
	import math

	def pt_lr_f(epoch):
		initial_lr = 1e-10
		max_lr = 1e-3
		# return max_lr
		d_epochs = 5
		p = np.clip(epoch/d_epochs, 0, 1)
		return initial_lr+p*(max_lr-initial_lr)

	pt_opt_kwargs_f = {
		'lr':pt_lr_f,
		}
	pt_optimizer = LossOptimizer(model, optims.AdamW, pt_opt_kwargs_f, # SGD Adagrad Adadelta RMSprop Adam AdamW
		clip_grad=1.,
		)

	### MONITORS
	from fuzzytools.prints import print_bar
	from fuzzytorch.handlers import ModelTrainHandler
	from fuzzytorch.monitors import LossMonitor
	from fuzzytorch import C_
	import math

	pt_loss_monitors = LossMonitor(pt_loss, pt_optimizer, pt_metrics,
		val_epoch_counter_duration=0, # every k epochs check
		earlystop_epoch_duration=1e6,
		target_metric_crit=f'{metric_prefix}wmse',
		#save_mode=C_.SM_NO_SAVE,
		#save_mode=C_.SM_ALL,
		#save_mode=C_.SM_ONLY_ALL,
		save_mode=C_.SM_ONLY_INF_METRIC,
		#save_mode=C_.SM_ONLY_INF_LOSS,
		#save_mode=C_.SM_ONLY_SUP_METRIC,
		)

	### TRAIN
	train_mode = 'pre-training'
	extra_model_name_dict = {
			'b':f'{main_args.batch_size}',
			}
	extra_model_name_dict.update(get_dict_from_string(main_args.extra_model_name))
	pt_model_train_handler = ModelTrainHandler(model, pt_loss_monitors,
		id=main_args.mid,
		epochs_max=150, # 50 100 150 # limit this as the pre-training is very time consuming
		extra_model_name_dict=extra_model_name_dict,
		)
	complete_model_name = pt_model_train_handler.get_complete_model_name()
	pt_model_train_handler.set_complete_save_roodir(f'../save/{complete_model_name}/{train_mode}/_training/{cfilename}/{main_args.kf}@train')
	pt_model_train_handler.build_gpu(device)
	print(pt_model_train_handler)
	if main_args.only_attn_exp:
		pass
	else:
		if not bypass_autoencoder:
			pt_model_train_handler.fit_loader(s_train_loader_da, {
				#'train':s_train_loader,
				'val':r_val_loader,
				},
				train_dataset_method_call='pre_epoch_step',
				) # main fit
			del s_train_loader_da
		pass
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

	if main_args.only_attn_exp:
		pt_exp_kwargs = {
			'm':3,
			}
		save_attention_statistics(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/attn_stats/{cfilename}', **pt_exp_kwargs)
		save_attention_statistics(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/attn_stats/{cfilename}', **pt_exp_kwargs)

		save_attn_scores_animation(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/attn_scores/{cfilename}', **pt_exp_kwargs) # sanity check / slow
		#save_attn_scores_animation(pt_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/attn_scores/{cfilename}', **pt_exp_kwargs) # sanity check
		#save_attn_scores_animation(pt_model_train_handler, r_val_loader, f'../save/{complete_model_name}/{train_mode}/attn_scores/{cfilename}', **pt_exp_kwargs)
		save_attn_scores_animation(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/attn_scores/{cfilename}', **pt_exp_kwargs)
		continue # breaks the normal training

	if not bypass_autoencoder:
		pt_exp_kwargs = {
			'm':20,
			'target_is_onehot':False,
			'classifier_key':'y_last_pt',
			}
		save_reconstructions(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs) # sanity check / slow
		#save_reconstructions(pt_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs) # sanity check
		#save_reconstructions(pt_model_train_handler, r_val_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs)
		save_reconstructions(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs)

		save_model_info(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/model_info/{cfilename}', **pt_exp_kwargs) # crash when bypassing autoencoder
		save_temporal_encoding(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/temporal_encoding/{cfilename}', **pt_exp_kwargs)
	
	###################################################################################################################################################
	###################################################################################################################################################
	###################################################################################################################################################

	### fine-tuning
	model_copy = deepcopy(pt_model_train_handler.load_model())
	model_copy.init_fine_tuning()
	lcset_name = f'{main_args.kf}@train'
	lcset_copy = copy(lcdataset[lcset_name])
	for classifier_mid in range(0, main_args.classifier_mids):
		r_train_dataset_da = CustomDataset(lcset_name, lcset_copy, 'cpu',
			precomputed_copies=0,
			uses_daugm=True,
			uses_dynamic_balance=True,
			ds_mode=ds_mode,
			ds_p=ds_p,
			std_scale=std_scale,
			**dataset_kwargs,
			)
		r_train_loader_da = DataLoader(r_train_dataset_da,
			shuffle=True,
			drop_last=True,
			batch_size=32,
			num_workers=main_args.num_workers,
			pin_memory=main_args.pin_memory,
			# prefetch_factor=5,
			worker_init_fn=lambda id:np.random.seed(torch.initial_seed() // 2**32+id), # num_workers-numpy bug
			persistent_workers=main_args.num_workers>0,
			)
		# r_train_dataset_da = CustomDataset(lcset_name, lcset_copy, device,
		# 	precomputed_copies=150, # 1 50 100 150
		# 	uses_daugm=True,
		# 	uses_dynamic_balance=True,
		# 	ds_mode=ds_mode,
			# ds_p=ds_p,
			# std_scale=std_scale,
		# 	**dataset_kwargs,
		# 	)
		# r_train_loader_da = DataLoader(r_train_dataset_da,
		# 	shuffle=True,
		# drop_last=True,
		# 	batch_size=32,
		# 	)

		r_train_dataset_da.set_scalers_from(s_train_dataset_da).calcule_precomputed(verbose=1)

		import torch.optim as optims
		from fuzzytorch.optimizers import LossOptimizer

		def ft_lr_f(epoch):
			initial_lr = 1e-10
			max_lr = 1e-3
			# return max_lr
			d_epochs = 50
			p = np.clip(epoch/d_epochs, 0, 1)
			return initial_lr+p*(max_lr-initial_lr)

		ft_opt_kwargs_f = {
			'lr':ft_lr_f,
			}
		# ft_model = deepcopy(model_copy)
		# 
		# tf_model_to_optimize = ft_model.get_classifier_model(); tf_model_to_optimize.reset_parameters()
		# # tf_model_to_optimize = ft_model

		classifier = model_copy.get_classifier_model()
		classifier.reset_parameters()
		ft_optimizer = LossOptimizer(classifier, optims.AdamW, ft_opt_kwargs_f, # SGD Adagrad Adadelta RMSprop Adam AdamW
			clip_grad=1.,
			)

		### monitors
		from fuzzytools.prints import print_bar
		from fuzzytorch.handlers import ModelTrainHandler
		from fuzzytorch.monitors import LossMonitor
		from fuzzytorch import C_
		import math

		ft_loss_monitors = LossMonitor(ft_loss, ft_optimizer, ft_metrics,
			val_epoch_counter_duration=0, # every k epochs check
			earlystop_epoch_duration=250,
			target_metric_crit=f'{metric_prefix}xentropy',
			#save_mode=C_.SM_NO_SAVE,
			#save_mode=C_.SM_ALL,
			#save_mode=C_.SM_ONLY_ALL,
			save_mode=C_.SM_ONLY_INF_METRIC,
			#save_mode=C_.SM_ONLY_INF_LOSS,
			#save_mode=C_.SM_ONLY_SUP_METRIC,
			)
		### TRAIN
		train_mode = 'fine-tuning'
		extra_model_name_dict = {
				'b':f'{main_args.batch_size}',
				}
		extra_model_name_dict.update(get_dict_from_string(main_args.extra_model_name))
		mtrain_config = {
			'id':f'{main_args.mid}c{classifier_mid}',
			'epochs_max':1e6, # limit this as the pre-training is very time consuming 5 10 15 20 25 30
			'save_rootdir':f'../save/{train_mode}/_training/{cfilename}',
			'extra_model_name_dict':extra_model_name_dict,
			}
		ft_model_train_handler = ModelTrainHandler(model_copy, ft_loss_monitors, **mtrain_config)
		complete_model_name = ft_model_train_handler.get_complete_model_name()
		ft_model_train_handler.set_complete_save_roodir(f'../save/{complete_model_name}/{train_mode}/_training/{cfilename}/{main_args.kf}@train')
		ft_model_train_handler.build_gpu(device)
		print(ft_model_train_handler)
		ft_model_train_handler.fit_loader(r_train_loader_da, {
			'train':r_train_loader,
			'val':r_val_loader,
			},
			train_dataset_method_call='pre_epoch_step',
			) # main fit
		del r_train_loader_da
		ft_model_train_handler.load_model() # important, refresh to best model

		###################################################################################################################################################
		from lcclassifier.experiments.performance import save_performance

		ft_exp_kwargs = {
			'm':15,
			'target_is_onehot':False,
			'classifier_key':'y_last_ft',
			}	
		#save_performance(ft_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs) # sanity check / slow
		#save_performance(ft_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs) # sanity check
		#save_performance(ft_model_train_handler, r_val_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs)
		save_performance(ft_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs)

		save_model_info(ft_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/model_info/{cfilename}', **ft_exp_kwargs)