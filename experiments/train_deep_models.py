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
parser.add_argument('--batch_size',  type=int, default=32)
parser.add_argument('--save_rootdir',  type=str, default='../save')
parser.add_argument('--mid',  type=str, default='0')
parser.add_argument('--kf',  type=str, default='0')
parser.add_argument('--bypass_synth',  type=int, default=0) # 0 1
parser.add_argument('--bypass_autoencoder',  type=int, default=0) # 0 1
parser.add_argument('--invert_mpg',  type=int, default=0) # 0 1
parser.add_argument('--only_perform_exps',  type=int, default=0) # 0 1
parser.add_argument('--extra_model_name',  type=str, default='')
parser.add_argument('--classifier_mids',  type=int, default=1)
parser.add_argument('--num_workers',  type=int, default=12)
parser.add_argument('--pin_memory',  type=int, default=1) # 0 1
parser.add_argument('--pt_balanced_metrics',  type=int, default=1)
parser.add_argument('--ft_balanced_metrics',  type=int, default=1)
parser.add_argument('--precompute_only',  type=int, default=0)
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
lcdataset.only_keep_kf(main_args.kf) # saves ram
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
import lcclassifier.losses as losses
import lcclassifier.metrics as metrics
from fuzzytorch.metrics import LossWrapper

loss_kwargs = {
	'class_names':lcdataset[lcdataset.get_lcset_names()[0]].class_names,
	'band_names':lcdataset[lcdataset.get_lcset_names()[0]].band_names,
	'target_is_onehot':False,
	}
	
### pt
weight_key = f'target/balanced_w' if main_args.pt_balanced_metrics else None
pt_loss = losses.LCCompleteLoss('wmse+binxentropy', None, **loss_kwargs)
pt_metrics = [
	LossWrapper(losses.LCCompleteLoss(('b-' if main_args.pt_balanced_metrics else '')+'wmse+binxentropy', weight_key, **loss_kwargs)),
	LossWrapper(losses.LCBinXEntropy(('b-' if main_args.pt_balanced_metrics else '')+'binxentropy', weight_key, **loss_kwargs)),
	metrics.LCAccuracy(('b-' if main_args.pt_balanced_metrics else '')+'accuracy', weight_key, **loss_kwargs),
	]

### ft
weight_key = f'target/balanced_w' if main_args.ft_balanced_metrics else None
ft_loss = losses.LCBinXEntropy('binxentropy', None, **loss_kwargs)
ft_metrics = [
	LossWrapper(losses.LCBinXEntropy(('b-' if main_args.ft_balanced_metrics else '')+'binxentropy', weight_key, **loss_kwargs)),
	metrics.LCAccuracy(('b-' if main_args.ft_balanced_metrics else '')+'accuracy', weight_key, **loss_kwargs),
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
	s_train_dataset_da_kwargs = {
		'precomputed_copies':10,
		'uses_daugm':True,
		'uses_dynamic_balance':True,
		# 'ds_mode':{'random':1.0, 'left':0.0, 'none':0.0}, # avoid none
		'ds_mode':{'random':.8, 'left':.2, 'none':.0},
		'ds_p':10/100,
		'std_scale':1/2,
		'k_n':0.5,
		}
	s_train_dataset_da_kwargs.update(dataset_kwargs)

	lcset_name = f'{main_args.kf}@train.{main_args.method}' if not main_args.bypass_synth else f'{main_args.kf}@train'
	if 1:
	# if main_args.precompute_only:
		s_train_dataset_da = CustomDataset(lcset_name, lcdataset[lcset_name],
			precomputed_mode='disk', # disk online device
			device='cpu',
			**s_train_dataset_da_kwargs)
		s_train_loader_da = DataLoader(s_train_dataset_da,
			shuffle=True,
			drop_last=True,
			batch_size=main_args.batch_size,
			num_workers=main_args.num_workers,
			pin_memory=main_args.pin_memory,
			worker_init_fn=lambda id:np.random.seed(torch.initial_seed() // 2**32+id), # num_workers-numpy bug
			persistent_workers=main_args.num_workers>0,
			)
	else:
		s_train_dataset_da = CustomDataset(lcset_name, lcdataset[lcset_name],
			precomputed_mode='device', # disk online device
			device=device,
			**s_train_dataset_da_kwargs)
		s_train_loader_da = DataLoader(s_train_dataset_da,
			shuffle=True,
			drop_last=True,
			batch_size=main_args.batch_size,
			)

	lcset_name = f'{main_args.kf}@train.{main_args.method}'
	s_train_dataset = CustomDataset(lcset_name, lcdataset[lcset_name],
		precomputed_mode='online', # disk online device
		device='cpu',
		k_n=s_train_dataset_da_kwargs['k_n'],
		**dataset_kwargs)
	s_train_loader = DataLoader(s_train_dataset, shuffle=False, batch_size=main_args.batch_size)

	lcset_name = f'{main_args.kf}@train'
	r_train_dataset = CustomDataset(lcset_name, lcdataset[lcset_name],
		precomputed_mode='device',
		device=device,
		**dataset_kwargs)
	r_train_loader = DataLoader(r_train_dataset, shuffle=False, batch_size=main_args.batch_size)

	lcset_name = f'{main_args.kf}@val'
	r_val_dataset = CustomDataset(lcset_name, lcdataset[lcset_name],
		precomputed_mode='device',
		device=device,
		**dataset_kwargs)
	r_val_loader = DataLoader(r_val_dataset, shuffle=False, batch_size=main_args.batch_size)

	lcset_name = f'{main_args.kf}@test'
	r_test_dataset = CustomDataset(lcset_name, lcdataset[lcset_name],
		precomputed_mode='device',
		device=device,
		**dataset_kwargs)
	r_test_loader = DataLoader(r_test_dataset, shuffle=False, batch_size=main_args.batch_size)

	### compute datasets
	s_train_dataset_da.set_scalers_from(s_train_dataset_da).calcule_precomputed(verbose=1, read_from_disk=not main_args.precompute_only); print(s_train_dataset_da)
	if main_args.precompute_only:
		assert 0 # exit
	s_train_dataset.set_scalers_from(s_train_dataset_da).calcule_precomputed(verbose=1); print(s_train_dataset)
	r_train_dataset.set_scalers_from(s_train_dataset_da).calcule_precomputed(verbose=1); print(r_train_dataset)
	r_val_dataset.set_scalers_from(s_train_dataset_da).calcule_precomputed(verbose=1); print(r_val_dataset)
	r_test_dataset.set_scalers_from(s_train_dataset_da).calcule_precomputed(verbose=1); print(r_test_dataset)

	### GET MODEL
	mp_grid['mdl_kwargs']['input_dims'] = s_train_loader.dataset.get_output_dims()
	model = mp_grid['mdl_kwargs']['C'](**mp_grid) # model creation

	### pre-training
	### OPTIMIZER
	import torch.optim as optims
	from fuzzytorch.optimizers import LossOptimizer
	import math

	def pt_lr_f(epoch):
		min_lr, max_lr = 1e-10, 1e-3
		d_epochs = 10
		exp_decay_k = 0
		p = np.clip(epoch/d_epochs, 0, 1) # 0 > 1
		lr = (1-p)*min_lr+p*max_lr
		lr = math.exp(-np.clip(epoch-d_epochs, 0, None)*exp_decay_k)*lr
		return lr

	pt_opt_kwargs_f = {
		'lr':pt_lr_f,
		}
	pt_optimizer = LossOptimizer(model, optims.Adam, pt_opt_kwargs_f, # SGD Adagrad Adadelta RMSprop Adam AdamW
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
		target_metric_crit=('b-' if main_args.pt_balanced_metrics else '')+'wmse+binxentropy',
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
		epochs_max=100, # 50 100 150 200 # limit this as the pre-training is very time consuming
		extra_model_name_dict=extra_model_name_dict,
		)
	complete_model_name = pt_model_train_handler.get_complete_model_name()
	pt_model_train_handler.set_complete_save_roodir(f'../save/{complete_model_name}/{train_mode}/_training/{cfilename}/{main_args.kf}@train')
	pt_model_train_handler.build_gpu(device)
	print(pt_model_train_handler)
	if main_args.bypass_autoencoder or main_args.only_perform_exps:
		pass
	else:
		pt_model_train_handler.fit_loader(s_train_loader_da, {
			#'train':s_train_loader,
			'val':r_val_loader,
			},
			train_dataset_method_call='pre_epoch_step',
			) # main fit
	del s_train_loader_da
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
	from lcclassifier.experiments.dim_reductions import save_dim_reductions
	from lcclassifier.experiments.model_info import save_model_info
	from lcclassifier.experiments.temporal_encoding import save_temporal_encoding
	from lcclassifier.experiments.performance import save_performance
	from lcclassifier.experiments.attnscores import save_attnscores_animation
	from lcclassifier.experiments.attnstats import save_attnstats

	### modulation and attn
	pt_exp_kwargs = {
		'm':3,
		}
	# save_temporal_encoding(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/temporal_encoding/{cfilename}', **pt_exp_kwargs)
	# save_attnscores_animation(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/attn_scores/{cfilename}', **pt_exp_kwargs) # sanity check / slow
	# save_attnscores_animation(pt_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/attn_scores/{cfilename}', **pt_exp_kwargs) # sanity check
	# save_attnscores_animation(pt_model_train_handler, r_val_loader, f'../save/{complete_model_name}/{train_mode}/attn_scores/{cfilename}', **pt_exp_kwargs)
	# save_attnscores_animation(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/attn_scores/{cfilename}', **pt_exp_kwargs)
	# save_attnstats(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/attn_stats/{cfilename}', **pt_exp_kwargs)
	# save_attnstats(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/attn_stats/{cfilename}', **pt_exp_kwargs)

	### experiments
	pt_exp_kwargs = {
		'm':20,
		'target_is_onehot':False,
		}
	save_dim_reductions(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/dim_reductions/{cfilename}', **pt_exp_kwargs)
	save_reconstructions(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs) # sanity check / slow
	# save_reconstructions(pt_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs) # sanity check
	# save_reconstructions(pt_model_train_handler, r_val_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs)
	save_reconstructions(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs)
	save_model_info(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/model_info/{cfilename}', **pt_exp_kwargs) # crash when bypassing autoencoder
	
	###################################################################################################################################################
	###################################################################################################################################################
	###################################################################################################################################################

	### fine-tuning
	for classifier_mid in range(0, main_args.classifier_mids):
		pt_model_cache = deepcopy(pt_model_train_handler.load_model()) # copy
		# pt_model_cache.init_finetuning() # optional

		lcset_name = f'{main_args.kf}@train'
		r_train_dataset_da = CustomDataset(lcset_name, lcdataset[lcset_name],
			precomputed_mode='online',
			device='cpu',
			uses_daugm=True,
			uses_dynamic_balance=True,
			ds_mode={'random':.8, 'left':.2, 'none':.0},
			ds_p=10/100,
			std_scale=1/2,
			**dataset_kwargs,
			)
		r_train_loader_da = DataLoader(r_train_dataset_da,
			shuffle=True,
			drop_last=True,
			batch_size=16,
			num_workers=main_args.num_workers,
			pin_memory=main_args.pin_memory,
			worker_init_fn=lambda id:np.random.seed(torch.initial_seed() // 2**32+id), # num_workers-numpy bug
			persistent_workers=main_args.num_workers>0,
			)

		r_train_dataset_da.set_scalers_from(s_train_dataset_da).calcule_precomputed(verbose=1)

		import torch.optim as optims
		from fuzzytorch.optimizers import LossOptimizer

		def ft_lr_f(epoch):
			# min_lr, max_lr = 1e-10, .1e-1
			# d_epochs = 10
			# exp_decay_k = 0.01
			# p = np.clip(epoch/d_epochs, 0, 1) # 0 > 1
			# lr = (1-p)*min_lr+p*max_lr
			# lr = math.exp(-np.clip(epoch-d_epochs, 0, None)*exp_decay_k)*lr
			# return lr

			# min_lr, max_lr = 1e-10, .5e-1
			# half_period = 10
			# # offset = half_period
			# offset = 0
			# # _epoch = epoch%half_period+offset
			# _epoch = epoch+offset
			# p = (math.cos(2*math.pi*_epoch/(half_period*2))+1)/2 # 1 > 0
			# lr = (p)*max_lr+(1-p)*min_lr
			# return lr

			return 1e-3

		ft_opt_kwargs_f = {
			'lr':ft_lr_f,
			'momentum':lambda epoch:0.9,
			}
		ft_optimizer = LossOptimizer(pt_model_cache.get_finetuning_parameters(), optims.SGD, ft_opt_kwargs_f, # SGD Adagrad Adadelta RMSprop Adam AdamW
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
			earlystop_epoch_duration=150,
			target_metric_crit=('b-' if main_args.ft_balanced_metrics else '')+'binxentropy',
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
			'epochs_max':300, # limit this as the pre-training is very time consuming 5 10 15 20 25 30
			'save_rootdir':f'../save/{train_mode}/_training/{cfilename}',
			'extra_model_name_dict':extra_model_name_dict,
			}
		ft_model_train_handler = ModelTrainHandler(pt_model_cache, ft_loss_monitors, **mtrain_config)
		complete_model_name = ft_model_train_handler.get_complete_model_name()
		ft_model_train_handler.set_complete_save_roodir(f'../save/{complete_model_name}/{train_mode}/_training/{cfilename}/{main_args.kf}@train')
		ft_model_train_handler.build_gpu(device)
		print(ft_model_train_handler)
		if main_args.only_perform_exps:
			pass
		else:
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
			}	
		#save_performance(ft_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs) # sanity check / slow
		#save_performance(ft_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs) # sanity check
		save_performance(ft_model_train_handler, r_val_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs)
		save_performance(ft_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs)

		save_model_info(ft_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/model_info/{cfilename}', **ft_exp_kwargs)