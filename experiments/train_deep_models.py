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

parser = argparse.ArgumentParser('usage description')
parser.add_argument('--method',  type=str, default='spm-mcmc-estw')
parser.add_argument('--gpu',  type=int, default=-1)
parser.add_argument('--mc',  type=str, default='parallel_rnn_models')
parser.add_argument('--batch_size',  type=int, default=128) # *** 16 32 64 128
parser.add_argument('--batch_size_c',  type=int, default=128) # *** 16 32 64 128
parser.add_argument('--epochs_max',  type=int, default=1e4)
parser.add_argument('--save_rootdir',  type=str, default='../save')
parser.add_argument('--mid',  type=str, default='1000')
parser.add_argument('--kf',  type=str, default='0')
parser.add_argument('--bypass',  type=int, default=0) # 0 1
parser.add_argument('--train_ae',  type=int, default=1) # 0 1
parser.add_argument('--only_attn_exp',  type=int, default=0) # 0 1
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

from copy import copy
for k in lcdataset.lcsets.keys():
    for lcobj_name in lcdataset[k].get_lcobj_names():
        for b in lcdataset[k].band_names:
            lcdataset[k][lcobj_name].get_b(b).astype(np.float32) # fixme

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

for mp_grid in model_collections.mps: # MODEL CONFIGS
	from lcclassifier.datasets import CustomDataset
	from torch.utils.data import DataLoader
	from fuzzytools.files import get_filedirs, copy_filedir
	import torch
	from copy import copy, deepcopy

	### DATASETS
	dataset_kwargs = mp_grid['dataset_kwargs']
	device = 'cuda:0'
	#device = 'cpu'
	repeats = 5
	synth_repeats = 12

	### WITH DATA
	lcset_name = f'{main_args.kf}@train.{main_args.method}'
	s_train_dataset_da = CustomDataset(lcset_name, copy(lcdataset[lcset_name]), device,
		balanced_repeats=repeats,
		precomputed_copies=10, # 1 5 10
		uses_daugm=True,
		uses_dynamic_balance=True,
		ds_mode={'random':.75, 'left':.0, 'none':.25,},
		**dataset_kwargs,
		)
	s_train_loader_da = DataLoader(s_train_dataset_da,
		shuffle=True,
		batch_size=main_args.batch_size,
		)

	USES_CPU = 0
	if USES_CPU: # CPU
		lcset_name = f'{main_args.kf}@train'
		r_train_dataset_da = CustomDataset(lcset_name, copy(lcdataset[lcset_name]), 'cpu',
			balanced_repeats=repeats*synth_repeats,
			uses_precomputed_copies=False,
			uses_daugm=True,
			uses_dynamic_balance=True,
			ds_mode={'random':.9, 'left':.1, 'none':.0,}, # avoid none as it's not random
			**dataset_kwargs,
			)
		r_train_loader_da = DataLoader(r_train_dataset_da,
			shuffle=True,
			batch_size=main_args.batch_size_c,#main_args.batch_size_c, batch_size_c
			num_workers=4, # 0 2 4 6 # memory leak with too much workers?
			pin_memory=True, # False True
			persistent_workers=True,
			worker_init_fn=lambda id:np.random.seed(torch.initial_seed() // 2**32+id), # num_workers-numpy bug
			)
	else: # GPU
		lcset_name = f'{main_args.kf}@train'
		r_train_dataset_da = CustomDataset(lcset_name, copy(lcdataset[lcset_name]), device,
			balanced_repeats=repeats*synth_repeats,
			precomputed_copies=50, # 1 50 100
			uses_daugm=True,
			uses_dynamic_balance=True,
			ds_mode={'random':.9, 'left':.1, 'none':.0,}, # avoid none as it's not random
			**dataset_kwargs,
			)
		r_train_loader_da = DataLoader(r_train_dataset_da,
			shuffle=True,
			batch_size=main_args.batch_size_c,#main_args.batch_size_c, batch_size_c
			)

	####
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
	r_train_dataset_da.set_scalers_from(s_train_dataset_da).calcule_precomputed(verbose=1)
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

	pt_opt_kwargs_f = {
		'lr':lambda epoch:1e-3, # ***
		}
	pt_optimizer_kwargs = {
		'clip_grad':1.,
		}
	pt_optimizer = LossOptimizer(model, optims.Adam, pt_opt_kwargs_f, **pt_optimizer_kwargs) # SGD Adagrad Adadelta RMSprop Adam AdamW

	### MONITORS
	from fuzzytools.prints import print_bar
	from fuzzytorch.handlers import ModelTrainHandler
	from fuzzytorch.monitors import LossMonitor
	from fuzzytorch import C_
	import math

	monitor_config = {
		'val_epoch_counter_duration':0, # every k epochs check
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
		'id':main_args.mid,
		'epochs_max':1500, # limit this as the pre-training is very time consuming
		'extra_model_name_dict':{
			#'mode':train_mode,
			#'ef-be':f'1e{math.log10(s_train_loader.dataset.effective_beta_eps)}',
			#'ef-be':s_train_loader.dataset.effective_beta_eps,
			#'b':f'{main_args.batch_size}.{main_args.batch_size_c}',
			'bypass':main_args.bypass,
			},
		}
	pt_model_train_handler = ModelTrainHandler(model, pt_loss_monitors, **mtrain_config)
	complete_model_name = pt_model_train_handler.get_complete_model_name()
	pt_model_train_handler.set_complete_save_roodir(f'../save/{complete_model_name}/{train_mode}/_training/{cfilename}/{main_args.kf}@train')
	pt_model_train_handler.build_gpu(device if main_args.gpu>=0 else None)
	print(pt_model_train_handler)
	if main_args.only_attn_exp:
		pass
	else:
		if main_args.train_ae:
			pt_model_train_handler.fit_loader(s_train_loader_da, {
				#'train':s_train_loader,
				'val':r_val_loader,
				}) # main fit
			pass
		else:
			assert 0, 'fixme'
			filedirs = get_filedirs(pt_model_train_handler.complete_save_roodir, fext='tfes')
			first_model_id = model_ids[0]
			src_filedir = [filedir for filedir in filedirs if f'id={first_model_id}' in filedir][0]
			dst_filedir = src_filedir.replace(f'id={first_model_id}',f'id={model_id}')
			#print(filedirs, src_filedir, dst_filedir)
			copy_filedir(src_filedir, dst_filedir) # time saving
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

	if main_args.train_ae:
		pt_exp_kwargs = {
			'm':20,
			'target_is_onehot':False,
			'classifier_key':'y_last_pt',
			}
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
		max_lr = 1e-3
		d_epochs = 25
		p = np.clip(epoch/d_epochs, 0, 1)
		return initial_lr+p*(max_lr-initial_lr)

	ft_opt_kwargs_f = {
		#'lr':lambda epoch:1e-3, # ***
		'lr':ft_lr_f, # ***
		#'weight_decay':lambda epoch:1e-5,
		}
	ft_optimizer_kwargs = {
		'clip_grad':1.,
		}
	classifier = model.get_classifier_model()
	classifier.init_parameters() # to ensure random inits
	ft_optimizer = LossOptimizer(classifier, optims.Adam, ft_opt_kwargs_f, **ft_optimizer_kwargs) # SGD Adagrad Adadelta RMSprop Adam AdamW

	### MONITORS
	from fuzzytools.prints import print_bar
	from fuzzytorch.handlers import ModelTrainHandler
	from fuzzytorch.monitors import LossMonitor
	from fuzzytorch import C_
	import math

	monitor_config = {
		'val_epoch_counter_duration':0, # every k epochs check
		'earlystop_epoch_duration':200,
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
		'id':main_args.mid,
		'epochs_max':1e6, # limit this as the pre-training is very time consuming 5 10 15 20 25 30
		'save_rootdir':f'../save/{train_mode}/_training/{cfilename}',
		'extra_model_name_dict':{
			#'mode':train_mode,
			#'ef-be':f'1e{math.log10(s_train_loader.dataset.effective_beta_eps)}',
			#'ef-be':s_train_loader.dataset.effective_beta_eps,
			#'b':f'{main_args.batch_size}.{main_args.batch_size_c}',
			'bypass':main_args.bypass,
			},
		}
	ft_model_train_handler = ModelTrainHandler(model, ft_loss_monitors, **mtrain_config)
	complete_model_name = ft_model_train_handler.get_complete_model_name()
	ft_model_train_handler.set_complete_save_roodir(f'../save/{complete_model_name}/{train_mode}/_training/{cfilename}/{main_args.kf}@train')
	ft_model_train_handler.build_gpu(device if main_args.gpu>=0 else None)
	print(ft_model_train_handler)
	ft_model_train_handler.fit_loader(r_train_loader_da, {
		'train':r_train_loader,
		'val':r_val_loader,
		}) # main fit
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