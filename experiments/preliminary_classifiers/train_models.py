#!/usr/bin/env python3
import sys
sys.path.append('../../') # or just install the module
sys.path.append('../../../fuzzy-torch') # or just install the module
sys.path.append('../../../flaming-choripan') # or just install the module
sys.path.append('../../../astro-lightcurves-handler') # or just install the module
sys.path.append('../../../sne-lightcurves-synthetic') # or just install the module

if __name__== '__main__':
	### parser arguments
	import argparse
	from flamingchoripan.prints import print_big_bar

	parser = argparse.ArgumentParser('usage description')
	parser.add_argument('-method',  type=str, default='.', help='method')
	parser.add_argument('-gpu',  type=int, default=-1, help='gpu_index')
	parser.add_argument('-batch_size',  type=int, default=1024, help='batch_size')
	parser.add_argument('-epochs_max',  type=int, default=1e4, help='epochs_max')
	parser.add_argument('-save_rootdir',  type=str, default='', help='save_rootdir')
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

	methods = main_args.method
	methods = ['linear-fstw', 'bspline-fstw', 'spm-mle-fstw', 'spm-mcmc-fstw', 'spm-mle-estw', 'spm-mcmc-estw'] if methods=='.' else methods
	methods = [methods] if isinstance(methods, str) else methods

	for method in methods:
		filedir = f'../../../surveys-save/alerceZTFv7.1/survey=alerceZTFv7.1°bands=gr°mode=onlySNe°method={method}.splcds'
		filedict = get_dict_from_filedir(filedir)
		root_folder = filedict['*rootdir*']
		cfilename = filedict['*cfilename*']
		survey = filedict['survey']
		lcdataset = load_pickle(filedir)
		print(lcdataset)

		###################################################################################################################################################
		from lcclassifier.preliminary_classifiers.classifiers import SimpleRNNClassifier, SimpleRFClassifier

		model_collections = [
			{'C':SimpleRNNClassifier},
			{'C':SimpleRFClassifier},
			]
		print(model_collections)

		###################################################################################################################################################
		### LOSS & METRICS
		from lcclassifier.losses import LCMSEReconstruction, LCXEntropy, LCCompleteLoss
		from lcclassifier.metrics import LCXEntropyMetric, LCAccuracy

		loss_kwargs = {
			'model_output_is_with_softmax':False,
			'target_is_onehot':False,
			'uses_poblation_weights':True,
		}
		loss = LCXEntropy('xentropy', **loss_kwargs)
		metrics = [
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
		from lcclassifier.preliminary_classifiers.datasets import CustomDataset
		from torch.utils.data import DataLoader

		for mp_grid in model_collections:
			### DATASETS
			#train_dataset = CustomDataset(lcdataset, f'train.{method}')
			train_dataset = CustomDataset(lcdataset, f'train')
			val_dataset = CustomDataset(lcdataset, 'val')
			train_dataset.transfer_to(val_dataset) # transfer information to val/test
			print('train_dataset:', train_dataset)
			print('val_dataset:', val_dataset)
			
			train_dataset.precompute_samples(1)
			val_dataset.precompute_samples(1)
			
			### DATALOADERS
			loader_kwargs = {
				'batch_size':main_args.batch_size,
				#'num_workers':2, # bug?
			}
			train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
			val_loader = DataLoader(val_dataset, **loader_kwargs)

			### IDS
			model_ids = list(range(main_args.iid, main_args.fid+1))
			for ki,model_id in enumerate(model_ids): # IDS
				### GET MODEL
				model = mp_grid['C'](train_dataset.band_names, len(train_dataset.class_names))
				
				### OPTIMIZER
				import torch.optim as optims
				from fuzzytorch.optimizers import LossOptimizer

				optimizer_kwargs = {
					'opt_kwargs':{
						'lr':1e-3,
					},
					'decay_kwargs':{
						'lr':0.9,
					}
				}
				optimizer = LossOptimizer(model, optims.Adam, **optimizer_kwargs)

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
					#'save_mode':C_.SM_ONLY_INF_METRIC,
					'save_mode':C_.SM_ONLY_INF_LOSS,
					#'save_mode':C_.SM_ONLY_SUP_METRIC,
				}
				loss_monitors = LossMonitor(loss, optimizer, metrics, **monitor_config)
				
				### TRAIN
				mtrain_config = {
					'id':model_id,
					'epochs_max':1e5,
					'save_rootdir':f'../preliminary_save/training',
				}
				model_train_handler = ModelTrainHandler(model, loss_monitors, **mtrain_config)
				model_train_handler.build_gpu(0 if main_args.gpu>=0 else None)
				if ki==0:
					print(model_train_handler)
				model_train_handler.fit_loader(train_loader, val_loader)

				###################################################################################################################################################
				import fuzzytorch
				import fuzzytorch.plots
				import fuzzytorch.plots.training as ffplots

				### training plots
				plot_kwargs = {
					'save_rootdir':f'../preliminary_save/train_plots',
				}
				ffplots.plot_loss(model_train_handler, **plot_kwargs)
				#ffplots.plot_evaluation_loss(train_handler, **plot_kwargs)
				#ffplots.plot_evaluation_metrics(train_handler, **plot_kwargs)

				###################################################################################################################################################
				import lcclassifier.experiments.images as exp_img

				### perform the experiments
				exp_kwargs = {
					'save_rootdir':f'../preliminary_save/experiments',
					'm':4,
					'send_email':0,
				}
				#exp_img.reconstructions(model_train_handler, s_train_loader, **exp_kwargs)
				exp_img.reconstructions(model_train_handler, s_val_loader, **exp_kwargs)

				###################################################################################################################################################
				import lcclassifier.experiments.performance as exp_perf

				### perform the experiments
				exp_kwargs = {
					'save_rootdir':f'../preliminary_save/experiments',
					'target_is_onehot':False,
					'send_email':0,
				}
				#exp_perf.reconstruction_along_days(model_train_handler, s_train_loader, **exp_kwargs) # over real
				exp_perf.reconstruction_along_days(model_train_handler, s_val_loader, **exp_kwargs) # over real

				###################################################################################################################################################
				import lcclassifier.experiments.performance as exp_perf

				### perform the experiments
				exp_kwargs = {
					'save_rootdir':f'../preliminary_save/experiments',
					'target_is_onehot':False,
					'send_email':0,
				}
				#exp_perf.metrics_along_days(model_train_handler, s_val_loader, **exp_kwargs) # over real
				exp_perf.metrics_along_days(model_train_handler, s_val_loader, **exp_kwargs) # over real