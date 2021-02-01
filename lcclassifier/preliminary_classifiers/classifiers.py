from __future__ import print_function
from __future__ import division
from . import C_

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fuzzytorch.utils import get_model_name, print_tdict
import fuzzytorch.models.rnn.basics as ft_rnn
from fuzzytorch.models.basics import MLP, Linear
import fuzzytorch.models.seq_utils as seq_utils
#from .GRUD_layer import GRUD_cell
from .GRUD import GRUD
#from imblearn.ensemble import BalancedRandomForestClassifier

###################################################################################################################################################

'''
def get_fitted_classifiers(lcdataset, train_lcset_name, load_rootdir,
	max_model_ids=20,
	add_real_samples=False,
	real_repeat=64,
	):
	train_lcset = lcdataset[train_lcset_name]
	class_names = train_lcset.class_names
	classifier_dict = {}
	model_ids = list(range(0, max_model_ids))
	bar = ProgressBar(len(model_ids))
	for id in model_ids:
		brf_kwargs = {
			'n_jobs':C_.N_JOBS,
			'n_estimators':100,
			#'max_depth':20,
			#'max_features':'auto',
			#'class_weight':None,
			#'criterion':'entropy',
			#'min_samples_split':2,
			#'min_samples_leaf':1,
			#'verbose':1,
			'bootstrap':True,
			'max_samples':200, # REALLY IMPORTANT PARAMETER
			#'class_weight':'balanced_subsample',
		}
		### fit
		brf = BalancedRandomForestClassifier(**brf_kwargs)
		#brf = RandomForestClassifier(**brf_kwargs)
		x_df, y_df = load_features(f'{load_rootdir}/{train_lcset_name}.ftres')
		is_real_lcset = not '.' in train_lcset_name

		if is_real_lcset:
			x_df = pd.concat([x_df]*real_repeat*2, axis=0)
			y_df = pd.concat([y_df]*real_repeat*2, axis=0)
		else:
			if add_real_samples:
				real_lcset_name = train_lcset_name.split('.')[0]
				#print(real_lcset_name)
				rx_df, ry_df = load_features(f'{load_rootdir}/{real_lcset_name}.ftres')
				x_df = pd.concat([x_df]+[rx_df]*real_repeat, axis=0)
				y_df = pd.concat([y_df]+[ry_df]*real_repeat, axis=0)
			else:
				x_df = pd.concat([x_df]*2, axis=0)
				y_df = pd.concat([y_df]*2, axis=0)

		bar(f'training id: {id} - samples: {len(y_df)} - features: {len(x_df.columns)}')
		#print(x_df.columns, x_df)
		brf.fit(x_df.values, y_df.values[...,0])

		### rank
		features = list(x_df.columns)
		rank = TopRank('features')
		rank.add_list(features, brf.feature_importances_)
		rank.calcule_rank()
		classifier_dict[id] = {
			'brf':brf,
			'features':features,
			'rank':rank,
		}
	bar.done()
	return classifier_dict, model_ids
'''


class SimpleRFClassifier():
	def __init__(self, **kwargs):
		super().__init__()

		self.pytorch_based = False

		brf_kwargs = {
			'n_jobs':C_.N_JOBS,
			'n_estimators':100,
			#'max_depth':20,
			#'max_features':'auto',
			#'class_weight':None,
			#'criterion':'entropy',
			#'min_samples_split':2,
			#'min_samples_leaf':1,
			#'verbose':1,
			'bootstrap':True,
			'max_samples':200, # REALLY IMPORTANT PARAMETER
			#'class_weight':'balanced_subsample',
		}
		self.classifier = BalancedRandomForestClassifier(**brf_kwargs)

	def get_name(self):
		return get_model_name({
			'mdl':f'SimpleRF',
		})

	def fit(self, x_df, y_df, **kwargs):
		self.classifier.fit(x_df.values, y_df.values[...,0])

		### rank
		features = list(x_df.columns)
		rank = TopRank('features')
		rank.add_list(features, brf.feature_importances_)
		rank.calcule_rank()
		classifier_dict[id] = {
			'brf':brf,
			'features':features,
			'rank':rank,
		}

	def predict(self):
		pass

###################################################################################################################################################

class SimpleRNNClassifier(nn.Module):
	def __init__(self, band_names, output_dims,
		rnn_embd_dims=30,
		**kwargs):
		super().__init__()

		self.pytorch_based = True

		self.band_names = band_names
		self.output_dims = output_dims
		self.rnn_embd_dims = rnn_embd_dims
		self.dt = 2

		### RNN STACK
		self.encoder = nn.ModuleDict({b:ft_rnn.MLGRU(2, self.rnn_embd_dims, []) for b in self.band_names})
		#self.encoder = nn.ModuleDict({b:GRUD(1, self.rnn_embd_dims, self.rnn_embd_dims, [0]) for b in self.band_names})
		print('encoder', self.encoder)

		mlp_kwargs = {
			'activation':'relu',
			'last_activation':'linear',
			'in_dropout':0.333,
			'dropout':0.333,
		}
		self.mlp = MLP(self.rnn_embd_dims*len(self.band_names), self.output_dims, [self.rnn_embd_dims], **mlp_kwargs)
		print('mlp', self.mlp)

	def get_name(self):
		return get_model_name({
			'mdl':f'SimpleRNN',
		})

	def forward(self, tdict, **kwargs):
		#print(tdict)
		last_z_dic = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = tdict['input'][f'onehot.{b}'][...,0]
			p_obs = tdict['input'][f'binned_obs.{b}']
			p_missing_mask = tdict['input'][f'missing_mask.{b}']
			p_x = torch.cat([p_obs, p_missing_mask.float()], dim=-1)
			p_z,_ = self.encoder[b](p_x, p_onehot)

			### get last element
			last_z_dic[b] = seq_utils.seq_last_element(p_z, p_onehot) # get last value of sequence according to onehot

			if 0:
				X = p_obs
				Mask = p_missing_mask.float()
				Delta = torch.full_like(Mask, self.dt)
				grud_input = torch.cat([X[:,None,...], X[:,None,...], Mask[:,None,...], Delta[:,None,...]], dim=1)
				print(grud_input.shape)
				p_z, hidden_tensor = self.encoder[b](grud_input)
				print(p_z.shape, hidden_tensor.shape)
				last_z_dic[b] = hidden_tensor


		last_z = torch.cat([last_z_dic[b] for b in self.band_names], dim=-1)
		tdict['model'] = {} if not 'model' in tdict.keys() else tdict['model']
		tdict['model']['y.last'] = self.mlp(last_z)
		#print_tdict(tdict)
		return tdict