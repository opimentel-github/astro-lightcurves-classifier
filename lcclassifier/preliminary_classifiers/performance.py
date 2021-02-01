from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from flamingchoripan.datascience.statistics import TopRank
from flamingchoripan.datascience.metrics import get_multiclass_metrics
from .files import load_features
import numpy as np

###################################################################################################################################################

def evaluate_classifiers(lcdataset, lcset_name, classifier_dict, model_ids, load_rootdir):
	lcset = lcdataset[lcset_name]
	class_names = lcset.class_names
	results_dict = {}
	for id in model_ids:
		brf = classifier_dict[id]['brf']
		x_df, y_df = load_features(f'{load_rootdir}/{lcset_name}.ftres')
		y_target = y_df.values[...,0]
		y_pred = brf.predict(x_df.values)
		metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred, y_target, class_names, pred_is_onehot=False)
		results_dict[id] = {
			'lcset_name':lcset_name,
			'class_names':class_names,
			'metrics_cdict':metrics_cdict,
			'metrics_dict':metrics_dict,
			'cm':cm,
			'features':classifier_dict[id]['features'],
			'rank':classifier_dict[id]['rank'],
		}

	return results_dict