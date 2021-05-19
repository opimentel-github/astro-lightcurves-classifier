from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import fuzzytools.files as fcfiles
import fuzzytools.strings as strings
from fuzzytools.dataframes import DFBuilder
from fuzzytools.datascience.statistics import XError
from . import utils as utils
import pandas as pd
from nested_dict import nested_dict

###################################################################################################################################################

def get_times_df(rootdir, cfilename, kf, lcset_name, model_names,
	arch_modes=['Parallel', 'Serial'],
	train_modes=['pre-training', 'fine-tuning'],
	n=1e3,
	#override_model_name=True,
	label_keys=[],
	):
	info_df = DFBuilder()
	for uses_avg in [False, True]:
		for arch_mode in arch_modes:
			for kmn,model_name in enumerate(model_names):
				d = {}
				for train_mode in train_modes:
					_model_name = model_name.replace('Serial', arch_mode).replace('Parallel', arch_mode)
					load_roodir = f'{rootdir}/{_model_name}/{train_mode}/exp=performance/{cfilename}/{kf}@{lcset_name}'
					print(load_roodir)
					files, files_ids = fcfiles.gather_files_by_id(load_roodir, fext='d')
					print(f'ids={files_ids}(n={len(files_ids)}#) - model={model_name}')
					if len(files)==0:
						continue

					survey = files[0]()['survey']
					band_names = files[0]()['band_names']
					class_names = files[0]()['class_names']
					days = files[0]()['days']
					mn_dict = strings.get_dict_from_string(model_name)
					rsc = mn_dict['rsc']
					mdl = mn_dict['mdl']
					is_parallel = 'Parallel' in mdl

					#print(files[0]().keys())
					p = [f()['parameters'] for f in files]
					t1 = [f()['xentropy']['time_per_iteration'] for f in files]
					t2 = [f()['xentropy']['time_per_epoch'] for f in files]
					t3 = [f()['xentropy']['total_time'] for f in files]
					d[train_mode] = XError(t3)

				print(d)
				_d_key = strings.get_string_from_dict({k:mn_dict[k] for k in mn_dict.keys() if k in label_keys}, key_key_separator=' - ')
				d_key = f'{mdl} ({_d_key})'
				#d_key = f'{mdl} [{arch_mode}]' if override_model_name else f'{label} [{arch_mode}]'
				info_df.append(d_key, d)

	return info_df()