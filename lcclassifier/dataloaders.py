from __future__ import print_function
from __future__ import division
from . import C_

import random
import torch
import torch.tensor as Tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from fuzzytorch.utils import print_tdict
from fuzzytorch.models.seq_utils import seq_clean, get_seq_onehot_mask

###################################################################################################################################################

class CustomDataLoader(DataLoader):
	def __init__(self, dataset,
		random_subcrops:int=2,
		min_length:int=5,
		batch_size=1,
		shuffle=False,
		sampler=None,
		batch_sampler=None,
		num_workers=0,
		collate_fn=None,
		pin_memory=False,
		drop_last=False,
		timeout=0,
		worker_init_fn=None,
		multiprocessing_context=None,
		prefetch_factor=1,
		):
		super().__init__(dataset,
			batch_size=batch_size,
			shuffle=shuffle,
			sampler=sampler,
			batch_sampler=batch_sampler,
			num_workers=num_workers,
			collate_fn=collate_fn,
			pin_memory=pin_memory,
			drop_last=drop_last,
			timeout=timeout,
			worker_init_fn=worker_init_fn,
			multiprocessing_context=multiprocessing_context,
			prefetch_factor=prefetch_factor,
			)
		assert random_subcrops>=0
		assert min_length>0

		self.random_subcrops = random_subcrops
		self.min_length = min_length
		self.collate_fn = self.custom_collate_fn
		self.reset()

	def reset(self):
		self.eval()

	def train(self):
		self.training = True
		self.dataset.train()

	def eval(self):
		self.training = False
		self.dataset.eval()

	def custom_collate_fn(self, batch):
		if self.random_subcrops>0 and self.training:
			# add subcrops - it's slow
			new_batch_dicts = []
			for tdict in batch:
				length = tdict['input']['onehot'].detach().sum().item()
				assert length>=0
				new_lengths = [length if k==0 else random.randint(self.min_length, max(self.min_length, length-1)) for k in range(0, self.random_subcrops+1)]
				#print(length, new_lengths)
				for l in new_lengths:
					new_tdict = {'input':{}, 'target':{}}
					for k in list(new_tdict.keys()):
						for k2 in tdict[k].keys():
							#print(k,k2)
							x = tdict[k][k2]
							shape = list(x.shape)
							if len(shape)==2:
								new_x = torch.clone(x)
								new_x[l:,...] = 0
								new_tdict[k][k2] = new_x
							else:
								new_tdict[k][k2] = x.clone()

					#print_tdict(new_tdict)
					new_batch_dicts.append(new_tdict)

			return default_collate(new_batch_dicts)
		else:
			# only complete curves
			return default_collate(batch)