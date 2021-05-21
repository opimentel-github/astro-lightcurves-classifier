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
		#prefetch_factor=1,
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
			#prefetch_factor=prefetch_factor,
			)
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
		return default_collate(batch)