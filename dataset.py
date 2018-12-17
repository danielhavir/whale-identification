import os
import multiprocessing as mp
from threading import Thread
from itertools import chain
import random

import torch
import torchvision
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

def cv_loader(path):
	""" Loads 1000 images in 7.41s """
	img = cv2.imread(path)[:,:,::-1]
	return img

def fast_collate(batch):
	""" Collate function for DataLoader replace slow transforms.ToTensor """
	images = [img[0] for img in batch]
	targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
	w = images[0].size[0]
	h = images[0].size[1]
	tensor = torch.zeros( (len(images), 3, h, w), dtype=torch.uint8 )
	for i, img in enumerate(images):
		nump_array = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
		tensor[i] += torch.from_numpy(nump_array)
	
	return tensor, targets

def pair_collate(batch):
	""" fast_collate for pairs """
	images1 = [img[0] for img in batch]
	images2 = [img[1] for img in batch]
	targets = torch.tensor([target[2] for target in batch], dtype=torch.int64)
	w = images1[0].size[0]
	h = images1[0].size[1]
	tensor = torch.zeros( (2, len(images1), 3, h, w), dtype=torch.uint8 )
	for i, (img1, img2) in enumerate(zip(images1, images2)):
		nump_array1 = np.asarray(img1, dtype=np.uint8).transpose(2, 0, 1)
		nump_array2 = np.asarray(img2, dtype=np.uint8).transpose(2, 0, 1)
		tensor[0,i] += torch.from_numpy(nump_array1)
		tensor[1,i] += torch.from_numpy(nump_array2)
	
	return tensor, targets

class Dataset(torch.utils.data.Dataset):
	""" Wrapper for image data to store in memory """
	def __init__(self, csv_filename="train_1.csv", num_workers=10, test_size=0.1, seed=42, filter_new_whale=True,
	transform=None, print_fn=print):
		self.df = pd.read_csv(os.path.join(os.environ["data"], "humpback-whale-identification", csv_filename))
		self.transform = transform
		self.filter_new_whale = filter_new_whale
		if filter_new_whale:
			self.df = self.df[self.df.Id!="new_whale"].reset_index(drop=True)

		categories = self.df.Id.unique()
		categories.sort()
		label2idx = dict(zip(categories, list(range(categories.shape[0]))))
		self.df["Label"] = self.df.Id.apply(lambda x: label2idx[x])
		self.idx2label = dict(zip(list(range(categories.shape[0])), categories))
		
		self.num_workers = num_workers
		pool = mp.Pool(processes=num_workers)
		img_paths = [os.path.join("dataset", "train", name) for name in self.df.Image]
		self.images = pool.map(cv_loader, img_paths)

		if test_size > 0:
			self.__train_idx, self.__test_idx = train_test_split(self.df.index.values, test_size=test_size, random_state=seed)
			self.__train_idx, self.__test_idx = torch.from_numpy(self.__train_idx), torch.from_numpy(self.__test_idx)
		else:
			self.__train_idx, self.__test_idx = torch.from_numpy(self.df.index.values), []

	def reset_index(self, train, test):
		self.__train_idx, self.__test_idx = train, test
	
	def get_train_test_split(self):
		return self.__train_idx, self.__test_idx
	
	def __len__(self):
		return len(self.images)
	
	def __getitem__(self, idx):
		image = self.images[idx]

		if self.transform is not None:
			image = self.transform(image)
		
		return image, self.df.Label[idx], idx

class PairExtension(torch.utils.data.Dataset):
	""" Extension of Dataset when training on pairs """
	__MAX_TRIALS = 10000

	def __init__(self, dataset):
		self.dataset = dataset
	
	def get_train_indices(self, p=0.5):
		indices, _ = self.dataset.get_train_test_split()
		labels = [self.dataset.df.Label[i] for i in indices]
		perm = torch.randperm(len(indices))
		pair_indices = []
		for idx, label in zip(indices, labels):
			rand_idx = indices[random.randint(0, indices.size(0)-1)]
			if random.random() < p:
				i = 0
				for perm_idx in perm:
					i += 1
					if labels[perm_idx] == label and perm_idx != idx:
						rand_idx = indices[perm_idx]
						perm = torch.randperm(len(indices))
						break
					if i > self.__MAX_TRIALS:
						break
			
			else:
				while idx == rand_idx:
					rand_idx = indices[random.randint(0, indices.size(0)-1)]
			pair_indices.append((idx, rand_idx))
		
		return pair_indices
	
	def __len__(self):
		return len(self.dataset)
		
	def __getitem__(self, idxs: tuple):
		idx1, idx2 = idxs
		image1 = self.dataset.images[idx1]
		image2 = self.dataset.images[idx2]

		if self.dataset.transform is not None:
			image1 = self.dataset.transform(image1)
			image2 = self.dataset.transform(image2)
		
		# 0 is label idx for new whale
		if not self.dataset.filter_new_whale and (self.dataset.df.Label[idx1] == 0 or self.dataset.df.Label[idx2] == 0):
			label = 1
		else:
			label = int(self.dataset.df.Label[idx1] != self.dataset.df.Label[idx2])
		return image1, image2, label, idxs

class Prefetcher(object):
	""" Prefetcher returns a preloaded batch and starts copying next batch to GPU. """
	def __init__(self, loader):
		self.loader = iter(loader)
		self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
		self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
		self.stream = torch.cuda.Stream()
		self.preload()

	def preload(self):
		try:
			self.next_inputs, self.next_targets = next(self.loader)
		except StopIteration:
			self.next_inputs = None
			self.next_targets = None
			return
		with torch.cuda.stream(self.stream):
			self.next_inputs = self.next_inputs.cuda(non_blocking=True)
			self.next_targets = self.next_targets.cuda(non_blocking=True)
			self.next_inputs = self.next_inputs.float()
			self.next_inputs = self.next_inputs.sub_(self.mean).div_(self.std)
			
	def next_batch(self):
		torch.cuda.current_stream().wait_stream(self.stream)
		inputs = self.next_inputs
		targets = self.next_targets
		self.preload()
		return inputs, targets
