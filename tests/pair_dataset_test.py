import os
import sys
from time import time
import unittest

sys.path.append(os.getcwd())

import torch
import torch.utils.data as thd
import torchvision.transforms as transforms
import numpy as np
import PIL
from dataset import pair_collate, PairDataset, Prefetcher
from ctransforms import ConditionalPad

class PairDatasetTest(unittest.TestCase):
    image_size = 224
    batch_size = 16
    dataset = PairDataset()

    def test_dataset(self):
        t0 = time()
        batch = self.dataset[0]
        self.assertEqual(type(batch[0]), np.ndarray)
        self.assertEqual(type(batch[1]), np.int64)
        print("Dataset: %.2fs" % (time() - t0))

    def test_split(self):
        t0 = time()
        train_ids, test_ids = self.dataset.get_train_test_split()
        self.assertEqual(train_ids.size(0)+test_ids.size(0), len(self.dataset))
        self.dataset.transform = None
        train_loader = thd.DataLoader(self.dataset, batch_size=1, sampler=thd.SubsetRandomSampler(train_ids))
        test_loader = thd.DataLoader(self.dataset, batch_size=1, sampler=thd.SubsetRandomSampler(test_ids))
        loader_train_ids, _ = torch.tensor([batch[2][0] for batch in train_loader]).sort()
        loader_test_ids, _ = torch.tensor([batch[2][0] for batch in test_loader]).sort()
        self.assertEqual(loader_train_ids.eq(train_ids.sort()[0]).sum(), train_ids.size(0))
        self.assertEqual(loader_test_ids.eq(test_ids.sort()[0]).sum(), test_ids.size(0))
        # Check that test IDs do not leak with training
        self.assertEqual(np.intersect1d(loader_train_ids.numpy(), test_ids.numpy()).shape[0], 0)
        self.assertEqual(np.intersect1d(loader_train_ids.numpy(), loader_test_ids.numpy()).shape[0], 0)
        self.assertEqual(np.intersect1d(loader_test_ids.numpy(), train_ids.numpy()).shape[0], 0)
        print("Split: %.2fs" % (time() - t0))

if __name__ == '__main__':
    unittest.main()
