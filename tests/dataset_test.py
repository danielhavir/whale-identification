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
from dataset import fast_collate, Dataset, PairDataset, Prefetcher
from ctransforms import ConditionalPad

class DatasetTest(unittest.TestCase):
    image_size = 224
    batch_size = 16
    dataset = Dataset()

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

    def test_transforms(self):
        t0 = time()
        self.dataset.transform = transforms.Compose([transforms.ToPILImage(), ConditionalPad(self.image_size), transforms.RandomCrop(self.image_size)])
        batch = self.dataset[0]
        self.assertEqual(type(batch[0]), PIL.Image.Image)
        self.assertEqual(batch[0].size, (self.image_size, self.image_size))
        self.assertEqual(type(batch[1]), np.int64)
        print("Transforms: %.2fs" % (time() - t0))

    def test_loaders(self):
        t0 = time()
        train_ids, _ = self.dataset.get_train_test_split()
        self.dataset.transform = transforms.Compose([transforms.ToPILImage(), ConditionalPad(self.image_size), transforms.RandomCrop(self.image_size)])
        train_loader = thd.DataLoader(self.dataset, batch_size=self.batch_size, sampler=thd.SubsetRandomSampler(train_ids), collate_fn=fast_collate)
        batch = iter(train_loader).next()
        self.assertEqual(type(batch[0]), torch.Tensor)
        self.assertEqual(batch[0].dtype, torch.uint8)
        self.assertEqual(batch[0].size(), torch.Size([self.batch_size, 3, self.image_size, self.image_size]))
        self.assertEqual(batch[0].device, torch.device("cpu"))
        self.assertEqual(type(batch[1]), torch.Tensor)
        self.assertEqual(batch[1].dtype, torch.int64)
        self.assertEqual(batch[1].size(), torch.Size([self.batch_size]))
        print("Transforms: %.2fs" % (time() - t0))

    def test_prefetcher(self):
        t0 = time()
        train_ids, _ = self.dataset.get_train_test_split()
        train_loader = thd.DataLoader(self.dataset, batch_size=self.batch_size, sampler=thd.SubsetRandomSampler(train_ids), collate_fn=fast_collate)
        prefetcher = Prefetcher(train_loader)
        t1 = time()
        p_init_time = t1 - t0
        batch = prefetcher.next_batch()
        p_batch_time = time() - t1
        p_total_time = time() - t0
        self.assertEqual(batch[0].dtype, torch.float32)
        self.assertEqual(batch[0].size(), torch.Size([self.batch_size, 3, self.image_size, self.image_size]))
        self.assertEqual(batch[0].device, torch.device("cuda", index=0))
        self.assertEqual(batch[1].dtype, torch.int64)
        self.assertEqual(batch[1].size(), torch.Size([self.batch_size]))
        self.assertEqual(batch[1].device, torch.device("cuda", index=0))
        print("Prefetcher: %.2fs Init time: %.2fs Batch time: %.2fs" % (p_total_time, p_init_time, p_batch_time))


if __name__ == '__main__':
    unittest.main()
