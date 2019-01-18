import os
import sys
import yaml
from types import SimpleNamespace
from collections import namedtuple
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as thd
import torchvision.transforms as transforms
from tqdm import tqdm

from dataset import Dataset, PairExtension, fast_collate, pair_collate, Prefetcher
from ctransforms import ConditionalPad
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from densenet import densenet121, densenet169, densenet201, densenet161
from siamese import SiameseWrapper
from loss import ContrastiveLoss, MarginLoss
from schedulers import Scheduler
from mixup import Mixup
from metrics import AverageMeter, topk_accuracy

def load_config(path):
	with open(path, "r") as f:
		c = yaml.load(f)
	config = SimpleNamespace(**c)
	return config

def save_config(config):
	with open(os.path.join(config.RUN_DIR, "config.yaml"), "w") as f:
		yaml.dump(config.__dict__, f, default_flow_style=False)

def setup_logger(no_snaps):
	global config
	logger = logging.getLogger()
	RESULTS_DIR = os.path.join(os.getcwd(), "checkpoints")
	if not os.path.exists(RESULTS_DIR):
		os.mkdir(RESULTS_DIR)
	
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)-8s %(message)s")
	std_handler = logging.StreamHandler(sys.stdout)
	std_handler.setFormatter(formatter)
	std_handler.setLevel(logging.INFO)
	logger.addHandler(std_handler)

	if not no_snaps:
		RUN_DIR = os.path.join(RESULTS_DIR, time.strftime("%Y%m%d-%X"))
		if not os.path.exists(RUN_DIR):
			os.mkdir(RUN_DIR)
		file_handler = logging.FileHandler(os.path.join(RUN_DIR, "experiment.log"), encoding="utf-8")
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
		config.RUN_DIR = RUN_DIR
	return logger

def set_seed(seed):
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.cuda.synchronize()
	torch.manual_seed(seed)

def get_transforms(config):
	transform = [transforms.ToPILImage()]
	for t in config.TRANSFORMS.split("|"):
		if t == "CPad":
			transform.append(ConditionalPad(config.IMAGE_SIZE))
		elif t == "RandomCrop":
			transform.append(transforms.RandomCrop(config.IMAGE_SIZE))
	return transforms.Compose(transform)

def get_dataset(config, logger):
	logger.info(f"Loading dataset with images of size {config.IMAGE_SIZE}")
	
	t0 = time.time()
	dataset = Dataset(csv_filename=config.TRAIN_FILE, num_workers=config.NW, test_size=config.TEST_SIZE,
	seed=config.SEED, filter_new_whale=config.EXCLUDE_WHALE, transform=get_transforms(config))
	logger.info("Dataset loaded at %.2fs" % (time.time() - t0))

	return dataset

def get_loaders(dataset, config, logger, pair_loaders=False):
	if pair_loaders:
		t0 = time.time()
		train_indices = dataset.get_train_indices(p=config.PAIR_SPLIT_P)
		logger.info("Indices created at %.2fs" % (time.time() - t0))
		loaders = {
			"train": thd.DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=thd.SubsetRandomSampler(train_indices), collate_fn=pair_collate, num_workers=config.NW),
		}
	else:
		train_indices, test_indices = dataset.get_train_test_split()
		phase_loader = namedtuple("PhaseLoader", ["batch", "single"])
		loaders = {
			"train": phase_loader(batch=thd.DataLoader(dataset, batch_size=config.EVAL_BATCH_SIZE, sampler=thd.SubsetRandomSampler(train_indices), collate_fn=fast_collate, num_workers=config.NW),
			single=thd.DataLoader(dataset, batch_size=1, sampler=thd.SubsetRandomSampler(train_indices), collate_fn=fast_collate, num_workers=config.NW)),
			"test": phase_loader(batch=thd.DataLoader(dataset, batch_size=config.EVAL_BATCH_SIZE, sampler=thd.SubsetRandomSampler(test_indices), collate_fn=fast_collate, num_workers=config.NW),
			single=thd.DataLoader(dataset, batch_size=1, sampler=thd.SubsetRandomSampler(test_indices), collate_fn=fast_collate, num_workers=config.NW))
		}
	return loaders

def load_model(config, logger):
	pretrained_models = {
		"resnet18": resnet18,
		"resnet34": resnet34,
		"resnet50": resnet50,
		"resnet101": resnet101,
		"resnet152": resnet152,
		"densenet121": densenet121,
		"densenet169": densenet169,
		"densenet201": densenet201,
		"densenet161": densenet161,
	}

	start = time.time()
	classifier = lambda num_features: nn.Linear(num_features, config.OUT_DIM)
	logger.info(f"Loading {config.MODEL}")
	if config.MODEL.startswith("densenet"):
		model = pretrained_models[config.MODEL](pretrained=True)
		num_ftrs = model.classifier.in_features
		model.classifier = classifier(num_ftrs)
	elif config.MODEL.startswith("resnet"):
		model = pretrained_models[config.MODEL](pretrained=True)
		num_ftrs = model.fc.in_features
		model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
		model.fc = classifier(num_ftrs)
	
	model = SiameseWrapper(model, distance_fn=F.pairwise_distance)
	if config.MULTI_GPU:
		logger.info("Using multiple GPUs")
		model = nn.DataParallel(model)
	model = model.cuda()

	logger.info("Model loaded at %.2fs" % (time.time() - start))
	return model

def load_criterion(config, logger):
	logger.info(f"Using {config.LOSS} loss function")
	if config.LOSS == "contrastive":
		criterion = ContrastiveLoss(margin=config.MARGIN)
	elif config.LOSS == "margin":
		criterion = MarginLoss(loss_lambda=config.LOSS_LAMBDA)
	else:
		raise ValueError(f"Received unknown loss function: {config.LOSS}")

	return criterion

def load_optimizer(model, config, logger):
	logger.info(f"Using {config.OPTIMIZER} optimization with learning rate {config.LR} and weight decay {config.WEIGHT_DECAY}")
	if config.OPTIMIZER == "sgd":
		optimizer = optim.SGD(model.parameters(), lr=config.LR, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
	elif config.OPTIMIZER == "adam":
		optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
	else:
		raise ValueError(f"Received unknown optimizer: {config.OPTIMIZER}")
	
	return optimizer

def save_model(model, config, snap_fname):
	if config.MULTI_GPU:
		torch.save(model.module.model.state_dict(), os.path.join(config.RUN_DIR, snap_fname))
	else:
		torch.save(model.model.state_dict(), os.path.join(config.RUN_DIR, snap_fname))

def train_loop(epoch, loader, model, criterion, optimizer, config, logger, mixup=None):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()

	model.train()
	end = time.time()
	scheduler = Scheduler(optimizer, len(loader), config.LR)
	pbar = tqdm(desc=f"TRAIN Epoch {epoch+1}", total=len(loader))
	prefetcher = Prefetcher(loader)

	inputs, targets = prefetcher.next_batch()
	i = -1
	while inputs is not None:
		i += 1

		scheduler.adjust_lr(epoch, i)
		data_time.update(time.time() - end)

		if mixup is not None:
			inputs, targets = mixup(inputs, targets)
		outputs = model(inputs)
		loss = criterion(outputs, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.update(loss.item())

		torch.cuda.synchronize()
		batch_time.update(time.time() - end)
		pbar.set_postfix(loss=losses.avg)
		pbar.update(1)
		end = time.time()
		inputs, targets = prefetcher.next_batch()
	
	pbar.close()

	logger.info("TRAIN Epoch: %d Loss: %.2f Batch Time: %.2fs Data Time: %.2fs" % (epoch+1, losses.avg, batch_time.avg, data_time.avg))

def eval_loop(epoch, loaders, model, criterion, config, logger, phase=""):
	batch_time = AverageMeter()
	image_time = AverageMeter()
	losses = AverageMeter()
	model.eval()
	pbar = tqdm(desc=f"{phase.upper()} EVAL Epoch {epoch+1}", total=len(loaders.single))
	prefetcher = Prefetcher(loaders.single)
	end_outer = time.time()
	with torch.no_grad():
		image_i = -1
		image, label = prefetcher.next_batch()
		image = image.expand(config.EVAL_BATCH_SIZE, -1, -1, -1)

		end = time.time()
		image_distances = torch.zeros(len(loaders.batch), config.EVAL_BATCH_SIZE).cuda()
		singles_targets = torch.zeros(len(loaders.single)).cuda().long()
		predictions = torch.zeros(len(loaders.single), 5).cuda().long()
		while image is not None:
			image_i += 1
			singles_targets[image_i].add_(label.long().item())

			batch_prefetcher = Prefetcher(loaders.batch)
			batch_targets = torch.zeros(len(loaders.batch), config.EVAL_BATCH_SIZE).cuda().long()
			inputs, targets = batch_prefetcher.next_batch()

			batch_i = -1
			image_distances.mul_(0)
			while inputs is not None:
				batch_i += 1

				distances = model.predict(image, inputs)
				if distances.size(0) < config.EVAL_BATCH_SIZE:
					image_distances[batch_i][:distances.size(0)].add_(distances)
					batch_targets[batch_i][:targets.size(0)].add_(targets.long())
				else:
					image_distances[batch_i].add_(distances)
					batch_targets[batch_i].add_(targets.long())

				torch.cuda.synchronize()
				batch_time.update(time.time() - end)
				end = time.time()
				inputs, targets = batch_prefetcher.next_batch()

			image_distances = image_distances.view(-1)[:len(loaders.single)]
			# TODO: Do something with pred_indexes to get label predictions
			values, indices = image_distances.topk(5, largest=False)
			
			pred_targets = batch_targets.view(-1)[:len(loaders.batch.dataset)][indices]
			if values.max() > criterion.margin:
				pred_targets[4] = 0
			predictions[image_i].add_(pred_targets)
			image_time.update(time.time() - end_outer)
			end_outer = time.time()
			image, label = prefetcher.next_batch()
			pbar.set_postfix()
			pbar.update(1)
	
	pbar.close()
	accuracies = topk_accuracy_preds(predictions, singles_targets, topk=(1,3,5))

	logger.info("%s \tEVAL Epoch: %d Loss: %.2f Batch Time: %.2fs Image Time: %.2fs" % (phase.upper(), epoch+1, losses.avg, batch_time.avg, image_time.avg))
	logger.info("\t\t\tTop-1: %.2f Top-3 %.2f Top-5 %.2f" % tuple(accuracies))

def single_run(dataset, model, config, logger, run_num=0):
	start = time.time()
	criterion = load_criterion(config, logger)
	optimizer = load_optimizer(model, config, logger)

	if config.MIXUP:
		logger.info(f"Using mixup with alpha={config.ALPHA}")
		mixup = Mixup(config.ALPHA)
	else:
		mixup = None

	loaders = get_loaders(dataset, config, logger)
	pair_dataset = PairExtension(dataset)
	train_loader = get_loaders(pair_dataset, config, logger, pair_loaders=True)["train"]
	for epoch in range(config.EPOCHS):

		if ((epoch+1) % config.REINDEX_INTERVAL) == 0:
			logger.info(f"Reindexing dataset at epoch {epoch+1}")
			train_loader = get_loaders(pair_dataset)["train"]

		train_loop(epoch, train_loader, model, criterion, optimizer, config, logger, mixup=mixup)

		if (epoch+1) % config.EVAL_INTERVAL == 0:
			for phase in ["train", "test"]:
				eval_loop(epoch, loaders[phase], model, criterion, config, logger, phase=phase)
	
	end = time.time() - start
	logger.info("Run %d finished at %dmin %.2fs" % (run_num, end // 60, end % 60))

def cross_validate(dataset, model, config, logger):
	kfold = KFold(n_splits=5, random_state=config.SEED)

	for split_num, (train_idx, test_idx) in enumerate(kfold.split(dataset.df.index.values)):
		logger.info(f"Running split {split_num}")
		dataset.reset_index(train_idx, test_idx)
		single_run(dataset, model, config, logger, run_num=split_num)

def main(args):
	global config
	config = load_config(args.config_file)
	config.NW = args.num_workers
	config.MULTI_GPU = args.multi_gpu
	config.CV = args.cross_validate
	config.MIXUP = args.mixup
	logger = setup_logger(args.no_snaps)
	if not args.no_snaps:
		save_config(config)

	logger.info(f"Setting seed {config.SEED}")
	set_seed(config.SEED)

	dataset = get_dataset(config, logger)
	model = load_model(config, logger)

	if config.CV:
		cross_validate(dataset, model, config, logger)
	else:
		single_run(dataset, model, config, logger)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Training CLI")
	parser.add_argument("config_file", metavar="FILEPATH", type=str, help="Config file.")
	parser.add_argument("--mixup", action="store_false", help="Flag whether to prevent mixup.")
	parser.add_argument("-cv", "--cross_validate", action="store_true", help="Flag whether to use cross validation.")
	parser.add_argument("--no_snaps", action="store_true", help="Flag whether to prevent from storing snapshots.")
	parser.add_argument("-nw", "--num_workers", metavar="INT", type=int, default=6, help="Number of processes (workers).")
	parser.add_argument("--gpu_device", metavar="INT", type=int, default=None, help="ID of a GPU to use when multiple GPUs are available.")
	parser.add_argument("--multi_gpu", action="store_true", help="Flag whether to use all available GPUs.")
	args = parser.parse_args()

	if args.gpu_device is not None:
		torch.cuda.set_device(args.gpu_device)

	main(args)
