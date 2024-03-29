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

import dataset as d
import ctransforms
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from densenet import densenet121, densenet169, densenet201, densenet161
from siamese import SiameseWrapper, TripletWrapper, similarity_matrix, TripletHead
from loss import ContrastiveLoss, TripletLoss, MarginLoss
from schedulers import Scheduler
from mixup import Mixup
from metrics import AverageMeter, topk_accuracy, topk_accuracy_preds, unique

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
	boxtransform = []
	for t in config.TRANSFORMS.split("|"):
		if t == "CPad":
			transform.append(ctransforms.ConditionalPad(config.IMAGE_SIZE))
		elif t == "Resize":
			transform.append(transforms.Resize(config.IMAGE_SIZE+1))
		elif t == "RandomCrop":
			transform.append(transforms.RandomCrop(config.IMAGE_SIZE))
		elif t == "Gray":
			transform.append(transforms.Grayscale(num_output_channels=3))
		elif t == "Pad":
			transform.append(transforms.Pad(20, padding_mode="edge"))
		elif t == "Rotate":
			transform.append(transforms.RandomRotation(10))
	for t in config.BOXTRANSFORMS.split("|"):
		if t == "Square":
			boxtransform.append(ctransforms.Square(config.IMAGE_SIZE))
		elif t == "Resize":
			boxtransform.append(ctransforms.Resize(config.IMAGE_SIZE+1))
		elif t == "RandomCrop":
			boxtransform.append(ctransforms.RandomCrop(config.IMAGE_SIZE))
	return (transforms.Compose(boxtransform), transforms.Compose(transform))

def get_dataset(config, logger):
	logger.info(f"Loading dataset with images of size {config.IMAGE_SIZE}")
	
	t0 = time.time()
	box_transform, transform = get_transforms(config)
	dataset = d.Dataset(csv_filename=config.TRAIN_FILE, num_workers=config.NW, test_size=config.TEST_SIZE,
	seed=config.SEED, filter_new_whale=config.EXCLUDE_WHALE, transform=transform, box_transform=box_transform)
	logger.info("Dataset loaded at %.2fs" % (time.time() - t0))

	return dataset

def get_loaders(dataset, config, logger, n=1):
	if n == 2:
		t0 = time.time()
		train_indices = dataset.get_train_indices(p=config.PAIR_SPLIT_P)
		logger.info("Indices created at %.2fs" % (time.time() - t0))
		loaders = {
			"train": thd.DataLoader(d.PairExtension(dataset), batch_size=config.BATCH_SIZE, sampler=thd.SubsetRandomSampler(train_indices), collate_fn=d.pair_collate, num_workers=config.NW),
		}
	elif n == 3:
		train_indices, test_indices = dataset.get_train_test_split()
		if dataset.transform is None:
			collate_fn = d.triple_collate
		else:
			collate_fn = d.triple_collate_pil
		loaders = {
			"train": thd.DataLoader(d.TripletExtension(dataset), batch_size=config.BATCH_SIZE, sampler=thd.SubsetRandomSampler(train_indices), collate_fn=collate_fn, num_workers=config.NW),
		}
	else:
		train_indices, test_indices = dataset.get_train_test_split()
		if dataset.transform is None:
			collate_fn = d.fast_collate
		else:
			collate_fn = d.fast_collate_pil
		loaders = {
			"train": thd.DataLoader(dataset, batch_size=config.EVAL_BATCH_SIZE, sampler=thd.SubsetRandomSampler(train_indices), collate_fn=collate_fn, num_workers=config.NW),
			"test": thd.DataLoader(dataset, batch_size=config.EVAL_BATCH_SIZE, sampler=thd.SubsetRandomSampler(test_indices), collate_fn=collate_fn, num_workers=config.NW)
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
	logger.info(f"Loading {config.MODEL}")
	if config.MODEL.startswith("head"):
		model = TripletHead(pretrained_models[config.MODEL[5:]](pretrained=True))
	else:
		classifier = lambda num_features: nn.Sequential(
			nn.Linear(num_features, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, config.OUT_DIM)
		)
		if config.MODEL.startswith("densenet"):
			model = pretrained_models[config.MODEL](pretrained=True)
			num_ftrs = model.classifier.in_features
			model.classifier = classifier(num_ftrs)
		elif config.MODEL.startswith("resnet"):
			model = pretrained_models[config.MODEL](pretrained=True)
			num_ftrs = model.fc.in_features
			model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
			model.fc = classifier(num_ftrs)
		
		if config.DATASET_N == 2:
			model = SiameseWrapper(model, distance_fn=F.pairwise_distance)
		elif config.DATASET_N == 3:
			model = TripletWrapper(model)
		else:
			raise ValueError(f"Received unknown dataset n: {config.DATASET_N}")
	
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
	elif config.LOSS == "triplet":
		criterion = TripletLoss(margin=config.MARGIN)
	elif config.LOSS == "margin":
		criterion = MarginLoss(loss_lambda=config.LOSS_LAMBDA)
	elif config.LOSS == "bce":
		criterion = nn.BCEWithLogitsLoss()
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

def train_loop(epoch, loader, model, criterion, optimizer, config, logger, mixup=None, scheduler=None):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	acc = AverageMeter()

	model.train()
	end = time.time()
	#scheduler = Scheduler(optimizer, len(loader), config.LR)
	pbar = tqdm(desc=f"TRAIN Epoch {epoch+1}", total=len(loader))
	prefetcher = d.TripletPrefetcher(loader)

	inputs = prefetcher.next_batch()
	ones = torch.ones(inputs.size(0)).cuda()
	zeros = torch.zeros(inputs.size(0)).cuda()
	i = -1
	while inputs is not None:
		i += 1
		batch_size = inputs.size(0)

		#scheduler.adjust_lr(epoch, i)
		data_time.update(time.time() - end)

		if mixup is not None:
			inputs = mixup(inputs)
		
		positive, negative = model(inputs)
		
		pos_loss = criterion(positive.view(-1), ones[:batch_size])
		neg_loss = criterion(negative.view(-1), zeros[:batch_size])
		loss = pos_loss + neg_loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.update(loss.item(), n=batch_size)
		acc.update(((torch.sigmoid(positive.view(-1))>0.5)==ones[:batch_size].byte()).float().mean().item(), n=batch_size)
		acc.update(((torch.sigmoid(negative.view(-1))>0.5)==zeros[:batch_size].byte()).float().mean().item(), n=batch_size)

		torch.cuda.synchronize()
		batch_time.update(time.time() - end)
		pbar.set_postfix(loss=losses.avg, acc=acc.avg)
		pbar.update(1)
		end = time.time()
		inputs = prefetcher.next_batch()
	
	pbar.close()
	if scheduler is not None:
		scheduler.step()

	logger.info("TRAIN Epoch: %d Loss: %.2f Acc: %.2f Batch Time: %.2fs Data Time: %.2fs" % (epoch+1, losses.avg, acc.avg, batch_time.avg, data_time.avg))

def eval_loop(epoch, loaders, model, criterion, config, logger):
	batch_time = AverageMeter()
	total_images = len(loaders["train"].dataset)
	num_images = loaders["train"].sampler.indices.size(0)
	num_test_images = loaders["test"].sampler.indices.size(0)
	model.eval()
	pbar = tqdm(desc=f"TRAIN EVAL Epoch {epoch+1}", total=len(loaders["train"]))
	prefetcher = d.Prefetcher(loaders["train"])
	with torch.no_grad():
		batch_i = -1
		inputs, targets = prefetcher.next_batch()

		end = time.time()
		train_labels = torch.zeros(len(loaders["train"]), config.EVAL_BATCH_SIZE).cuda().long()
		# First num_images are train predictions, the rest (num_test_images) are test predictions
		predictions = torch.zeros(len(loaders["train"])+len(loaders["test"]), config.EVAL_BATCH_SIZE, config.OUT_DIM).cuda()
		while inputs is not None:
			batch_i += 1
			preds = model.features(inputs)
			if preds.size(0) < config.EVAL_BATCH_SIZE:
				predictions[batch_i][:preds.size(0)].add_(preds)
				train_labels[batch_i][:targets.size(0)].add_(targets.long())
			else:
				predictions[batch_i].add_(preds)
				train_labels[batch_i].add_(targets.long())

			torch.cuda.synchronize()
			batch_time.update(time.time() - end)
			end = time.time()
			inputs, targets = prefetcher.next_batch()
			pbar.set_postfix()
			pbar.update(1)
	
		pbar.close()
		pbar = tqdm(desc=f"TEST EVAL Epoch {epoch+1}", total=len(loaders["test"]))
		prefetcher = d.Prefetcher(loaders["test"])
		inputs, targets = prefetcher.next_batch()

		end = time.time()
		test_labels = torch.zeros(len(loaders["test"]), config.EVAL_BATCH_SIZE).cuda().long()
		test_i = -1
		while inputs is not None:
			batch_i += 1
			test_i += 1
			preds = model.features(inputs)
			if preds.size(0) < config.EVAL_BATCH_SIZE:
				predictions[batch_i][:preds.size(0)].add_(preds)
				test_labels[test_i][:targets.size(0)].add_(targets.long())
			else:
				predictions[batch_i].add_(preds)
				test_labels[test_i].add_(targets.long())

			torch.cuda.synchronize()
			batch_time.update(time.time() - end)
			end = time.time()
			inputs, targets = prefetcher.next_batch()
			pbar.set_postfix()
			pbar.update(1)
	pbar.close()

	train_predictions = predictions[:len(loaders["train"])].view(-1, config.OUT_DIM)[:num_images]
	test_predictions = predictions[len(loaders["train"]):].view(-1, config.OUT_DIM)[:num_test_images]
	predictions = torch.zeros(total_images, config.OUT_DIM).cuda()
	predictions[:num_images].add_(train_predictions)
	predictions[num_images:].add_(test_predictions)
	del inputs, targets, train_predictions, test_predictions, prefetcher
	torch.cuda.empty_cache()

	train_labels = train_labels.view(-1)[:num_images]
	test_labels = test_labels.view(-1)[:num_test_images]
	#breakpoint()
	head_batch_size = 256*config.EVAL_BATCH_SIZE
	loader = thd.DataLoader(thd.TensorDataset(predictions[:num_images]), batch_size=head_batch_size, shuffle=False)
	matrix = torch.zeros(total_images, num_images).cuda()
	with torch.no_grad():
		for i, p in enumerate(tqdm(predictions, desc=f"HEAD EVAL", total=total_images)):
			p = p.expand(num_images, predictions.size(1))
			if i < num_images-1:
				i += 1
				#breakpoint()
				try:
					tmp = torch.sigmoid(model.head(p[i:], predictions[i:num_images])).view(-1)
				except:
					print(i)
					breakpoint()
				matrix[i,i:].add_(tmp)
			elif i >= num_images:
				tmp = torch.sigmoid(model.head(p, predictions[:num_images])).view(-1)
				matrix[i].add_(tmp)
	
	matrix[:num_images,:] = matrix[:num_images,:].add(matrix[:num_images,:].t())
	values, indices = matrix.topk(6, 1, largest=True)
	pred_targets = train_labels[indices]
	unique_preds = unique(pred_targets)
	unique_preds[:,-1].mul_((pred_targets.min(1)[0] == 0).long())
	accuracies = topk_accuracy_preds(unique_preds[:num_images], train_labels, topk=(1,3,5))
	logger.info("TRAIN EVAL Epoch: %d Top-1: %.2f Top-3 %.2f Top-5 %.2f" % ((epoch+1,) + tuple(accuracies)))
	accuracies = topk_accuracy_preds(unique_preds[num_images:], test_labels, topk=(1,3,5))
	logger.info("TEST EVAL Epoch: %d Top-1: %.2f Top-3 %.2f Top-5 %.2f" % ((epoch+1,) + tuple(accuracies)))

def single_run(dataset, model, config, logger, run_num=0):
	start = time.time()
	criterion = load_criterion(config, logger)
	optimizer = load_optimizer(model, config, logger)
	scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

	if config.MIXUP:
		logger.info(f"Using mixup with alpha={config.ALPHA}")
		mixup = Mixup(config.ALPHA)
	else:
		mixup = None

	loaders = get_loaders(dataset, config, logger)
	train_loader = get_loaders(dataset, config, logger, n=config.DATASET_N)["train"]
	for epoch in range(config.EPOCHS):

		train_loop(epoch, train_loader, model, criterion, optimizer, config, logger, mixup=mixup, scheduler=scheduler)

		if (epoch+1) % config.EVAL_INTERVAL == 0:
			eval_loop(epoch, loaders, model, criterion, config, logger)

		if config.DATASET_N == 2 and ((epoch+1) % config.REINDEX_INTERVAL) == 0 and (epoch+1) < config.EPOCHS:
			logger.info(f"Reindexing dataset at epoch {epoch+1}")
			train_loader = get_loaders(dataset, config, logger, n=config.DATASET_N)["train"]
	
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
	parser.add_argument("--mixup", action="store_true", help="Flag whether to prevent mixup.")
	parser.add_argument("-cv", "--cross_validate", action="store_true", help="Flag whether to use cross validation.")
	parser.add_argument("--no_snaps", action="store_true", help="Flag whether to prevent from storing snapshots.")
	parser.add_argument("-nw", "--num_workers", metavar="INT", type=int, default=6, help="Number of processes (workers).")
	parser.add_argument("--gpu_device", metavar="INT", type=int, default=None, help="ID of a GPU to use when multiple GPUs are available.")
	parser.add_argument("--multi_gpu", action="store_true", help="Flag whether to use all available GPUs.")
	args = parser.parse_args()

	if args.gpu_device is not None:
		torch.cuda.set_device(args.gpu_device)

	main(args)
