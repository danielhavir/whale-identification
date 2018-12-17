# Custom transforms

import torch
import torchvision.transforms.functional as F

class ConditionalPad(object):
	""" ConditionalPad pads the image if and only if either of the image sizes is below a given threshold """
	def __init__(self, size, fill=0, padding_mode='symmetric'):
		self.size = size
		self.fill = fill
		self.padding_mode = padding_mode

	def __call__(self, img):
		w, h = img.size
		padding_right = (self.size - w) if w < self.size else 0
		padding_top = (self.size - h) if h < self.size else 0
		return F.pad(img, (0, padding_top, padding_right, 0), self.fill, self.padding_mode)
