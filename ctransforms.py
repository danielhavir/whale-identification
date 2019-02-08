# Custom transforms

import torch
import torchvision.transforms.functional as F
import cv2

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

class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bb = sample['image'], sample['bb']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        image = cv2.resize(image, (new_w, new_h))
        # h and w are swapped for mask because for images,
        # x and y axes are axis 1 and 0 respectively
        ratio = 2*[new_w / w, new_h / h]
        bb = bb * ratio
        sample['image'] = image
        sample['bb'] = bb.astype(int)
        return sample


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, bb = sample['image'], sample['bb']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h, left: left + new_w]
        bb = bb - [left, top, left, top]

        sample['image'] = image
        sample['bb'] = np.clip(bb, 0, new_h)
        sample['num'] = bb.shape[0]
        sample['shape'] = image.shape[:2]

        return sample

class Square(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        image, bb = sample['image'], sample['bb']
        x1, y1, x2, y2 = bb
        bb_w, bb_h = (x2 - x1), (y2 - y1)
        if bb_w > bb_h:
            image = image[:, x1:x2, :]
            bb -= [x1, 0, x1, 0]
        elif bb_w < bb_h:
            image = image[y1:y2, :, :]
            bb -= [0, y1, 0, y1]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size, self.output_size
        image = cv2.resize(image, (new_w, new_h))

        ratio = 2*[new_w / w, new_h / h]
        bb = bb * ratio

        sample['image'] = image
        sample['bb'] = bb.astype(int)
        return sample
