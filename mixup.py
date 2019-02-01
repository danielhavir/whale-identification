import torch
import torch.nn as nn
import torch.nn.functional as F

class Mixup(object):
    def __init__(self, alpha, beta=None):
        alpha1 = alpha
        alpha2 = beta if beta is not None else alpha
        self.beta = torch.distributions.Beta(alpha2, alpha1)
        self.one = torch.tensor(1.).cuda()
    
    def __call__(self, images, targets=None):
        perm = torch.randperm(images.size(0))
        perm_images = images[perm]
        lam = self.beta.sample((images.size(0),)).cuda()
        images = lam.view(-1,1,1,1,1).expand(-1,3,-1,-1,-1).mul(images) + (self.one-lam).view(-1,1,1,1,1).expand(-1,3,-1,-1,-1).mul(perm_images)
        if targets is not None:
            perm_targets = targets[perm]
            targets = targets.mul_(lam) + perm_targets.mul_(self.one-lam)
            return images, targets
        else:
            return images

class BinaryCrossEntropy(nn.Module):
    def __init__(self, device: torch.device, weight=torch.tensor([1., 1.]), size_average=True):
        super(BinaryCrossEntropy, self).__init__()
        self.size_average = size_average
        self.one = torch.tensor(1.).to(device)
        self.weight = weight.to(device)
    
    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)

        if self.size_average:
            return torch.mean(- torch.log(inputs) * targets * self.weight[1] - (self.one - targets) * torch.log(self.one - inputs) * self.weight[0])
        else:
            return torch.sum(- torch.log(inputs) * targets * self.weight[1] - (self.one - targets) * torch.log(self.one - inputs) * self.weight[0])

class OneHotCrossEntropy(nn.Module):
    def __init__(self, device: torch.device):
        super(OneHotCrossEntropy, self).__init__()
        self.one = torch.tensor(1.).to(device)
    
    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)

        return torch.sum(- inputs * targets) / targets.size(0)
