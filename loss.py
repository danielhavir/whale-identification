import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginLoss(nn.Module):
    def __init__(self, size_average=False, loss_lambda=0.5):
        '''
        Margin loss for digit existence
        Eq. L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda * (1 - T_k) * max(0, ||v_k|| - m-)^2
        
        Args:
            size_average: should the losses be averaged (True) or summed (False) over observations for each minibatch.
            loss_lambda: parameter for down-weighting the loss for missing digits
        '''
        super(MarginLoss, self).__init__()
        self.size_average = size_average
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.loss_lambda = loss_lambda

    def forward(self, inputs, labels):
        L_k = labels * F.relu(self.m_plus - inputs)**2 + self.loss_lambda * (1 - labels) * F.relu(inputs - self.m_minus)**2
        L_k = L_k.sum(dim=1)

        if self.size_average:
            return L_k.mean()
        else:
            return L_k.sum()

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class TripletLoss(nn.Module):
    """
    Triplet loss
    """

    def __init__(self, margin, size_average=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if self.size_average else losses.sum()
