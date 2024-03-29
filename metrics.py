import torch

class AverageMeter(object):
    """
        Computes and stores the average and current value
        Reference: https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main.py#L442-L457
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def unique(preds, num=5):
    dset_size = preds.size(0)
    res = torch.zeros(dset_size, num).cuda().long().sub_(1)
    for i in range(dset_size):
        u = preds[i].unique(sorted=False)
        if u.size(0) > num:
            res[i,:u.size(0)].mul_(0).add_(preds[i, :num])
        else:
            res[i,:u.size(0)].mul_(0).add_(u)
    return res

def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / dset_size).item())
    return res

def topk_accuracy_preds(pred, target, topk=(1,)):
    dset_size = target.size(0)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / dset_size).item())
    return res

