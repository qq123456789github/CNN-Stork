import torch.nn.functional as F


def nll_loss(output, target):
    return F.cross_entropy(output, target)
