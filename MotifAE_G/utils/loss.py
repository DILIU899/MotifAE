import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_soft_sort.pytorch_ops import soft_rank


def zero_one_loss(input: torch.Tensor):
    loss = torch.mean(input * (1 - input))

    return loss


def sparsity_loss(input: torch.Tensor):
    # l1 loss
    return input.norm(p=1, dim=-1).mean()


def spearman_loss(pred: torch.Tensor, label: torch.Tensor, reg_strength=1, reg_method="kl"):
    # pred: [batch]
    # label: [batch]

    res = torch.stack([pred, label])
    res_rank = soft_rank(
        res, regularization_strength=reg_strength, regularization=reg_method, direction="ASCENDING"
    )

    pred_n = res_rank[0] - res_rank[0].mean()
    target_n = res_rank[1] - res_rank[1].mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    soft_spearman = (pred_n * target_n).sum()

    loss = 1 - soft_spearman

    return loss


def l2_loss(pred: torch.Tensor, label: torch.Tensor):
    # pred: [batch]
    # label: [batch]
    return F.mse_loss(pred, label)
