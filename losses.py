import torch
import torch.nn as nn
import torch.nn.functional as F


def ava_pose_softmax_func(logits):
    pose_logits = nn.Softmax(dim=1)(logits[:, :13])
    interact_logits = nn.Sigmoid()(logits[:, 13:])
    logits = torch.cat([pose_logits, interact_logits], dim=1)
    logits = torch.clamp(logits, min=0., max=1.)
    return logits


def ava_pose_softmax_criterion(logits, targets):
    logits = ava_pose_softmax_func(logits)
    return F.binary_cross_entropy(logits, targets)


def ava_criterion(pose_softmax=False):
    if pose_softmax:
        return ava_pose_softmax_criterion, ava_pose_softmax_func
    return nn.BCEWithLogitsLoss(), nn.Sigmoid()
