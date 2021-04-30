# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from mobile_net_v2 import mobile_net_v2

class NIMA(nn.Module):
    def __init__(self, base_model):
        super(NIMA, self).__init__()
        base_model = nn.Sequential(*list(base_model.children())[:-2])

        self.features = base_model

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(100352,1024),
            nn.Linear(1024,2),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

        


def single_emd_loss(p, q, r=2):
    """
    Earth Mover's Distance of one sample

    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += sum(torch.abs(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)


def emd_loss(p, q, r=2):
    """
    Earth Mover's Distance on a batch

    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size

