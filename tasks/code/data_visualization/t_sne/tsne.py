# -*- coding: utf-8 -*-
import torch
import torch.autograd
from torch import nn

from scipy.spatial import distance


class TSNE(nn.Module):
    def __init__(self, n_points, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super(TSNE, self).__init__()
        # Logit of datapoint-to-topic weight
        self.logits = nn.Embedding(n_points, n_dim)

    def forward(self, pij, i, j):
        # TODO: реализуйте вычисление матрицы сходства для точек отображения и расстояние Кульбака-Лейблера
        # pij - значения сходства между точками данных
        # i, j - индексы точек
        #qij = exp(distance.euclidean(i, j))
        dist = nn.modules.distance.PairwiseDistance()
        nominator = -torch.exp(dist(self.logits[i]), dist(self.logits[i]))
        denominator = torch.sum(torch.exp(dist(self.logits, self.ligits[i]))) -1 
        qij = nominator / denominator
        division = torch.div(pij, qij)
        loss_kld = torch.mul(pij, torch.log(division))
        return loss_kld.sum()

    def __call__(self, *args):
        return self.forward(*args)
