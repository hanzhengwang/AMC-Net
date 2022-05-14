import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable


class hetero_loss(nn.Module):
    def __init__(self, margin=0.1, dist_type='l2'):
        super(hetero_loss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()
        self.ranking_loss = nn.MarginRankingLoss(margin=0.3)
        # self.w1 = nn.Parameter(torch.tensor(1).float().cuda())
        # self.w2 = nn.Parameter(torch.tensor(1).float().cuda())

    def forward(self, feat1, feat2, label1, label2):
        feats = torch.cat((feat1, feat2), dim=0)
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        # loss = Variable(.cuda())
        for i in range(label_num):
            center1 = torch.mean(feat1[i], dim=0)
            center2 = torch.mean(feat2[i], dim=0)
            if self.dist_type == 'l2' or self.dist_type == 'l1':
                if i == 0:
                    dist_ = max(0, self.dist(center1, center2) - self.margin)
                else:
                    dist_ += max(0, self.dist(center1, center2) - self.margin)
            elif self.dist_type == 'cos':
                if i == 0:
                    dist_ = max(0, 1 - self.dist(center1, center2) - self.margin)
                else:
                    dist_ += max(0, 1 - self.dist(center1, center2) - self.margin)

        return dist_

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx
