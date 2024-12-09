import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DSCLoss(nn.Module):
    def __init__(self, num_samples, device):
        super(DSCLoss, self).__init__()
        self.N = num_samples
        self.device = device

    def forward(self, xi, xj, xi_recon, xj_recon, zi, zj, zi_recon, zj_recon, self_expression1, self_expression2, weight_coef, weight_selfExp):
        loss_ae1 = F.mse_loss(xi_recon, xi, reduction='sum')
        loss_ae2 = F.mse_loss(xj_recon, xj, reduction='sum')
        loss_ae = loss_ae1 + loss_ae2

        target = torch.zeros(xi.shape[0], xi.shape[0], device=xi.device)
        loss_coef1 = F.mse_loss(self_expression1, target, reduction='sum') + torch.diagonal(torch.square(self_expression1)).sum() # torch.sum(torch.pow(self.self_expression1.Coefficient, 2))#
        loss_coef2 = F.mse_loss(self_expression2, target, reduction='sum') + torch.diagonal(torch.square(self_expression2)).sum() # torch.sum(torch.pow(self.self_expression2.Coefficient, 2))#
        loss_coef = loss_coef1 + loss_coef2
        loss_selfExp1 = F.mse_loss(zi_recon, zi, reduction='sum')
        loss_selfExp2 = F.mse_loss(zj_recon, zj, reduction='sum')
        loss_selfExp = loss_selfExp1 + loss_selfExp2
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * 0.5*loss_selfExp
        loss /= self.N

        return loss

class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j, con):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return (loss + ne_loss)*con

def ranking_statistics(s, K, symetric=False):
    self_expression = s.detach()
    if symetric:
        self_expression = self_expression - torch.diagflat(torch.diagonal(self_expression))
    s_sort, indices = torch.topk(self_expression, K, dim=1)
    s_sort_min, _ = torch.min(s_sort, 1)
    mask = torch.greater(self_expression, s_sort_min.unsqueeze(1))
    mask = mask.to(torch.float32).fill_diagonal_(1)
    se = s.clone()
    s = se + torch.finfo(torch.float32).tiny * torch.ones_like(s)
    s_mask = (s / s.detach()) * mask.to(s.device)
    psim = torch.matmul(s_mask, s_mask.T) / K
    return psim


class Ranking_Loss(nn.Module):
    def __init__(self, num_sample):
        super(Ranking_Loss, self).__init__()
        self.N = num_sample
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.softmax = nn.Softmax(dim=1)
    def forward(self, z1, z2, s, lambda1, lambda2, neg, lambda4, k, theta):
        z1 = z1 / torch.norm(z1, dim=-1, keepdim=True)
        z2 = z2 / torch.norm(z2, dim=-1, keepdim=True)
        simz = torch.mm(z1, z2.T)

        ranking_z = ranking_statistics(simz, K=k, symetric=True)
        filter_z = torch.where(ranking_z > theta, torch.ones_like(ranking_z), torch.zeros_like(ranking_z))
        b = ranking_z.clone()
        ranking_z = b + torch.finfo(torch.float32).tiny * torch.ones_like(ranking_z)
        ranking_z_cut = (ranking_z / ranking_z.detach()) * filter_z

        sims = (abs(s) + abs(s.t())) / 2
        psim = ranking_statistics(sims, K=k, symetric=True)
        mval, _ = sims.max(dim=1)
        mval = mval.unsqueeze(dim=1)
        sims1 = sims / mval.repeat(1, sims.size(1))

        filter_s = torch.where(psim > theta, torch.ones_like(psim), torch.zeros_like(psim))
        a = psim.clone()
        psim = a + torch.finfo(torch.float32).tiny * torch.ones_like(psim)
        psim_cut = (psim / psim.detach()) * filter_s

        loss1 = self.criterion(ranking_z_cut.detach(), sims1)/self.N
        loss2 = self.criterion(psim_cut.detach(), simz)/self.N
        label = (psim_cut.detach() + ranking_z_cut.detach())/2
        loss3 = self.criterion(ranking_z, label)/self.N
        loss4 = self.criterion(psim, label)/self.N

        loss = lambda1*loss1 + lambda2*loss2 + neg*loss3 + lambda4*loss4

        return loss, simz, ranking_z, ranking_z_cut, psim, psim_cut


