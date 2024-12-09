import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.loss_mnist2 import *
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class Conv2dSamePad(nn.Module):
    """
    Implement Tensorflow's 'SAME' padding mode in Conv2d.
    When an odd number, say `m`, of pixels are need to pad, Tensorflow will pad one more column at right or one more
    row at bottom. But Pytorch will pad `m+1` pixels, i.e., Pytorch always pads in both sides.
    So we can pad the tensor in the way of Tensorflow before call the Conv2d module.
    """
    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):
    """
    This module implements the "SAME" padding mode for ConvTranspose2d as in Tensorflow.
    A tensor with width w_in, feed it to ConvTranspose2d(ci, co, kernel, stride), the width of output tensor T_nopad:
        w_nopad = (w_in - 1) * stride + kernel
    If we use padding, i.e., ConvTranspose2d(ci, co, kernel, stride, padding, output_padding), the width of T_pad:
        w_pad = (w_in - 1) * stride + kernel - (2*padding - output_padding) = w_nopad - (2*padding - output_padding)
    Yes, in ConvTranspose2d, more padding, the resulting tensor is smaller, i.e., the padding is actually deleting row/col.
    If `pad`=(2*padding - output_padding) is odd, Pytorch deletes more columns in the left, i.e., the first ceil(pad/2) and
    last `pad - ceil(pad/2)` columns of T_nopad are deleted to get T_pad.
    In contrast, Tensorflow deletes more columns in the right, i.e., the first floor(pad/2) and last `pad - floor(pad/2)`
    columns are deleted.
    For the height, Pytorch deletes more rows at top, while Tensorflow at bottom.
    In practice, we usually want `w_pad = w_in * stride` or `w_pad = w_in * stride - 1`, i.e., the "SAME" padding mode
    in Tensorflow. To determine the value of `w_pad`, we should pass it to this function.
    So the number of columns to delete:
        pad = 2*padding - output_padding = w_nopad - w_pad
    If pad is even, we can directly set padding=pad/2 and output_padding=0 in ConvTranspose2d.
    If pad is odd, we can use ConvTranspose2d to get T_nopad, and then delete `pad` rows/columns by
    ourselves.
    This module should be called after the ConvTranspose2d module with shared kernel_size and stride values.
    """

    def __init__(self, output_size):
        super(ConvTranspose2dSamePad, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = in_height - self.output_size[0]
        pad_width = in_width - self.output_size[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module(
                'conv%d' % i,
                nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2)
            )
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        sizes = [[7, 7], [14, 14], [28, 28]]
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module(
                'deconv%d' % (i + 1),
                nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2)
            )
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(sizes[i]))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y

class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-5 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


class DSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample, dim, num_classes, temperature, device):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.cluster_num = num_classes
        self.rep_dim = dim
        self.ae1 = ConvAE(channels, kernels)
        self.ae2 = ConvAE(channels, kernels)
        self.self_expression1 = SelfExpression(self.n)
        self.self_expression2 = SelfExpression(self.n)
        self.cl1 = nn.Sequential(
            nn.Linear(self.rep_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.cluster_num),
            nn.Softmax(dim=1)
        )
        self.cl2 = nn.Sequential(
            nn.Linear(self.rep_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.cluster_num),
            nn.Softmax(dim=1)
        )
        self.dscloss = DSCLoss(self.n, device)
        self.cluster_loss = ClusterLoss(self.cluster_num, temperature, device)
        self.ranking_loss = Ranking_Loss(self.n)

    def forward(self, xi, xj):  # shape=[n, c, w, h]
        zi = self.ae1.encoder(xi)
        zj = self.ae1.encoder(xj)#

        shape = zi.shape
        zi = zi.view(self.n, -1)
        zj = zj.view(self.n, -1)
        zi_recon = self.self_expression1(zi)
        zj_recon = self.self_expression1(zj) #
        zi_recon_reshape = zi_recon.view(shape)
        zj_recon_reshape = zj_recon.view(shape)
        xi_recon = self.ae1.decoder(zi_recon_reshape)
        xj_recon = self.ae1.decoder(zj_recon_reshape)#

        ci = self.cl1(zi)
        cj = self.cl2(zj)

        return xi_recon, xj_recon, zi, zj, zi_recon, zj_recon, ci, cj, self.self_expression1.Coefficient, self.self_expression2.Coefficient

    def loss_fn(self, xi, xj, xi_recon, xj_recon, zi, zj, zi_recon, zj_recon, ci, cj, weight_coef, weight_selfExp, weight_con, lambda1, lambda2, neg,lambda4, k, theta):
        dscloss = self.dscloss(xi, xj, xi_recon, xj_recon, zi, zj, zi_recon, zj_recon, self.self_expression1.Coefficient, self.self_expression1.Coefficient, weight_coef, weight_selfExp)
        cluster_loss = self.cluster_loss(ci, cj, weight_con)
        dis_loss, simz, ranking_z, ranking_z_cut, psim, psim_cut = self.ranking_loss(ci, cj, self.self_expression1.Coefficient, lambda1, lambda2, neg, lambda4, k, theta)
        loss = dscloss + cluster_loss + dis_loss

        return loss, dscloss, cluster_loss, dis_loss

