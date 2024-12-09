import random
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
from utils.transform import Transforms
from utils.loss_coil20 import *

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
    In practice, we usually want `w_pad = w_in * stride`, i.e., the "SAME" padding mode in Tensorflow,
    so the number of columns to delete:
        pad = 2*padding - output_padding = kernel - stride
    We can solve the above equation and get:
        padding = ceil((kernel - stride)/2), and
        output_padding = 2*padding - (kernel - stride) which is either 1 or 0.
    But to get the same result with Tensorflow, we should delete values by ourselves instead of using padding and
    output_padding in ConvTranspose2d.
    To get there, we check the following conditions:
    If pad = kernel - stride is even, we can directly set padding=pad/2 and output_padding=0 in ConvTranspose2d.
    If pad = kernel - stride is odd, we can use ConvTranspose2d to get T_nopad, and then delete `pad` rows/columns by
    ourselves; or we can use ConvTranspose2d to delete `pad - 1` by setting `padding=(pad - 1) / 2` and `ouput_padding=0`
    and then delete the last row/column of the resulting tensor by ourselves.
    Here we implement the former case.
    This module should be called after the ConvTranspose2d module with shared kernel_size and stride values.
    And this module can only output a tensor with shape `stride * size_input`.
    A more flexible module can be found in `yaleb.py` which can output arbitrary size as specified.
    """

    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
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
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module('conv%d' % i,
                                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2))
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module('deconv%d' % (i + 1),
                                    nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2))
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(kernels[i], 2))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y


class DSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample, rep_dim, num_cluster, device):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.rep_dim = rep_dim
        self.cluster_num = num_cluster
        self.ae1 = ConvAE(channels, kernels)
        self.ae2 = ConvAE(channels, kernels)
        self.cl1 = nn.Sequential(
            nn.Linear(self.rep_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.cluster_num),
            nn.Softmax(dim=1)
        )
        self.cl2 = nn.Sequential(
            nn.Linear(self.rep_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.cluster_num),
            nn.Softmax(dim=1)
        )
        self.cluster_loss = ClusterLoss(self.cluster_num, temperature=1.0, device=device)

    def forward(self, xi, xj):  # shape=[n, c, w, h]
        zi = self.ae1.encoder(xi)
        zj = self.ae1.encoder(xj)
        xi_recon = self.ae1.decoder(zi)  # shape=[n, c, w, h]
        xj_recon = self.ae1.decoder(zj)
        zi = zi.view(self.n, -1)
        zj = zj.view(self.n, -1)        # shape=[n, c, w, h]
        ci = self.cl1(zi)
        cj = self.cl2(zj)
        return xi_recon, xj_recon, zi, zj, ci, cj

    def loss_fn(self, xi, xj, xi_recon, xj_recon, ci, cj, con):
        loss_ae1 = F.mse_loss(xi_recon, xi, reduction='sum')
        loss_ae2 = F.mse_loss(xj_recon, xj, reduction='sum')
        cluster_loss = self.cluster_loss(ci, cj, con)
        return loss_ae1, loss_ae2, cluster_loss


def train(model,  # type: DSCNet
          x_w, x_s, y, epochs, lr=1e-3, device='cuda', show=1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    x_w = torch.tensor(x_w, dtype=torch.float32)
    x_s = torch.tensor(x_s, dtype=torch.float32)
    x_w = x_w.to(device)
    x_s = x_s.to(device)

    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    for epoch in range(epochs):
        xi_recon, xj_recon, zi, zj, ci, cj = model(x_w, x_s)
        loss1, loss2, closs = model.loss_fn(x_w, x_s, xi_recon, xj_recon, ci, cj, con=1)
        loss = loss1 + loss2 + closs
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % show == 0 or epoch == epochs - 1:
            print('Epoch %02d: loss1=%.4f loss2=%.4f closs=%.4f' %
                  (epoch, loss1.item(), loss2.item(), closs.item()))


if __name__ == "__main__":
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='mnist',
                        choices=['coil100', 'orl', 'usps', 'mnist'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--show-freq', default=10, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='pretrained_weights_original')
    parser.add_argument('--seed', default=3407, type=int)
    args = parser.parse_args()
    print(args)
    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    device = args.device
    db = args.db

    if db == 'coil100':
        # load data
        data = sio.loadmat('datasets/COIL100.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        subjects = 40
        y = y[:72 * subjects]

        data = sio.loadmat('datasets/coil100_xw_xs.mat')
        x_w, x_s = data['original'], data['augmentation']
        x_w = x_w[:subjects * 72]
        x_s = x_s[:subjects * 72]

        # train_transformer = Transforms()
        # x = torch.tensor(x, dtype=torch.float32)
        # x_w, x_s = train_transformer(x)
        # from scipy.io import savemat
        # x_save = x_w.numpy()
        # x_save = x_save.astype(np.float)
        # xs_save = x_s.numpy()
        # xs_save = xs_save.astype(np.float)
        # temp = {'original': x_save, 'augmentation': xs_save}
        # savemat('coil100_xw_xs.mat', temp)

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 50]
        kernels = [5]
        epochs = 2000
        fc_nhiddens = 20 * 16 * 16

    elif db == 'orl':
        # load data
        data = sio.loadmat('datasets/ORL_32x32.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        data = sio.loadmat('datasets/orl_xw_xs.mat')
        x_w, x_s = data['original'], data['augmentation']

        # train_transformer = Transforms()
        # x = torch.tensor(x, dtype=torch.float32)
        # x_w, x_s = train_transformer(x)
        # from scipy.io import savemat
        # x_save = x_w.numpy()
        # x_save = x_save.astype(np.float)
        # xs_save = x_s.numpy()
        # xs_save = xs_save.astype(np.float)
        # temp = {'original': x_save, 'augmentation': xs_save}
        # savemat('orl_xw_xs.mat', temp)

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 2000
        weight_coef = 2.0
        weight_selfExp = 0.2
        fc_nhiddens = 5 * 4 * 4

    elif db == 'mnist':
        data = sio.loadmat('datasets/MNIST-1000.mat')
        x, y = np.transpose(data['HRdata']).reshape((-1, 1, 28, 28)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        data = sio.loadmat('datasets/mnist_xw_xs.mat')
        x_w, x_s = data['original'], data['augmentation']

        # train_transformer = Transforms()
        # x = torch.tensor(x, dtype=torch.float32)
        # x_w, x_s = train_transformer(x)
        # from scipy.io import savemat
        # x_save = x_w.numpy()
        # x_save = x_save.astype(np.float)
        # xs_save = x_s.numpy()
        # xs_save = xs_save.astype(np.float)
        # temp = {'original': x_save, 'augmentation': xs_save}
        # savemat('mnist_xw_xs.mat', temp)

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 20, 10, 5]
        kernels = [5, 3, 3]
        epochs = 5000
        fc_nhiddens = 5 * 4 * 4

    elif db == 'umist':
        # load data
        data = sio.loadmat('datasets/UMIST.mat')
        x, y = np.transpose(data['fea']).reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        data = sio.loadmat('datasets/umist_xw_xs.mat')
        x_w, x_s = data['original'], data['augmentation']

        # train_transformer = Transforms()
        # x = torch.tensor(x, dtype=torch.float32)
        # x_w, x_s = train_transformer(x)
        # from scipy.io import savemat
        # x_save = x_w.numpy()
        # x_save = x_save.astype(np.float)
        # xs_save = x_s.numpy()
        # xs_save = xs_save.astype(np.float)
        # temp = {'original': x_save, 'augmentation': xs_save}
        # savemat('umist_xw_xs.mat', temp)

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 15, 10, 5]
        kernels = [5, 3, 3]
        epochs = 2000
        fc_nhiddens = 5 * 4 * 4

    num_class = len(np.unique(y))
    dscnet = DSCNet(num_sample=num_sample, channels=channels, kernels=kernels, rep_dim=fc_nhiddens, num_cluster=num_class, device=device)
    dscnet.cuda(device)

    train(dscnet, x_w, x_s, y, epochs=epochs, show=args.show_freq, device=device)
    state_dict = {
        'ae1': dscnet.ae1.state_dict(),
        'cl1': dscnet.cl1.state_dict(),
        'cl2': dscnet.cl2.state_dict()
    }
    torch.save(state_dict, args.save_dir + '/%s_2000.pth' % args.db)