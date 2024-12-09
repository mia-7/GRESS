import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random
import torch
import torch.optim as optim
import numpy as np
from post_clustering2 import spectral_clustering, acc, err_rate
import scipy.io as sio

def train(model, x_w, x_s, y, epochs, theta, k,  lr=1e-3, weight_coef=1.0, weight_selfExp=150, weight_con=1, lambda1=1, lambda2=1, neg=1, lambda4=1,
          alpha=0.04, dim_subspace=12, ro=8, show=10, device='cuda', noise_level=0.):

    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_w = x_w.to(device)
    x_s = x_s.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))

    accw_max = 0.
    acc_c = 0.

    for epoch in range(epochs):
        xi_recon, xj_recon, zi, zj, zi_recon, zj_recon, ci, cj, si, sj = model(x_w, x_s)
        loss, loss_dsc, loss_cluster, loss_neg = \
            model.loss_fn(x_w, x_s, xi_recon, xj_recon, zi, zj, zi_recon, zj_recon, ci, cj, weight_coef, weight_selfExp, weight_con,
                          lambda1, lambda2, neg, lambda4, k, theta)
        print('Epoch %02d: loss=%.4f, loss_dsc=%.4f, loss_cluster=%.4f, loss_neg=%.4f' %
              (epoch, loss.item(), loss_dsc.item(), loss_cluster.item(),loss_neg.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % show == 0 or epoch == epochs - 1:
            C_w = model.self_expression1.Coefficient.detach().to('cpu').numpy()
            y_predw = spectral_clustering(C_w, K, dim_subspace, alpha, ro)
            yc_pred = torch.argmax(ci, 1)
            values, _ = torch.max(ci, 1)
            err_t, nmi_t, _ = err_rate(y, y_predw)
            acc_t = 1-err_t
            if acc_t > accw_max:
                state_dict = {
                    'ae1': model.ae1.state_dict(),
                    'self_expression1': model.self_expression1.state_dict(),
                    'cl1': model.cl1.state_dict(),
                    'cl2': model.cl2.state_dict(),
                }
                accw_max = acc_t
                nmiw_max = nmi_t
            yc_pred = yc_pred.detach().cpu().numpy()
            acc_c = acc(y, yc_pred)
            print('subspace clustering acc: %.4f while cluster layer acc: %.4f' % (acc_t, acc(y, yc_pred)))
    reg = [lambda2, lambda4]
    print("this_max:", reg, accw_max, nmiw_max)
    para = [accw_max, nmiw_max]
    return para, acc_c, state_dict


if __name__ == "__main__":
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='mnist')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--show-freq', default=1, type=int)
    parser.add_argument('--lr', default=5e-5, type=float) #
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--seed', default=3407, type=int)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    device = args.device
    db = args.db

    if db == 'mnist':
        data = sio.loadmat('datasets/mnist.mat')
        x, y = data['x'].reshape((-1, 1, 28, 28)), data['y']
        y = np.reshape(y, (-1))
        data = sio.loadmat('datasets/mnist_xw_xs.mat')
        x_w, x_s = data['original'], data['augmentation']

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 20, 10, 5]
        kernels = [5, 3, 3]
        epochs = 300
        lr = 5e-5

        alpha = 0.12  # threshold of C
        dim_subspace = 4  # dimension of each subspace
        ro = 5  #
        num_class = len(np.unique(y))

        weight_coef = 1000 #100
        weight_selfExp = 0.001 #0.1
        weight_con = 1
        fc_nhiddens = 5 * 4 * 4
        lambda1, lambda3 = 0.0001, 0.001
        k, theta = 14, 0.6

        from utils import CovAE_mnist as backbone

        pretrain_path = 'pretrained_weights_original/mnist_cl_2000.pth'
        save_path = './save/MNIST'
    elif db == 'orl':
        data = sio.loadmat('datasets/ORL_32x32.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        num_class = len(np.unique(y))
        data = sio.loadmat('datasets/orl_xw_xs.mat')
        x_w, x_s = data['original'].astype(np.float32) / 255, data['augmentation'].astype(np.float32) / 255

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 3000  # 700
        weight_coef = 10  # 0.01
        weight_selfExp = 1000  # 100
        weight_con = 100  # 0.1
        fc_nhiddens = 5 * 4 * 4
        lr = 3e-4

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #

        k, theta = 6, 0.4
        lambda1, lambda2, lambda3, lambda4 = 0.005, 0.005, 5e-06, 5e-06
        from utils import CovAE_orl as backbone

        pretrain_path = 'pretrained_weights_original/orl_cl_2000.pth'
        save_path = './save/ORL'

    elif db == 'coil40':
        data = sio.loadmat('datasets/COIL100.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        subjects = 40
        y = y[:72 * subjects]
        num_class = len(np.unique(y))

        data = sio.loadmat('datasets/coil100_xw_xs.mat')
        x_w, x_s = data['original'].astype(np.float32) / 255.0, data['augmentation'].astype(np.float32) / 255.0
        x_w = x_w[:subjects * 72]
        x_s = x_s[:subjects * 72]

        # network and optimization parameters
        num_sample = x_w.shape[0]
        channels = [1, 20]
        kernels = [3]
        epochs = 300
        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 16  # dimension of each subspace
        ro = 6  #
        fc_nhiddens = 20 * 16 * 16
        lr = 3e-4

        weight_coef = 0.1  # 0.01#1
        weight_selfExp = 100  # 1000#150
        weight_con = 1000  # 0.1#1

        lambda1, lambda2, lambda3, lambda4 = 0.0001, 0.0001, 0.005, 0.005
        k=15
        theta=0.7
        from utils import CovAE_coil40 as backbone

        pretrain_path = 'pretrained_weights_original/coil100_cl_2000.pth'
        save_path = './save/COIL40'

    elif db == 'umist':
        data = sio.loadmat('datasets/umist-32-32.mat')
        Img = data['img']
        Label = data['label']
        n_input = [32, 32]
        x = Img.reshape((-1, 1, 32, 32))
        y = np.squeeze(Label)  # y in [0, 1, ..., K-1]
        data = sio.loadmat('datasets/umist_xw_xs.mat')
        x_w, x_s = data['original'].astype(np.float32) / 255, data['augmentation'].astype(np.float32) / 255
        # network and optimization parameters
        num_sample = x_w.shape[0]  # 20 classes 480 images
        channels = [1, 15, 10, 5]
        kernels = [5, 3, 3]
        epochs = 400
        lr = 3e-4

        alpha = 0.08  # threshold of C
        dim_subspace = 5  # dimension of each subspace
        ro = 8  #

        num_class = len(np.unique(y))

        # network and optimization parameters
        weight_coef = 1e-2
        weight_selfExp = 1
        weight_con = 1e3
        fc_nhiddens = 5 * 4 * 4

        k, theta = 9, 0.2
        lambda1, lambda2, lambda3, lambda4 = 5e-06, 5e-06, 5e-05, 5e-05
        from utils import CovAE_umist as backbone

        pretrain_path = 'pretrained_weights_original/umist_cl_2000.pth'
        save_path = './save/UMIST'

    else:
        print('No matching dataset')
        exit()

    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    x_w = torch.tensor(x_w, dtype=torch.float32)
    x_s = torch.tensor(x_s, dtype=torch.float32)

    mm, mmc =0, 0

    for iterloop in range(1):
        dscnet = backbone.DSCNet(channels=channels, kernels=kernels, num_sample=num_sample, dim=fc_nhiddens, num_classes=num_class, temperature=1.0,
                                  device=args.device)
        dscnet.cuda(device)

        ae_state_dict = torch.load(pretrain_path)
        dscnet.ae1.load_state_dict(ae_state_dict['ae1'])
        dscnet.cl1.load_state_dict(ae_state_dict['cl1'])
        dscnet.cl2.load_state_dict(ae_state_dict['cl2'])
        print("Pretrained ae weights are loaded successfully.")

        para, acc_m, state_dict = train(dscnet, x_w, x_s, y, epochs, theta, k, lr=lr,
                                  weight_coef=weight_coef, weight_selfExp=weight_selfExp,
                                  weight_con=weight_con, lambda1=lambda1, lambda2=lambda1, neg=lambda3, lambda4=lambda3,
                                  alpha=alpha, dim_subspace=dim_subspace, ro=ro, show=args.show_freq,
                                  device=device)
        torch.save(state_dict, save_path + '/%s_GEAR2.pth' % (args.db))



