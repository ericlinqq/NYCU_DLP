import numpy as np
import matplotlib.pyplot as plt
import torch

path = './logs/fp/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000/'

kld = []
with open(path+'train_record.txt') as train_record:
    for line in train_record.readlines():
        s = line.split(' | ')
        s = s[2].split(': ')
        kld.append(s[1])

saved_model = torch.load(path+'model.pth')
plot_record = saved_model['plot_record']

def plot_curve(rec_dict, kld, epoch, path, step = 5):
    epochs = np.arange(epoch)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(epochs, rec_dict["tfr"], '--b', label = "tfr")
    ax1.plot(epochs, rec_dict["KL_weight"], '--g', label = "KL weight")
    ax1a = ax1.twinx()
    ax1a.plot(epochs, kld, 'y', label = "KL loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("score/weight")
    ax1a.set_ylabel("loss")
    ax1.legend()
    ax1a.legend()

    epochs2 = np.arange(0, epoch, step)
    ax2.plot(epochs, rec_dict["tfr"], '--b', label = "tfr")
    ax2.plot(epochs, rec_dict["KL_weight"], '--g', label = "KL weight")
    ax2a = ax2.twinx()
    ax2a.plot(epochs2, rec_dict["psnr"], ':r', label = "psnr")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score/weight")
    ax2a.set_ylabel("psnr score")
    ax2.legend()
    ax2a.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.show()