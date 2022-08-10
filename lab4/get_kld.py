import numpy as np
import matplotlib.pyplot as plt
from train_fixed_prior import parse_args, kl_annealing

def plot_curve(tfr, kl_weight, kld, psnr, epoch, path, step = 5):
    epochs = np.arange(epoch)
    fig, ax = plt.subplots(1, 2)
    fig.set_figheight(8)
    fig.set_figwidth(20)
    ax[0].plot(epochs, tfr, linestyle='--', color='blue', label = "tfr")
    ax[0].plot(epochs, kl_weight, linestyle='--', color='green', label = "KL weight")
    ax1 = ax[0].twinx()
    ax1.plot(epochs, kld, 'y', label = "KL loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("score/weight")
    ax1.set_ylabel("loss")
    ax[0].legend()
    ax1.legend()

    epochs2 = np.arange(0, epoch, step)
    ax[1].plot(epochs, tfr, linestyle='--', color='blue', label = "tfr")
    ax[1].plot(epochs, kl_weight, linestyle='--', color='green', label = "KL weight")
    ax2 = ax[1].twinx()
    ax2.plot(epochs2, psnr, 'o', color='red', label = "psnr")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("score/weight")
    ax2.set_ylabel("psnr score")
    ax[1].legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(path)

if __name__ == '__main__':
    args = parse_args()
    path = './logs/fp/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000/'

    kl_anneal = kl_annealing(args)
    kld = []
    psnr = []
    tfr = []
    args.tfr_decay_step = 1. / (args.niter - args.tfr_start_decay_epoch - 1)
    for epoch in range(args.niter):
        if epoch >= args.tfr_start_decay_epoch:
                ### Update teacher forcing ratio ###
                args.tfr -= args.tfr_decay_step
                if args.tfr < args.tfr_lower_bound:
                    args.tfr = args.tfr_lower_bound
        tfr.append(args.tfr)

    with open(path+'train_record.txt') as train_record:
        for line in train_record.readlines():
            if line[0] == '[':
                s = line.split(' | ')
                s = s[2].split(': ')
                kld.append(float(s[1]))
            if line[0] == '=':
                s = line.split(' ')
                psnr.append(float(s[4]))
    
    plot_curve(tfr, kl_anneal.L, kld, psnr, args.niter, path+'curve.png')