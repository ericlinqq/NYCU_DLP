from train_learned_prior import kl_annealing, parse_args, pred
from utils import finn_eval_seq, save_gif
from dataset import bair_robot_pushing_dataset
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import numpy as np
import argparse

torch.backends.cudnn.benchmark = True

def plot_pred(x, cond, modules, epoch, args, device):
    gt_seq = [x[i] for i in range(len(x))]
    pred_seq = pred(x, cond, modules, args, device)

    pred_plot = []
    gt_plot = []
    gif = [[] for t in range(args.n_eval)]

    for t in range(args.n_eval):
        pred_plot.append(pred_seq[t][5])
        gt_plot.append(gt_seq[t][5])
        gif[t].append(gt_seq[t][5])
        gif[t].append(pred_seq[t][5])

    fname = f'{args.log_dir}/gen_lp/sample_{epoch}'
    save_image(make_grid(pred_plot, nrow=args.n_eval), fname+'_pred.png')
    save_image(make_grid(gt_plot, nrow=args.n_eval), fname+'_gt.png')
    save_gif(fname+'.gif', gif, duration=0.25)

    return pred_seq

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device.")
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)

    saved_model = torch.load(f"{args.model_dir}/model.pth")

    frame_predictor = saved_model['frame_predictor'].to(device)
    posterior = saved_model['posterior'].to(device)
    prior = saved_model['prior'].to(device)
    encoder = saved_model['encoder'].to(device)
    decoder = saved_model['decoder'].to(device)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'prior': prior,
        'encoder': encoder,
        'decoder': decoder,
    }

    frame_predictor.eval()
    posterior.eval()
    prior.eval()
    encoder.eval()
    decoder.eval()

    psnr_list = []
    for test_seq, test_cond in test_loader:
        test_seq, test_cond = test_seq.transpose_(0, 1).to(device), test_cond.transpose_(0, 1).to(device)
        with torch.no_grad():
            pred_seq = plot_pred(test_seq, test_cond, modules, 'test', args, device)
        _, _, psnr = finn_eval_seq(test_seq[args.n_past: args.n_past + args.n_future], pred_seq[args.n_past: args.n_past + args.n_future])
        psnr_list.append(psnr)

    ave_psnr = np.mean(np.concatenate(psnr_list))
    print(f"PSNR: {ave_psnr:.3f}")

if __name__ == '__main__':
    print(f"Torch version: {torch.__version__}")
    main()
