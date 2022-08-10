from train_fixed_prior import plot_pred, kl_annealing, parse_args
from utils import finn_eval_seq
from dataset import bair_robot_pushing_dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse

torch.backends.cudnn.benchmark = True

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device.")
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    saved_model = torch.load(f"{args.model_dir}/model.pth")

    frame_predictor = saved_model['frame_predictor'].to(device)
    posterior = saved_model['posterior'].to(device)
    encoder = saved_model['encoder'].to(device)
    decoder = saved_model['decoder'].to(device)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }

    frame_predictor.eval()
    posterior.eval()
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
