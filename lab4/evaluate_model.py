from train_fixed_prior import pred
from utils import plot_pred, finn_eval_seq
from dataset import bair_robot_pushing_dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    print(f"PSNR: {ave_psnr}")

if __name__ == '__main__':
    print(f"Torch version: {torch.__version__}")
    main()
