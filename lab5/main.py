import os
import shutil
import random
import argparse
import torch
from datahelper.dataloader import load_test_data, load_train_data
from models.build_models import build_models
from train import build_trainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--gan_type", default="cgan")
    parser.add_argument("--seed", default=1, type=int, help="manual seed")
    parser.add_argument("--lr_G", default=2e-4, type=float, help="learning rate for generator")
    parser.add_argument("--lr_D", default=2e-4, type=float, help="learning rate for discriminator")
    parser.add_argument("--batch_size", default=128, type=int, help="batch_size")
    parser.add_argument("--optimizer", default="adam", help="optimizer to train with")
    parser.add_argument("--beta1", default=0.5, type=float, help="beta1 for adam optimizer")
    parser.add_argument("--beta2", default=0.99, type=float, help="beta2 for adam optimizer")
    parser.add_argument("--epochs", default=300, type=int, help="number of epochs to train for")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--n_eval", default=10, help="number of iterations (fixed noise) to evaluate the model")
    
    parser.add_argument("--input_dim", default=64, type=int, help="dimension of input image")
    parser.add_argument("--z_dim", default=100, type=int, help="dimension of latent vector z")
    parser.add_argument("--c_dim", default=200, type=int, help="dimension of condition vector")
    parser.add_argument("--n_channel", default=3, type=int, help="number of channels of input image")

    parser.add_argument("--n_critic", default=5, type=int, help="number of iterations of the critic per generator iteration")
    parser.add_argument("--lambda_gp", default=10, type=int, help="factor of gradient penalty term")

    parser.add_argument("--report_freq", default=50, type=int, help="unit: steps (iterations), frequency to print loss values on termial")
    parser.add_argument("--save_img_freq", default=1, type=int, help="unit: epochs, frequency to save output images from generator")
    parser.add_argument("--checkpoint_epoch", default=None, type=str, help="the epoch of checkpoint you want to load")

    parser.add_argument("--model_dir", default="/mnt/d/DLP/lab5/checkpoints", help="base directory to save your model checkpoints")
    parser.add_argument("--data_root", default="/mnt/d/DLP/lab5/dataset", help="root directory for data")
    parser.add_argument("--result_dir", default="/mnt/d/DLP/lab5/result", help="base directory to save predicted images")
    parser.add_argument("--log_dir", default="/mnt/d/DLP/lab5/log", help="base directory to save training logs")
    parser.add_argument("--exp_name", default="cgan")
    parser.add_argument("--test_file", default="test.json")

    args = parser.parse_args()

    return args

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if args.train:
        if os.path.isdir(f"{args.model_dir}/{args.exp_name}"):
            shutil.rmtree(f"{args.result_dir}/{args.exp_name}")
        if os.path.isdir(f"{args.result_dir}/{args.exp_name}"):
            shutil.rmtree(f"{args.log_dir}/{args.exp_name}")
        if os.path.isdir(f"{args.log_dir}/{args.exp_name}"):
            shutil.rmtree(f"{args.log_dir}/{args.exp_name}")
        os.makedirs(f"{args.model_dir}/{args.exp_name}", exist_ok=True)
        os.makedirs(f"{args.result_dir}/{args.exp_name}", exist_ok=True)
        os.makedirs(f"{args.log_dir}/{args.exp_name}", exist_ok=True)
    elif args.test:
        if not os.path.isdir(f"{args.model_dir}/{args.exp_name}"):
            raise ValueError("Model checkpoints director does not exist!")

    
    if args.train:
        train_loader, test_loader = load_train_data(args)
    elif args.test:
        test_loader = load_test_data(args)
    else:
        raise ValueError("Mode [train/test] not determined!")
    

    models = build_models(args)

    trainer = build_trainer(args, device, models)

    if args.train:
        trainer.train(train_loader, test_loader)
    elif args.test:
        trainer.test(test_loader)

if __name__ == '__main__':
    args = parse_args()

    main(args) 