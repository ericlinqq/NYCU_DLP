import torch
import torch.nn as nn
from models import cgan, wgan

def init_weight(m, mean=0, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean, std)
        nn.init.zeros_(m.bias.data)

def build_models(args):
    print("\nBuilding models...")
    print(f"GAN type: {args.gan_type}")

    if args.gan_type == "cgan":
        netG = cgan.Generator(args)
        netD = cgan.Discriminator(args)

        if args.train:
            netG.apply(init_weight)
            netD.apply(init_weight)
        elif args.test:
            print("Loading model checkpoints...")
            try:
                netG.load_state_dict(torch.load(f"{args.model_dir}/{args.exp_name}/Generator_{args.checkpoint_epoch}.pth"))
                netD.load_state_dict(torch.load(f"{args.model_dir}/{args.exp_name}/Discriminator_{args.checkpoint_epoch}.pth"))
            except:
                raise ValueError("Model checkpoints do not exist!")
    elif args.gan_type == "wgan":
        netG = wgan.Generator(args)
        netD = wgan.Discriminator(args)

        if args.train:
            netG.apply(init_weight)
            netD.apply(init_weight)
        elif args.test:
            print("Loading model checkpoints...")
            try:
                netG.load_state_dict(torch.load(f"{args.model_dir}/{args.exp_name}/Generator_{args.checkpoint_epoch}.pth"))
                netD.load_state_dict(torch.load(f"{args.model_dir}/{args.exp_name}/Discriminator_{args.checkpoint_epoch}.pth"))
            except:
                raise ValueError("Model checkpoints do not exist!")

    return (netG, netD)
