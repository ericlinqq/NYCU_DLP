from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.utils import make_grid, save_image

from evaluator import evaluation_model

def build_trainer(args, device, models):
    print("\nBuilding trainer...")
    trainer = Trainer(args, device, models)
    return trainer

class Trainer:
    def __init__(self, args, device, models):
        self.args = args
        self.device = device

        if self.args.gan_type == "cgan":
            self.netG = models[0].to(self.device)
            self.netD = models[1].to(self.device)
            
            self.optimG = optim.Adam(self.netG.parameters(), lr=self.args.lr_G, betas=(self.args.beta1, self.args.beta2))
            self.optimD = optim.Adam(self.netD.parameters(), lr=self.args.lr_D, betas=(self.args.beta1, self.args.beta2))
        
        self.evaluator = evaluation_model(self.device)
        self.log_file = f"{args.log_dir}/{args.exp_name}/log.txt"
        self.log_writer = open(self.log_file, "w")

    def train(self, train_loader, test_loader):
        if self.args.gan_type == "cgan":
            self.train_cgan(train_loader, test_loader)
    
    def train_cgan(self, train_loader, test_loader):
        best_score = 0
        criterion = nn.BCELoss()

        test_cond = next(iter(test_loader)).to(self.device)
        fixed_noise = torch.randn(test_cond.shape[0], self.args.z_dim, device=self.device)

        print(f"Start training {self.args.gan_type}...")

        for epoch in range(self.args.epochs):
            total_loss_G = 0
            total_loss_D = 0
            self.netG.train()
            self.netD.train()
            for step, (img, cond) in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch:3d}")):   
                img = img.to(self.device)
                cond = cond.to(self.device)

                batch_len = img.shape[0]
                
                real_label = torch.ones(batch_len, device=self.device)
                fake_label = torch.zeros(batch_len, device=self.device)
                """
                train discriminator
                """
                self.netD.zero_grad()
                # for real images
                preds = self.netD(img, cond)
                loss_D_real = criterion(preds, real_label)
                # for fake images
                noise = torch.randn(batch_len, self.args.z_dim, device=self.device)
                fake = self.netG(noise, cond)
                preds = self.netD(fake.detach(), cond)
                loss_D_fake = criterion(preds, fake_label)

                loss_D = loss_D_real + loss_D_fake
                # backpropagation
                loss_D.backward()
                self.optimD.step()

                """
                train generator
                """
                for _ in range(4):
                    self.netG.zero_grad()
                    noise = torch.randn(batch_len, self.args.z_dim, device=self.device)
                    fake = self.netG(noise, cond)
                    preds = self.netD(fake, cond)

                    loss_G = criterion(preds, real_label)
                    # backpropagation
                    loss_G.backward()
                    self.optimG.step()

                total_loss_G += loss_G.item()
                total_loss_D += loss_D.item()
                print(f"[Epoch {epoch:3d}] Loss D: {loss_D.item():.4f} Loss G: {loss_G.item():.4f}")
            
            # evaluate
            self.netG.eval()
            self.netD.eval()
            with torch.no_grad():
                pred_img = self.netG(fixed_noise, test_cond)
            score = self.evaluator.eval(pred_img, test_cond)

            if score > best_score:
                best_score = score
                print(f"[Epoch {epoch:3d}] Saving model checkpoints with best score...")
                torch.save(self.netG.state_dict(), f"{self.args.model_dir}/{self.args.exp_name}/Generator_{epoch}.pth")
                torch.save(self.netD.state_dict(), f"{self.args.model_dir}/{self.args.exp_name}/Discriminator_{epoch}.pth")
            
            print(f"[Epoch {epoch:3d}] avg_loss_G: {total_loss_G / len(train_loader)} avg_loss_D: {total_loss_D / len(train_loader)}")
            print(f"Testing score: {score:.4f}") 
            print("-"*10)
            
            save_image(pred_img, f"{self.args.result_dir}/{self.args.exp_name}/pred_{epoch}.png", nrow=8, normalize=True)




    def test(self, test_loader):
        print("Start testing...")
        test_cond = next(iter(test_loader)).to(self.device)
        fixed_noise = torch.randn(test_cond.shape[0], self.args.z_dim, device=self.device)
        
        avg_score  = 0
        for _ in range(10):
            self.netG.eval()
            self.netD.eval()
            with torch.no_grad():
                pred_img = self.netG(fixed_noise, test_cond)
            score = self.evaluator.eval(pred_img, test_cond)
            print(f"Testing score: {score:.4f}")
            avg_score += score 
        
        save_image(pred_img, f"{self.args.result_dir}/{self.args.exp_name}/eval.png", nrow=8, normalize=True)
        print(f"avg score: {avg_score/10:.2f}")




