from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.utils import make_grid, save_image
import torch.autograd as autograd

from evaluator import evaluation_model

def build_trainer(args, device, models):
    print("\nBuilding trainer...")
    trainer = Trainer(args, device, models)
    return trainer

class Trainer:
    def __init__(self, args, device, models):
        self.args = args
        self.device = device

        if self.args.gan_type == "cgan" or self.args.gan_type == "wgan":
            self.netG = models[0].to(self.device)
            self.netD = models[1].to(self.device)
            
            self.optimG = optim.Adam(self.netG.parameters(), lr=self.args.lr_G, betas=(self.args.beta1, self.args.beta2))
            self.optimD = optim.Adam(self.netD.parameters(), lr=self.args.lr_D, betas=(self.args.beta1, self.args.beta2))
        
        self.evaluator = evaluation_model(self.device)
        self.log_file = f"{args.log_dir}/{args.exp_name}/log.txt"

    def train(self, train_loader, test_loader):
        if self.args.gan_type == "cgan":
            self.train_cgan(train_loader, test_loader)
        elif self.args.gan_type == "wgan":
            self.train_wgan(train_loader, test_loader)
    
    def compute_gp(self, real, fake, cond):
        """ Calculate the gradient penalty """
        # Random weights term for interpolation between real and fake samples
        alpha = torch.rand(real.shape[0], 1, 1, 1, device=self.device)

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        d_interpolates = self.netD(interpolates, cond)

        # Get gradients w.r.t interpolates
        gradients = autograd.grad(
            inputs=interpolates,
            outputs=d_interpolates,
            grad_outputs=torch.ones(d_interpolates.shape,  device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradients_penalty
    
    def train_cgan(self, train_loader, test_loader):
        best_score = 0
        criterion = nn.BCELoss()

        test_cond = next(iter(test_loader)).to(self.device)
        fixed_noise = torch.randn(test_cond.shape[0], self.args.z_dim, device=self.device)

        print(f"Start training {self.args.gan_type}...")

        progress = tqdm(total=self.args.epochs)
        for epoch in range(self.args.epochs):
            total_loss_G = 0
            total_loss_D = 0
            
            for i, (img, cond) in enumerate(train_loader):  
                self.netG.train()
                self.netD.train() 
                img = img.to(self.device)
                cond = cond.to(self.device)

                batch_len = img.shape[0]
                
                real_label = torch.ones(batch_len, device=self.device)
                fake_label = torch.zeros(batch_len, device=self.device)
                """
                train discriminator
                """
                self.optimD.zero_grad()
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
                    self.optimG.zero_grad()
                    noise = torch.randn(batch_len, self.args.z_dim, device=self.device)
                    fake = self.netG(noise, cond)
                    preds = self.netD(fake, cond)

                    loss_G = criterion(preds, real_label)
                    # backpropagation
                    loss_G.backward()
                    self.optimG.step()

                total_loss_G += loss_G.item()
                total_loss_D += loss_D.item()
            progress.update(1)
            
            # evaluate
            self.netG.eval()
            self.netD.eval()
            with torch.no_grad():
                pred_img = self.netG(fixed_noise, test_cond)
            score = self.evaluator.eval(pred_img, test_cond)

            if score > best_score:
                best_score = score
                torch.save(self.netG.state_dict(), f"{self.args.model_dir}/{self.args.exp_name}/Generator_{epoch}_{score:.2f}.pth")
                torch.save(self.netD.state_dict(), f"{self.args.model_dir}/{self.args.exp_name}/Discriminator_{epoch}_{score:.2f}.pth")
            
            with open(self.log_file, 'a') as train_record:
                train_record.write(f"[Epoch {epoch:3d}] avg_loss_G: {total_loss_G / len(train_loader):.4f} | avg_loss_D: {total_loss_D / len(train_loader):.4f} | Testing score: {score:.4f}\n")
            
            save_image(pred_img, f"{self.args.result_dir}/{self.args.exp_name}/pred_{epoch}.png", nrow=8, normalize=True)

    def train_wgan(self, train_loader, test_loader):
            best_score = 0
            

            test_cond = next(iter(test_loader)).to(self.device)
            fixed_noise = torch.randn(test_cond.shape[0], self.args.z_dim, device=self.device)

            print(f"Start training {self.args.gan_type}...")

            progress = tqdm(total=self.args.epochs)
            for epoch in range(self.args.epochs):
                total_loss_G = 0
                total_loss_D = 0
                
                for i, (img, cond) in enumerate(train_loader):  
                    self.netG.train()
                    self.netD.train() 
                    img = img.to(self.device)
                    cond = cond.to(self.device)

                    batch_len = img.shape[0]
                    
                    real_label = torch.ones(batch_len, device=self.device)
                    fake_label = torch.zeros(batch_len, device=self.device)
                    """
                    train discriminator
                    """
                    
                    for i in range(self.args.n_critic):
                        self.optimD.zero_grad()
                        
                        preds_real= self.netD(img, cond)

                        noise = torch.randn(batch_len, self.args.z_dim, device=self.device)
                        fake = self.netG(noise, cond)
                        preds_fake = self.netD(fake.detach(), cond)
                        
                        gp = self.compute_gp(img, fake, cond)

                        loss_D = -(torch.mean(preds_real) - torch.mean(preds_fake)) + self.args.lambda_gp * gp
                        loss_D.backward(retain_graph=True)
                        self.optimD.step()

                    """
                    train generator
                    """
                    self.optimG.zero_grad()
                    noise = torch.randn(batch_len, self.args.z_dim, device=self.device)
                    fake = self.netG(noise, cond)
                    preds_fake = self.netD(fake, cond)

                    loss_G = -torch.mean(preds_fake)
                    loss_G.backward()
                    self.optimG.step()

                    total_loss_G += loss_G.item()
                    total_loss_D += loss_D.item()
                progress.update(1)
                
                # evaluate
                self.netG.eval()
                self.netD.eval()
                with torch.no_grad():
                    pred_img = self.netG(fixed_noise, test_cond)
                score = self.evaluator.eval(pred_img, test_cond)

                if score > best_score:
                    best_score = score
                    torch.save(self.netG.state_dict(), f"{self.args.model_dir}/{self.args.exp_name}/Generator_{epoch}_{score:.2f}.pth")
                    torch.save(self.netD.state_dict(), f"{self.args.model_dir}/{self.args.exp_name}/Discriminator_{epoch}_{score:.2f}.pth")
                
                with open(self.log_file, 'a') as train_record:
                    train_record.write(f"[Epoch {epoch:3d}] avg_loss_G: {total_loss_G / len(train_loader):.4f} | avg_loss_D: {total_loss_D / len(train_loader):.4f} | Testing score: {score:.4f}\n")
                
                save_image(pred_img, f"{self.args.result_dir}/{self.args.exp_name}/pred_{epoch}.png", nrow=8, normalize=True)



    def test(self, test_loader):
        print("Start testing...")
        test_cond = next(iter(test_loader)).to(self.device)
        
        avg_score  = 0
        for _ in range(10):
            noise = torch.randn(test_cond.shape[0], self.args.z_dim, device=self.device)
            self.netG.eval()
            self.netD.eval()
            with torch.no_grad():
                pred_img = self.netG(noise, test_cond)
            score = self.evaluator.eval(pred_img, test_cond)
            print(f"Testing score: {score:.4f}")
            avg_score += score 
        
        save_image(pred_img, f"{self.args.result_dir}/{self.args.exp_name}/eval.png", nrow=8, normalize=True)
        print(f"avg score: {avg_score/10:.2f}")




