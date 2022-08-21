import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args

        self.cond_layer = nn.Sequential(
            nn.Linear(24, self.args.c_dim),
            nn.ReLU()
        )

        channels = [self.args.z_dim+self.args.c_dim, self.args.input_dim*8, self.args.input_dim*4, self.args.input_dim*2, self.args.input_dim]
        paddings = [0, 1, 1, 1]
        layer_list = []
        
        for i in range(1, len(channels)):
            layer_list.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels[i-1], channels[i], kernel_size=4, stride=2, padding=paddings[i-1], bias=False),
                    nn.BatchNorm2d(channels[i]),
                    nn.ReLU(True)
                )
            )
        layer_list.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.args.input_dim, self.args.n_channel, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()
            )
        )

        self.layer = nn.Sequential(*layer_list)

    def forward(self, z, c):
        z = z.view(-1, self.args.z_dim, 1, 1)
        c = self.cond_layer(c).view(-1, self.args.c_dim, 1, 1)
        out = torch.cat([z, c], dim=1)
        out = self.layer(out)

        return out

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args

        self.cond_layer = nn.Sequential(
            nn.Linear(24, self.args.input_dim*self.args.input_dim),
            nn.LeakyReLU()
        )

        channels = [self.args.n_channel+1, self.args.input_dim, self.args.input_dim*2, self.args.input_dim*4, self.args.input_dim*8]
        size = [int(self.args.input_dim/2), int(self.args.input_dim/4), int(self.args.input_dim/8), int(self.args.input_dim/16)]
        layer_list = []

        for i in range(1, len(channels)):
            layer_list.append(
                nn.Sequential(
                    nn.Conv2d(channels[i-1], channels[i], kernel_size=4, stride=2, padding=1, bias=False),
                    nn.LayerNorm([channels[i], size[i-1], size[i-1]]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        layer_list.append(
            nn.Sequential(
                nn.Conv2d(self.args.input_dim*8, 1, kernel_size=4, stride=1, bias=False)
            )
        )

        self.layer = nn.Sequential(*layer_list)
    
    def forward(self, X, c):
        c = self.cond_layer(c).view(-1, 1, self.args.input_dim, self.args.input_dim)
        out = torch.cat([X, c], dim=1)
        out = self.layer(out).view(-1)

        return out
