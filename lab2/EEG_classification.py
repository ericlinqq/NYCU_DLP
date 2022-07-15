#%%
from tkinter import W
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader import read_bci_data
%matplotlib inline

#%%
batch_size = 64
learning_rate = 1e-2
epochs = 300

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")
x_train, y_train, x_test, y_test = read_bci_data()

#%%
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#%%
TrainDataset = MyDataset(x_train, y_train)
TestDataset = MyDataset(x_test, y_test)

train_dataloader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(TestDataset, batch_size=batch_size, shuffle=True, num_workers=2)

#%%
class DeepConvNet(nn.Module):
    """
    Input: (B, 2, 750)
    Reshape: (B, 1, 2, 750)
    Conv2D: (B, 25, 2, 746)
    Conv2D: (B, 25, 1, 746)
    MaxPool2D: (B, 25, 1, 373)
    Conv2D: (B, 50, 1, 369)
    MaxPool2D: (B, 50, 1, 184)
    Conv2D: (B, 100, 1, 180)
    MaxPool2D: (B, 100, 1, 90)
    Conv2D: (B, 100, 1, 86)
    MaxPool2D: (B, 200, 1, 43)
    Flatten: (B, 8600)
    Dense: (B, 2)
    """
    def __init__(self, activation):
        super(DeepConvNet, self).__init__()
        self.conv0 = nn.Conv2d(1, 25, kernel_size=(1,5))
        channels = [25, 25, 50, 100, 200]
        kernel_size = [(2,1), (1,5), (1,5), (1,5)]

        for i in range(1, len(channels)):
            setattr(self, 'conv'+str(i), nn.Sequential(
                nn.Conv2d(channels[i-1], channels[i], kernel_size[i-1]),
                nn.BatchNorm2d(channels[i], eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5)
                )
            )
            
        self.dense = nn.Linear(8600, 2)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)

        return x

#%%
class EEGNet(nn.modules):
    def __init__(self, activation):
        super(EEGNet, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4),padding=0),
            nn.Dropout(p=0.25)
        )

        self.seperableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )

        self.classify = nn.Linear(in_features=736, out_features=2, bias=True)
    
    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.seperableConv(x)
        x = self.classify(x)

        return x

#%%
def train(train_dataloader, test_dataloader, Network, activations, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    accuracy = 0
    for activation in activations:
        model = Network(activation)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for i in range(1, epochs+1):
            # Train
            model.train()
            for batch_idx, (data, label) in enumerate(train_dataloader):
                data = data.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.long)
                output = model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                
# %%
