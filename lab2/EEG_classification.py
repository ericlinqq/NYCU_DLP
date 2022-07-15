#%%
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader import read_bci_data
%matplotlib inline
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
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
    def __init__(self, activation=nn.ELU()):
        super(DeepConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1,5))
        channels = [25, 25, 50, 100, 200]
        kernel_size = [(2,1), (1,5), (1,5), (1,5)]

        for i in range(1, len(channels)):
            setattr(self, 'conv'+str(i), nn.Sequntial(
                nn.Conv2d(channels[i-1], channels[i], kernel_size[i-1]),
                nn.BatchNorm2d(channels[i], eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1,2)),
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
def train(train_dataloader, test_dataloader, activations, device):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lr=learning_rate)
    for activation in activations:
        model = DeepConvNet(activation)
        model.to(device)
        for i in range(1, epochs+1):
            for batch_idx, (data, label) in enumerate(train_dataloader):
                data = data.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.long)