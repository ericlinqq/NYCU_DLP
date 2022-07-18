import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader import read_bci_data
import argparse

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

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
    Linear: (B, 2)
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

class EEGNet(nn.Module):
    """
    Input: (B, 2, 750)
    Reshape: (B, 1, 2, 750)
    Conv2d: (B, 16, 2, 750)
    Conv2d: (B, 32, 1, 750)
    AvgPool2d: (B, 32, 1, 187)
    Conv2d: (B, 32, 1, 187)
    AvgPool2d: (B, 32, 1, 23)
    Flatten: (B, 736)
    Linear: (B, 2)
    """
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
        x = x.view(x.shape[0], -1)
        x = self.classify(x)

        return x
    
def run_model(train_dataloader, test_dataloader, activation_dict, device):
    df = pd.DataFrame()
    df['Epoch'] = range(1, epochs+1)
    criterion = nn.CrossEntropyLoss()

    for name, activation in activation_dict.items():
        print(f"Activation function: {name}\n")
        model = Network(activation)
        print(f"{model.__class__.__name__}\n")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        acc_train = []
        acc_test = []
        best_weight = dict.fromkeys(activation_dict, None)
        best_accuracy = dict.fromkeys(activation_dict, 0)

        for epoch in range(1, epochs+1):
            # Train
            model.train()
            total_loss = 0
            accuracy = 0

            for batch_idx, (input, label) in enumerate(train_dataloader): 
                # training data and label
                input = input.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.long)
                
                # Zero the gradients for every batch
                optimizer.zero_grad()

                # Predict for this batch
                output = model(input)

                # Compute loss and its gradients
                loss = criterion(output, label)
                loss.backward()
                
                # Update weights
                optimizer.step()

                total_loss += loss.item()
                accuracy += output.max(dim=1)[1].eq(label).sum().item()

            total_loss /= len(train_dataloader.dataset)
            accuracy = 100. * accuracy / len(train_dataloader.dataset)
            acc_train.append(accuracy)
            if epoch % print_interval == 0:
                print(f"Epoch: {epoch} Loss: {total_loss} Accuracy: {accuracy}\n")

            # Test
            model.eval()
            with torch.no_grad():
                accuracy = 0
                for batch_idx, (input, label) in enumerate(test_dataloader):
                    input = input.to(device, dtype=torch.float)
                    label = label.to(device, dtype=torch.long)
                    output = model(input)
                    accuracy += output.max(dim=1)[1].eq(label).sum().item()
        
                accuracy = 100. * accuracy / len(test_dataloader.dataset)
                acc_test.append(accuracy)

            if accuracy > best_accuracy[name]:
                best_accuracy[name] = accuracy
                best_weight[name] = model.state_dict()

        df[name+'_train'] = acc_train
        df[name+'_test'] = acc_test 

    return df, model, best_weight, best_accuracy

def check_network_type(input):
    int_value = int(input)
    
    if int_value != 0 and int_value != 1:
        raise argparse.ArgumentTypeError(f"Network type should be 0 or 1")
    
    return int_value

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=64, help='Batch size')
    parser.add_argument('-l', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('-e', type=int, default=300, help='Number of epoch') 
    parser.add_argument('-i', type=int, default=10, help='Print every i epochs')
    parser.add_argument('-n', type=check_network_type, default=0, help='Netweok type, 0: DeepConvNet, 1: EEGNet')
    parser.add_argument('-w', type=float, default=0, help='optimizer weight decay')
    return parser.parse_args()

def plot(df, model):
    df.plot(title=f'Activation function comparison. ({model.__class__.__name__})', 
            xlabel='Epoch',
            ylabel='Accuracy(%)',
            legend=True,
            figsize=(10, 5)
    )

    plt.savefig(f"output/{model.__class__.__name__}/comparison_lr.png")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device\n")

    x_train, y_train, x_test, y_test = read_bci_data()
    TrainDataset = MyDataset(x_train, y_train)
    TestDataset = MyDataset(x_test, y_test)
    train_dataloader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(TestDataset, batch_size=batch_size, shuffle=False, num_workers=2)

    activation_dict = {'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'ELU': nn.ELU()}

    df, model, best_weight, best_accuracy = run_model(train_dataloader, test_dataloader, activation_dict, device)
    for name, weight in best_weight.items():
        torch.save(weight, f"output/{model.__class__.__name__}/model/{name}.pt")
    
    for name, accuracy in best_accuracy.items():
        print(f"{name}: {accuracy}")

    df.set_index('Epoch', inplace=True)

    df.to_csv(f'output/{model.__class__.__name__}/accuracy_lr.csv')
    plot(df, model)

if __name__ == '__main__':
    args = parse_argument()
    batch_size = args.b
    learning_rate = args.l
    epochs = args.e
    print_interval = args.i
    Network_type = args.n
    weight_decay = args.w

    if not Network_type:
        Network = DeepConvNet
    else:
        Network = EEGNet

    main()