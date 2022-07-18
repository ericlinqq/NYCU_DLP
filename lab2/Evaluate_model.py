from dataloader import read_bci_data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from EEG_classification import MyDataset, EEGNet, DeepConvNet, test

def main():
    _, _, x_test, y_test = read_bci_data()
    test_dataset = MyDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=540, shuffle=False, num_workers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device\n")

    model = EEGNet(nn.ReLU())
    model.load_state_dict(torch.load('output/EEGNet/model/ReLU.pt', map_location=device))
    model.to(device)
    print(model)

    return test(test_dataloader, model, device)

if __name__ == '__main__':
    accuracy = main() 
    print(accuracy)