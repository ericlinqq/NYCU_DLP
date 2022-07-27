from ResNet import ResNet, set_parameter_require_grad, params_to_learn
from torchvision import transforms
from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
import torchvision
import torch
import torch.optim as optim
import torch.nn  as nn
from torchvision.models import ResNet50_Weights, ResNet18_Weights
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
import copy
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

def train(model, dataloaders, criterion, optimizer, num_epochs, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_acc_history = list()
    test_acc_history = list()

    for epoch in range(1, num_epochs+1):

        if model.mode == "with pretraining" and epoch == num_ftr_extract + 1:
            model.feature_extract = False
            set_parameter_require_grad(model, model.feature_extract)
            params_to_update = params_to_learn(model, model.feature_extract)
            optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)

        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, pred = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = (running_corrects.double() / len(dataloaders[phase].dataset)).item()

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'test':
                test_acc_history.append(epoch_acc)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            else:
                train_acc_history.append(epoch_acc)
            
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Test Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts) 

    return model, train_acc_history, test_acc_history

def evaluate(model, test_dataloader, device, confusion_matrix=True):
    if confusion_matrix:
        cm = np.zeros([num_classes, num_classes])
    running_corrects = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)

            running_corrects += torch.sum(pred == labels.data)
            if confusion_matrix:
                for i in range(len(labels)):
                    cm[int(labels.data[i])][int(pred[i])] += 1

    accuracy = (running_corrects.double() / len(test_dataloader.dataset)).item()
    if confusion_matrix:
        cm /= cm.sum(axis=1).reshape(num_classes, 1)
        return accuracy, cm

    return accuracy

def main():
    data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    print(f"Initializing Datasets and Dataloaders...")

    image_dataset = {x: RetinopathyLoader('data/', x, data_transform) for x in ['train', 'test']}

    dataloaders_dict = {x: DataLoader(image_dataset[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}

    pretrained_model = ResNet(num_classes=num_classes, net_type=net_type, weights=weights, feature_extract=feature_extract)
    random_init_model = ResNet(num_classes=num_classes, net_type=net_type, weights=None, feature_extract=False)

    df = pd.DataFrame()
    df['Epoch'] = range(1, num_epochs+1)

    for model in [pretrained_model, random_init_model]:
        print(f"ResNet{model.type}({model.mode})")
        model = model.to(device)
        
        params_to_update = params_to_learn(model, feature_extract)
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model, train_acc_history, test_acc_history = train(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, device=device)
        _, cm = evaluate(model, dataloaders_dict['test'], device, True)

        model_scripted = torch.jit.script(model)
        
        name = model.mode.replace("/", "")
        
        model_scripted.save(f"model/ResNet{model.net_type}/model({name}).pt")

        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f"output/ResNet{model.net_type}/confusion_matrix({name}).png")
        plt.close()

        df[f'Train({model.mode})'] = train_acc_history
        df[f'Test({model.mode})'] = test_acc_history

    df.set_index('Epoch', inplace=True)
    df.to_csv(f'output/ResNet{model.net_type}/accuracy.csv')
    plt.figure()
    df.plot(title=f'Result Comparison (ResNet{model.net_type})', 
            xlabel='Epochs',
            ylabel='Accuracy(%)',
            legend=True,
            figsize=(10, 5)
    )
    plt.savefig(f"output/ResNet{model.net_type}/comparison.png")
    plt.close()

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=4, help='Batch size, default: 4')
    parser.add_argument('-c', type=int, default=5, help='Number of classes in dataset, default: 5')
    parser.add_argument('-t', type=check_network_type, default=18, help='ResNet18 (18) or ResNet50 (50), default: 18')
    parser.add_argument('-f', type=bool, default=True, help='Feature extract or not, default: True')
    parser.add_argument('-nf', type=int, default=5, help='How many epochs to train featrue extraction, then fine-tuning, default: 5')
    parser.add_argument('-e', type=int, default=5, help='Number of epochs, default: 5')
    parser.add_argument('-l', type=float, default=1e-3, help='Learning rate, default: 1e-3')
    parser.add_argument('-m', type=float, default=0.9, help='Momentum, default: 0.9')
    parser.add_argument('-w', type=float, default=5e-4, help='Weight decay, default: 5e-4')
    parser.add_argument('-wl', type=bool, default=False, help='Class-weighted loss, default: False')
    
    return parser.parse_args()

def check_network_type(input):
    int_value = int(input)
    
    if int_value != 18 and int_value != 50:
        raise argparse.ArgumentTypeError(f"Network type should be 18 or 50")
    
    return int_value

if __name__ == '__main__':

    print(f"Pytorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    args = parse_argument()
    batch_size = args.b
    num_classes = args.c
    net_type = args.t
    feature_extract = args.f
    num_ftr_extract = args.nf
    num_epochs = args.e
    lr = args.l
    momentum = args.m
    weight_decay = args.w
    weighted_loss = args.wl

    if net_type == 18:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = ResNet50_Weights.DEFAULT

    if weighted_loss:
        class_weights = torch.tensor([0.2649, 0.9304, 0.8502, 0.9752, 0.9793])
        class_weights = class_weights.to(device)
    else:
        class_weights = None


    main()