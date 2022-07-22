from ResNet import ResNet
from train import train
from torchvision import transforms
from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn  as nn
from torchvision.models import ResNet50_Weights, ResNet18_Weights
import pandas as pd
import matplotlib.pyplot as plt
from argparser import parse_argument

def main():
    data_transform = {
        'train':transforms.Compose([
            transforms.RandomResizedCrop(ResNet.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test':transforms.Compose([
            transforms.Resize(ResNet.input_size),
            transforms.CenterCrop(ResNet.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    print(f"Initializing Datasets and Dataloaders...")

    image_dataset = {x: RetinopathyLoader('data/', x, data_transform[x]) for x in ['train', 'test']}

    dataloaders_dict = {x: DataLoader(image_dataset[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_model = ResNet(num_classes=num_classes, net_type=net_type, weights=weights, feature_extract=feature_extract)
    random_init_model = ResNet(num_classes=num_classes, net_type=net_type, weights=None, feature_extract=False)

    df = pd.DataFrame()
    df['Epoch'] = range(1, num_epochs+1)

    for model in [pretrained_model, random_init_model]:
        print(f"ResNet{model.type}({model.mode})")
        model = model.to(device)

        params_to_update = model.parameters()
        print("Params to learn:")

        if model.feature_extract:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print(f"\t{name}")
        else:
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    print(f"\t{name}")

        optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)

        criterion = nn.CrossEntropyLoss()

        model, train_acc_history, test_acc_history = train(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, device=device)

        model_scripted = torch.jit.script(model)
        model_scripted.save(f'model/ResNet{model.net_type}/{model.mode}.pt')

        df[f'{model.mode}_train_acc'] = train_acc_history
        df[f'{model.mode}_test_acc'] = test_acc_history

    df.set_index('Epoch', inplace=True)
    df.to_csv(f'output/ResNet{model.net_type}/accuracy.csv')
    df.plot(title=f'Result Comparison (ResNet{model.net_type})', 
            xlabel='Epochs',
            ylabel='Accuracy(%)',
            legend=True,
            figsize=(10, 5)
    )
    plt.savefig(f"output/ResNet{model.net_type}/comparison.png")

if __name__ == '__main__':
    args = parse_argument()
    batch_size = args.b
    num_classes = args.c
    net_type = args.t
    feature_extract = args.f
    num_epochs = args.e
    lr = args.l
    momentum = args.m
    weight_decay = args.w

    if net_type == 18:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = ResNet50_Weights.DEFAULT

    main()