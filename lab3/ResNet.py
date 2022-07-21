import torch.nn as nn
from torchvision.models import resnet18, resnet50
import torch
import time
import copy

class ResNet(nn.Module):
    def __init__(self, num_classes=5, net_type=18, weights=None, feature_extract=False):
        super(ResNet, self).__init__()

        input_size = 224

        self.feature_extract = feature_extract
        
        if weights is not None:
            self.mode = 'pretrained'
        else:
            self.mode = 'random_init'

        self.net_type = net_type

        if net_type == 50:
            self.model = resnet50(weights=weights)
        else:
            self.model = resnet18(weights=weights)
        
        set_paramete_require_grad(self.model, self.feature_extract)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, num_classes)
    
    def forward(self, x):
        x = self.model(x)

        return x

def set_paramete_require_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.require_grad = False


def train(model, dataloaders, criterion, optimizer, num_epochs, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_acc_history = []
    test_acc_history = []

    for epoch in range(1, num_epochs+1):
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
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

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
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}")
    print(f"Best Test Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts) 

    return model, train_acc_history, test_acc_history