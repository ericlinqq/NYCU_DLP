import torch.nn as nn
from torchvision.models import resnet18, resnet50

class ResNet(nn.Module):
    def __init__(self, num_classes=5, net_type=18, weights=None, feature_extract=False):
        super(ResNet, self).__init__()

        self.feature_extract = feature_extract
        
        if weights is not None:
            self.mode = 'with pretraining'
        else:
            self.mode = 'w/o pretraining'

        self.net_type = net_type

        if net_type == 50:
            self.model = resnet50(weights=weights)
        else:
            self.model = resnet18(weights=weights)
        
        set_parameter_require_grad(self.model, self.feature_extract)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, num_classes)
    
    def forward(self, x):
        x = self.model(x)

        return x

def set_parameter_require_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

def params_to_learn(model, feature_extracting):
    params_to_update = model.parameters()
    print("Params to learn:")

    if feature_extracting:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print(f"\t{name}")
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print(f"\t{name}")
    
    return params_to_update