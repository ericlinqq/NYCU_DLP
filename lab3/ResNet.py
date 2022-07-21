import torch.nn as nn
from torchvision.models import resnet18, resnet50

class ResNet(nn.Module):
    def __init__(self, num_classes=5, type=0, weights=None, feature_extract=True):
        super(ResNet, self).__init__()
        
        if type:
            self.model = resnet50(weights=weights)
        else:
            self.model = resnet18(weights=weights)
        
        set_paramete_require_grad(self.model, feature_extract)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        x = self.model(x)

        return x

def set_paramete_require_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.require_grad = False
