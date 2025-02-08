import torch
import torch.nn as nn
import torch.functional as F

NORM_VALUES = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'tinyimagenet': [[0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]],
}

class ModelWrapper:
    """
    Implements a wrapper around a model to provide the feature_forward method as well as 
    normalising inputs before passing them to the base model.
    """
    def __init__(self, model, w_features=True, id_dataset='cifar10'):
        self.model = model
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = torch.tensor(NORM_VALUES[id_dataset][0]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(NORM_VALUES[id_dataset][1]).view(1, 3, 1, 1).to(self.device)
        self.normalise = True
        if w_features:
            # currently only supports getting penultimate layer features with ResNet18 models
            self.conv1 = model.conv1
            self.bn1 = model.bn1
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4
            self.linear = model.linear

    def feature_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out.view(out.size(0), -1)
    
    def __call__(self, x, return_feature=False):
        if self.normalise:
            # put input in the range [0, 1]
            x = x * self.std + self.mean
        if return_feature:
            logits = self.model(x)
            features = self.feature_forward(x)
            return logits, features
        return self.model(x)
    
    def eval(self):
        self.model.eval()
    
    def to(self, device):
        self.model.to(device)

    def set_normalise(self, normalise):
        self.normalise = normalise
    
    def __getattr__(self, name):
        return getattr(self.model, name)