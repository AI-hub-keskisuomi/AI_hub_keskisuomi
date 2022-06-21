import torch.nn as nn
from torchvision import models


class Model(nn.Module):
    """CNN model"""

    def __init__(self, model_param):
        super(Model, self).__init__()

        if model_param == "vgg19":
            self.model = models.vgg19(pretrained=True)
            self.model.classifier[-1] = nn.Linear(4096, 2)
        elif model_param == "resnext":
            self.model = models.resnext50_32x4d(pretrained=True)
            self.model.fc = nn.Linear(2048, 2)
        else:
            raise Exception("Error: request was unknown model")

    def forward(self, x):
        x = self.model(x)
        return x

    def last_conv(self, model_param):
        # for grad-cam
        if model_param == "resnext":
            return self.model.layer4[-1].conv3
