import torch
import torch.nn as nn
import torchvision.models as models


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(45525, 500),
            nn.Linear(500, num_classes),
            nn.ReLU(inplace=True))


    def forward(self, x):
        x = self.fc(x)

        return x