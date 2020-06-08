import torch
import torch.nn as nn
import torchvision.models as models
import math

class Simple_NN(torch.nn.Module):
    def __init__(self, max_vector):
        super(Simple_NN, self).__init__()

        self.fc1 = nn.Linear(200000, max_vector)
        self._initialize_weights()

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        

        h = x
        h = self.fc1(h)
        
        return h