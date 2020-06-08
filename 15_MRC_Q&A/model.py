import torch
import torch.nn as nn


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(128, 2)
        )

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
        h = self.linear(x)

        return h