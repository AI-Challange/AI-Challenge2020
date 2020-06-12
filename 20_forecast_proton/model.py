import torch.nn as nn


class LinearRegression(nn.Module):

    def __init__(self, nIn, nOut):
        super(LinearRegression, self).__init__()

        self.LR = nn.Linear(nIn, nOut)

    def forward(self, input):
        
        output = self.LR(input)  # [T * b, nOut]

        return output


'''
RNN Model Reference
https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection
'''