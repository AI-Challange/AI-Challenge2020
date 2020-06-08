import torch
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16, kernel_size=3,stride=1, padding=0)
        self.batch1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True)
        self.relu1= nn.ReLU()

        self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.batch2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.batch3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()

        self.deconv1=nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3)
        self.batch4 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        self.relu4=nn.ReLU()

        self.maxunpool1=nn.MaxUnpool2d(kernel_size=2)

        self.deconv2=nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=3)
        self.batch5 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True)
        self.relu5=nn.ReLU()

        self.maxunpool2=nn.MaxUnpool2d(kernel_size=2)

        self.deconv3=nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=3)
        self.batch6 = nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True)
        self.relu6=nn.ReLU()

        self._initialize_weights

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

    def forward(self,x):
        h = x
        h=self.conv1(h)
        h=self.batch1(h)
        h=self.relu1(h)
        size1 = h.size()
        h,indices1=self.maxpool1(h)
        h=self.conv2(h)
        h=self.batch2(h)
        h=self.relu2(h)
        size2 = h.size()
        h,indices2=self.maxpool2(h)
        h=self.conv3(h)
        h=self.batch3(h)
        h=self.relu3(h)

        h=self.deconv1(h)
        h=self.batch4(h)
        h=self.relu4(h)
        h=self.maxunpool1(h,indices2,size2)
        h=self.deconv2(h)
        h=self.batch5(h)
        h=self.relu5(h)
        h=self.maxunpool2(h,indices1,size1)
        h=self.deconv3(h)
        h=self.batch6(h)
        h=self.relu6(h)
        
        return(h)

