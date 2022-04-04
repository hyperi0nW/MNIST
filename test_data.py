import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2
import numpy

device = ('cuda')

test_data = torchvision.datasets.MNIST(
    root = './data/',
    train = False,
    transform = torchvision.transforms.ToTensor(),
    download = True
)

test_loader = Data.DataLoader(
    dataset = test_data,
    batch_size = 10,
    shuffle = True
)

class LeNet_NumRec(nn.Module):
    def __init__(self):
        super(LeNet_NumRec, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
                ),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.Out = nn.Linear(16*14*14, 10)
    
    def forward(self, x):
        x = self.Conv(x)
        x = x.view(x.size(0), -1)#batchsize行个tensor
        output = self.Out(x)
        return output

net1 = LeNet_NumRec()
net1.load_state_dict(torch.load('lenet.pth'))
print(net1)

for step, (batch_x, batch_y) in enumerate(test_loader):
    input = batch_x
    y = net1(input)
    pred, idx = y.max(1)
    print(idx[0], batch_y[0])
    a = batch_x[0][0]
    a = a.numpy()
    a = a*255
    cv2.imwrite('2.jpg',a)
    s = cv2.imread('2.jpg')
    cv2.imshow('img2',s)
    cv2.waitKey(0)