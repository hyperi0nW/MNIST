import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2

torch.manual_seed(1)#每次随机初始化相同

EPOCH = 5
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True
device = ('cuda')

train_data = torchvision.datasets.MNIST(
    root = './data/',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    
    download = DOWNLOAD_MNIST
)

test_data = torchvision.datasets.MNIST(
    root = './data/',
    train = False,
    transform = torchvision.transforms.ToTensor(),
    
    download = DOWNLOAD_MNIST
)

train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle = True
)

print(test_data[0][0].shape,test_data[0][1])

test_x = torch.unsqueeze(test_data.test_data, dim=1)[:200]
test_y = test_data.test_labels[:200]
print(test_x.shape, test_y.shape)

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

net = LeNet_NumRec()

print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_fun = nn.CrossEntropyLoss()

net = net.to(device)
print("training on ", device)

for epoch in range(EPOCH):
    #参数清零
    loss_sum = 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        #用gpu计算
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        #前向传播计算y_hat
        output = net(batch_x)
        loss = loss_fun(output, batch_y)
        #清空梯度，因为梯度数据是累加的
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #应用梯度
        optimizer.step()
        #记录loss
        loss_sum += loss.item()
    print("EPOCH ", epoch, "LOSS IS ", loss_sum, "\n")

torch.save(net.state_dict(), 'lenet.pth')