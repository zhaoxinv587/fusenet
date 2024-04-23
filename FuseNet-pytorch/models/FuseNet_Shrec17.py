import torch
from torch import nn
import torch.nn.functional as F


# 224 * 224 * 3
class FuseNet(nn.Module):
    def __init__(self):
        super(FuseNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # 224 * 224 * 64
            nn.BatchNorm2d(64),  # Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，一方面使得数据分布一致，另一方面避免梯度消失。
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 224 * 224 * 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 * 112 * 64
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 112 * 112 * 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # 112 * 112 * 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # 56 * 56 * 128
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # 28 * 28 * 256
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 28 * 28 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 28 * 28 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 28 * 28 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # 14 * 14 * 512
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # 7 * 7 * 512
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )



        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 55)
        )

    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)


        x1 = torch.max(x1.view(-1,1024,28*28), 2, keepdim=True)[0].view(-1,1024)
        x3 = torch.max(x3.view(-1,1024,14*14), 2, keepdim=True)[0].view(-1,1024)
        x5 = self.fc1(x5.view(-1,512*7*7))

        mvcnn1 = x1.view((int(x1.shape[0] / 12), 12, x1.shape[1]))
        mvcnn1_avg = torch.mean(mvcnn1,1)
        mvcnn1_max = torch.max(mvcnn1,1)[0]

        mvcnn3 = x3.view((int(x3.shape[0] / 12), 12, x3.shape[1]))
        mvcnn3_avg = torch.mean(mvcnn3,1)
        mvcnn3_max = torch.max(mvcnn3,1)[0]

        mvcnn5 = x5.view((int(x5.shape[0] / 12), 12, x5.shape[1]))
        mvcnn5_avg = torch.mean(mvcnn5,1)
        mvcnn5_max = torch.max(mvcnn5,1)[0]


        return mvcnn1_avg,mvcnn1_max,mvcnn3_avg,mvcnn3_max,mvcnn5_avg,mvcnn5_max


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 55)
        )


    def forward(self,x):
        # 归一化权重
        fuse =self.fc(x)
        return fuse


class W(nn.Module):
    def __init__(self):
        super(W, self).__init__()

        #设置权重
        # self.w = nn.Parameter(torch.ones(5))
        self.m1 = nn.Parameter(torch.ones(2))
        self.m2 = nn.Parameter(torch.ones(2))
        self.m3 = nn.Parameter(torch.ones(2))
        self.relu = nn.ReLU(inplace=True)
        #self.bn = nn.BatchNorm1d(512)
    def forward(self,x1,x2,x3,x4,x5,x6):
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x3 = self.relu(x3)
        x4 = self.relu(x4)
        x5 = self.relu(x5)
        x6 = self.relu(x6)
        m11 = torch.exp(self.m1[0]) / torch.sum(torch.exp(self.m1))
        m12 = torch.exp(self.m1[1]) / torch.sum(torch.exp(self.m1))
        fuse1 = m11*x1 + m12*x2
        m21 = torch.exp(self.m2[0]) / torch.sum(torch.exp(self.m2))
        m22 = torch.exp(self.m2[1]) / torch.sum(torch.exp(self.m2))
        fuse2 = m21*x3 + m22*x4
        m31 = torch.exp(self.m3[0]) / torch.sum(torch.exp(self.m3))
        m32 = torch.exp(self.m3[1]) / torch.sum(torch.exp(self.m3))
        fuse3 = m31*x5 + m32*x6
        # print(m11)
        # print(m12)
        # print(m22)
        # print(m21)
        # print(m31)
        # print(m32)
        return fuse1,fuse2,fuse3



class W1(nn.Module):
    def __init__(self):
        super(W1, self).__init__()
        #设置权重
        self.w = nn.Parameter(torch.ones(3))

    def forward(self,x1,x2,x3):
        # 归一化权重
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        fuse = x1*w1 + x2*w2 + x3*w3
        return fuse










