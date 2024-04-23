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

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool1 = nn.AdaptiveAvgPool2d((7,7))
        self.max_pool1 = nn.AdaptiveMaxPool2d((7,7))



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

            nn.Linear(256, 40)
        )

    def forward(self, x):
        # x = self.conv(x)
        #
        #
        # x = x.view(-1, 7 * 7 * 512)
        #
        #
        # y = x.view((int(x.shape[0]/12),12,x.shape[1]))#(8,12,512,7,7)
        #
        # y = torch.max(y, 1)[0].view(y.shape[0], -1)
        # x = y.view(-1, 7 * 7 * 512,1,1)
        # #
        # b, c, _, _ = x.size()
        # x = self.avg_pool(x).view(b, c)
        # x = self.fc(x).view(b, c)
        # y = x*y #8 25088
        # y = self.fc1(y)

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

        # fuse = fuse.view((int(fuse.shape[0] / 12), 12, fuse.shape[1]))
        # fuse_avg = torch.mean(fuse,1)
        # fuse_max = torch.max(fuse,1)[0].view(fuse.shape[0],-1)


        # 1 3 5
        # x1 = x1.view(96,256,56,56)
        # x1 = self.avg_pool(x1)
        # fuse = torch.cat((x1,x3),0).view(192,512,14,28)
        # fuse = self.avg_pool1(fuse)
        # fuse = torch.cat((x5,fuse),0)
        # fuse = fuse.view(-1,512*7*7)
        #
        # fuse = fuse.view((3,fuse.shape[1],int(fuse.shape[0]/36),12))#(8,12,1024)
        # fuse = torch.max(fuse,0)[0].view(8,12,25088)
        # fuse = torch.max(fuse, 1)[0].view(fuse.shape[0], -1)
        # fuse = self.fc1(fuse)

        # x1 = self.fc1(self.max_pool(x1.view(96,512,14,112)).view(-1,7 * 7 * 512))
        # x3 = self.fc1(self.max_pool(x3.view(96,512,14,28)).view(-1,7 * 7 * 512))
        # x5 = self.fc1(x5.view(-1,7 * 7 * 512))


        #12345 avg+max
        # x1m = self.max_pool(x1.view(-1,1024,28,28))
        # x1a = self.avg_pool(x1.view(-1,1024,28,28))
        # x1 = x1a + x1m
        # x2m = self.max_pool(x2.view(-1,1024,28,14))
        # x2a = self.avg_pool(x2.view(-1,1024,28,14))
        # x2 = x2a + x2m
        # x3m = self.max_pool(x3.view(-1,1024,14,14))
        # x3a = self.avg_pool(x3.view(-1,1024,14,14))
        # x3 = x3a + x3m
        # x4m = self.max_pool(x4.view(-1,1024,14,7))
        # x4a = self.avg_pool(x4.view(-1,1024,14,7))
        # x4 = x4a + x4m
        # x5m = self.max_pool1(x5)
        # x5a = self.avg_pool1(x5)
        # x5 = x5a + x5m
        # x5= self.fc1(x5.view(-1,512*7*7)).view(-1,1024,1,1)


        #fuse = torch.add(torch.add(x1,x3),x5)
        #fuse = torch.cat((x1,x3,x5),1)

        # fuse = x1 * 0.15 + x3 * 0.35 + x5 * 0.6 #0.9336569579288025
        # fuse = fuse.view((int(fuse.shape[0] / 12), 12, fuse.shape[1]))
        # fuse = torch.max(fuse,1)[0].view(fuse.shape[0],-1)


        # fuse = fuse.view((int(fuse.shape[0] / 12), 12, fuse.shape[1]))
        # fuse = torch.max(fuse,1)[0].view(fuse.shape[0],-1)

        #senet
        # y = x.view(-1, 7 * 7 * 512,1,1)
        # b, c, _, _ = y.size()
        # y = self.avg_pool(y).view(b, c)
        # y = self.fc(y).view(b, c)
        #x = x.view(-1, 7 * 7 * 512)
        # x = x*y #8 25088


        # x = self.fc1(x)
        # y = x.view((int(x.shape[0]/12),12,x.shape[1]))#(8,12,1024)
        #
        # y = torch.max(y, 1)[0].view(y.shape[0], -1) #8 25088
        #得到了基于注意力和MVCNN的多视图特征向量



        #y = self.fc1(y)





        # 这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成7*7*512列
        # 那不确定的地方就可以写成-1
        # 如果出现x.size(0)表示的是batchsize的值
        # x=x.view(x.size(0),-1)
        # x = x.view(-1, 7 * 7 * 512)
        # x = self.fc(x)

        # 1 3 5 和 point 123


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
            nn.Linear(512, 40)
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
        # 归一化权重
        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        # w4 = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))
        # w5 = torch.exp(self.w[4]) / torch.sum(torch.exp(self.w))
        # w6 = torch.exp(self.w[5]) / torch.sum(torch.exp(self.w))
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
        # fuse = x1 * w1 + x2 * w2 +x3 * w3 + x4 * w4 + x5 * w5
        # fuse = fuse.view((int(fuse.shape[0] / 12), 12, fuse.shape[1]))
        # fuse_avg = torch.mean(fuse,1)
        # fuse_max = torch.max(fuse,1)[0].view(fuse.shape[0],-1)
        # m1 = torch.exp(self.m[0]) / torch.sum(torch.exp(self.m))
        # m2 = torch.exp(self.m[1]) / torch.sum(torch.exp(self.m))
        # fuse = m1 * fuse_avg + m2 * fuse_max
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

class MS_CAM(nn.Module):
    '''
    单特征加权
    '''

    def __init__(self, channels=1024, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(512)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x*wei




class AFF(nn.Module):
    '''
    多特征融合 MS_CAM
    '''

    def __init__(self, channels=1024):
        super(AFF, self).__init__()
        inter_channels = 512


        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    # def forward(self, x, y):
    #     xa = x + y
    #     xa = xa.view(-1, 1024, 1, 1)
    #     xl = self.local_att(xa)
    #     xg = self.global_att(xa)
    #     xlg = xl + xg #([8, 2048, 1, 1])
    #     wei = self.sigmoid(xlg) #([8, 2048, 1, 1])
    #     x = 2 * x.view(-1, 1024, 1, 1) * wei
    #     y = 2 * y.view(-1, 1024, 1, 1) * (1 - wei)
    #     fuse = torch.cat((x ,y),1)
    #     #fuse = x + y
    #     return fuse

    def forward(self, x, y):
        x = x.view(8, 1024, 1, 1)
        y = y.view(8, 1024, 1, 1)
        xl = self.local_att(x)
        xg = self.global_att(x)
        yl = self.local_att(y)
        yg = self.global_att(y)
        xlg = xl + xg #([8, 2048, 1, 1])
        ylg = yl + yg
        xwei = self.sigmoid(xlg) #([8, 2048, 1, 1])
        ywei = self.sigmoid(ylg) #([8, 2048, 1, 1])
        x = x.view(8, 1024, 1, 1) * xwei
        y = y.view(8, 1024, 1, 1) * ywei
        fuse = torch.cat((x ,y),1)
        #fuse = x + y
        return fuse


class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=1024, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(512)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次局部注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xa = x + y
        xa = xa.view(8, 1024, 1, 1)
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        x = 2 * x.view(8, 1024, 1, 1) * wei
        y = 2 * y.view(8, 1024, 1, 1) * (1 - wei)
        xi = x+y

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        x = 2 * x.view(8, 1024, 1, 1) * wei2
        y = 2 * y.view(8, 1024, 1, 1) * (1 - wei2)
        #fuse = torch.cat((x,y),1)
        #fuse = torch.cat((x,y),1)
        fuse = x + y
        return fuse




class SEAttention(nn.Module):

    def __init__(self, channel, reduction=16):   # channel为输入通道数，reduction压缩大小
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
