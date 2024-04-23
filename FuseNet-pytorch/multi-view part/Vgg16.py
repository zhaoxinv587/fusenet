import torch
from torch import nn
import torch.nn.functional as F


# 224 * 224 * 3
class VGGNet16(nn.Module):
    def __init__(self):
        super(VGGNet16, self).__init__()

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


        # channel = 7*7*512
        # reduction = 16
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction),
        #     nn.ReLU(),
        #     nn.Linear(channel // reduction, channel),
        #     nn.Sigmoid()
        # )

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
        x = self.conv(x)

        # 这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成7*7*512列
        # 那不确定的地方就可以写成-1
        # 如果出现x.size(0)表示的是batchsize的值
        # x=x.view(x.size(0),-1)
        # x = x.view(-1, 7 * 7 * 512,1,1)
        #
        # b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        # x = x*y
        x = x.view(-1, 7 * 7 * 512)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



