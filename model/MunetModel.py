import torch
import torch.nn as nn

import torch.nn as nn
import torch
from torch import autograd
 
 
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)
 
 
class Munet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Munet, self).__init__()
 
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.detach1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.detachconv1 = DoubleConv(32, 32)
        self.predict1 = nn.Conv2d(32, out_ch, 1)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.detach21 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.detachconv21 = DoubleConv(64, 64)
        self.detach22 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.detachconv22 = DoubleConv(32, 32)
        self.predict2 = nn.Conv2d(32, out_ch, 1)


        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
 
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)# 128 128
        detach1 = self.detach1(p1.detach())
        detachconv1 = self.detachconv1(detach1)
        predict1 = self.predict1(detachconv1)
        # out1 = nn.Sigmoid()(predict1)


        c2 = self.conv2(p1)
        p2 = self.pool2(c2)# 64 64
        detach2 = self.detach21(p2.detach())
        detachconv21 = self.detachconv21(detach2)
        detach22 = self.detach22(detachconv21)
        detachconv22 = self.detachconv22(detach22)
        predict2 = self.predict2(detachconv22)
        # out2 = nn.Sigmoid()(predict2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)# 32 32
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)# 16 16
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1) #32
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1) #64
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1) #128
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1) #256
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10+predict1+predict2)
        return out


if __name__ == "__main__":
    model = Munet(1,1)
    print(model)
    input = torch.randn((8,1,256,256))
    out = model(input)
    print(out.shape)