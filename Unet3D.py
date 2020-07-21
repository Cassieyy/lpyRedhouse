import torch
import torch.nn as nn
from torch import autograd
import os
from PIL import Image
import cv2
os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
import numpy as np

class ROINet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ROINet, self).__init__()
    
    def forward(self, x):
        pass

class downDouble3dConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downDouble3dConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)
 
class upDouble3dConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upDouble3dConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)

class Unet3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet3D, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding = 1),    
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 64, 3, padding = 1),  
            nn.ReLU(inplace = True)
        )
        
        self.conv1 = downDouble3dConv(64, 128)
        self.pool1 = nn.MaxPool3d(2, 2) # (kernel_size, stride)
        self.conv2 = downDouble3dConv(128, 256)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = downDouble3dConv(256, 512)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.up4 = nn.ConvTranspose3d(512, 512, 2, stride = 2)
        self.conv4 = upDouble3dConv(768, 256)
        self.up5 = nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.conv5 = upDouble3dConv(384, 128)
        self.up6 = nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.conv6 = upDouble3dConv(192, 64)        
        self.conv7 = nn.Conv3d(64, 1, 3, padding = 1)
        self.BN3d = nn.BatchNorm3d(out_ch)
 
    def forward(self, x): # input 4, 1, 5, 256, 256
        
        c0 = self.conv0(x) 
        # print("c0.shape:", c0.shape) # 2, 64, 5, 256, 256
        # assert 1>3 #####为什么会输出两遍结果  为什么batchsize会降为1/2 因为用了多块卡跑
        p1 = self.pool1(c0)
        # print("p1.shape:", p1.shape) # 8, 64, 4, 128, 128
        # assert 1>3
        c1 = self.conv1(p1)
        # print("c1.shape:", c1.shape) # 8, 128, 4, 128, 128
        # assert 1>3    
        p2 = self.pool2(c1)# 64 64
        # print("p2.shape:", p2.shape) # 8, 128, 2, 64, 64
        # assert 1>3   
        c2 = self.conv2(p2)# 8, 256, 2, 64, 64
        # print("c2.shape:", c2.shape) # 8, 64, 2, 64, 64
        # assert 1>3  
        p3 = self.pool3(c2)# 32 32
        c3 = self.conv3(p3)
        # print("c3.shape:", c3.shape) # 8, 512, 1, 32, 32
        # assert 1>3 
        up_4 = self.up4(c3)
        # print("up_4.shape:", up_4.shape) # 8, 512, 2, 64, 64
        merge5 = torch.cat((up_4, c2), dim = 1)
        # print("merge5.shape:", merge5.shape)
        # assert 1>3
        c4 = self.conv4(merge5)
        # print("c4.shape", c4.shape) # 8 256 2 64 64
        # assert 1>3
        up_5 = self.up5(c4) #######注意啊！！！上采样不是pool实现！！！
        # print("up_5.shape", up_5.shape) # 8 256 1 32 32
        # assert 1>3
        merge6 = torch.cat([up_5, c1], dim=1) #32
        # print("merge6.shape:", merge6.shape)

        c5 = self.conv5(merge6)
        # print("c5.shape:", c5.shape)

        up_6 = self.up6(c5)
        # print("up_6.shape:", up_6.shape)
        # assert 1>3
        merge7 = torch.cat([up_6, c0], dim=1) #64
        # print("merge7.shape:", merge7.shape)
        # assert 1>3

        c6 = self.conv6(merge7)
        # print("c6.shape:", c6.shape)
        # assert 1>3

        c7 = self.conv7(c6)
        # print("c7.shape:", c7.shape)
        # assert 1>3
        out = self.BN3d(c7)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet3D(1,1).to(device)
    model = nn.DataParallel(model,device_ids=[0, 1])
    input = torch.randn(4, 1, 8, 256, 256) # BCDHW 
    input = input.to(device)
    out = model(input) 
    print("output.shape:", out.shape) # 4, 1, 8, 256, 256