import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as torchmodels
import functools
import numpy as np 

# spatialattention 和 channelattention 的混合使用
class SpatialAttention(nn.Module):
    def __init__(self,in_channels,kernel_size=9):
        super(SpatialAttention,self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size-1)//2

        self.conv1 = nn.Conv2d(self.in_channels,self.in_channels//2,(1,kernel_size),padding=(0,pad))
        self.batchnorm1 = nn.BatchNorm2d(self.in_channels//2)
        self.conv2 = nn.Conv2d(self.in_channels//2,1,(self.kernel_size,1),padding=(pad,0))
        self.batchnorm2 = nn.BatchNorm2d(1)

        self.conv3 = nn.Conv2d(self.in_channels,self.in_channels//2,(kernel_size,1),padding=(pad,0))
        self.batchnorm3 = nn.BatchNorm2d(self.in_channels//2)
        self.conv4 = nn.Conv2d(self.in_channels//2,1,(1,kernel_size),padding=(0,pad))
        self.batchnorm4 = nn.BatchNorm2d(1)
    
    def forward(self,x):
        feature1 = self.conv1(x)
        feature1 = F.relu(self.batchnorm1(feature1))
        feature1 = self.conv2(feature1)
        feature1 = F.relu(self.batchnorm2(feature1))

        feature2 = self.conv3(x)
        feature2 = F.relu(self.batchnorm3(feature2))
        feature2 = self.conv4(feature2)
        feature2 = F.relu(self.batchnorm4(feature2))

        weights = torch.sigmoid(torch.add(feature1,feature2))
        return weights.expand_as(x).clone()

# 是否就是se 模块
class ChannelAttention(nn.Module):
    def __init__(self,in_channels):
        super(ChannelAttention,self).__init__()
        self.in_channels = in_channels
        self.linear1 = nn.Linear(self.in_channels,self.in_channels//4)
        self.linear2 = nn.Linear(self.in_channels//4,self.in_channels)
    
    def forward(self,input):
        b,c,h,w = input.shape
        weights = F.adaptive_avg_pool2d(input,(1,1)).view(b,c)
        weights = F.relu(self.linear1(weights))
        weights = torch.sigmoid(self.linear2(weights))
        weights = weights.view((b,c,1,1))
        # # Activity regularizer 是否需要正则
        return weights.expand_as(input).clone()

# context-aware pyramid feature extraction
class Cpfe(nn.Module):
    def __init__(self,in_channels,feature_layer=None,out_channels=32):
        super(Cpfe,self).__init__()
        self.dil_rate = [3,5,7]
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1_1 = nn.Conv2d(in_channels=self.in_channels,out_channels=out_channels,kernel_size=1,bias=False)
        self.conv_dil_3 = nn.Conv2d(in_channels=self.in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=self.dil_rate[0],padding=self.dil_rate[0],bias=False)
        self.conv_dil_5 = nn.Conv2d(in_channels=self.in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=self.dil_rate[1],padding=self.dil_rate[1],bias=False)
        self.conv_dil_7 = nn.Conv2d(in_channels=self.in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=self.dil_rate[2],padding=self.dil_rate[2],bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels*4)

    def forward(self,input):
        conv1_1 = self.conv1_1(input)
        conv_dil_3 = self.conv_dil_3(input)
        conv_dil_5 = self.conv_dil_5(input)
        conv_dil_7 = self.conv_dil_7(input)

        concat_feat = torch.cat((conv1_1,conv_dil_3,conv_dil_5,conv_dil_7),dim=1)
        bn_feat = F.relu(concat_feat)
        return bn_feat

# 这边可换成resNet
vgg_cov1_2 = vgg_cov2_2 = vgg_cov3_3 = vgg_cov4_3 = vgg_cov5_3 = None
def conv1_2_hook(module,input,output):
    global vgg_cov1_2
    vgg_cov1_2 = output
    return None

def conv2_2_hook(module,input,output):
    global vgg_cov2_2
    vgg_cov2_2 = output
    return None

def conv3_3_hook(module,input,output):
    global vgg_cov3_3
    vgg_cov3_3 = output
    return None

def conv4_3_hook(module,input,output):
    global vgg_cov4_3
    vgg_cov4_3 = output
    return None

def conv5_3_hook(module,input,output):
    global vgg_cov5_3
    vgg_cov5_3 = output
    return None

# saliency colorization 模型
class SCModel(nn.Module):
    def __init__(self):
        super(SCModel,self).__init__()
        self.cpfe_conv3_3_in_channels = 256
        self.cpfe_conv4_3_in_channels = 512
        self.cpfe_conv5_3_in_channels = 512
        self.ca_in_channels = 384
        self.vgg16 = torchmodels.vgg16(pretrained=True).features
        self.vgg16[3].register_forward_hook(conv1_2_hook)
        self.vgg16[8].register_forward_hook(conv2_2_hook)
        self.vgg16[15].register_forward_hook(conv3_3_hook)
        self.vgg16[22].register_forward_hook(conv4_3_hook)
        self.vgg16[29].register_forward_hook(conv5_3_hook)

        self.cpfe_conv3_3 = Cpfe(in_channels=self.cpfe_conv3_3_in_channels)
        self.cpfe_conv4_3 = Cpfe(in_channels=self.cpfe_conv4_3_in_channels)
        self.cpfe_conv5_3 = Cpfe(in_channels=self.cpfe_conv5_3_in_channels)

        self.ca = ChannelAttention(in_channels=self.ca_in_channels)
        
        self.hight_level_conv = nn.Conv2d(384,64,(3,3),padding=1)
        self.hight_level_bn = nn.BatchNorm2d(64)

        self.low_level_conv1 = nn.Conv2d(64,64,(3,3),padding=1)
        self.low_level_bn1 = nn.BatchNorm2d(64)
        self.low_level_conv2 = nn.Conv2d(128,64,(3,3),padding=1)
        self.low_level_bn2 = nn.BatchNorm2d(64)
        self.low_level_conv3 = nn.Conv2d(128,64,(3,3),padding=1)
        self.low_level_bn3 = nn.BatchNorm2d(64)

        self.sa = SpatialAttention(in_channels=64)
        # self.fused_feat = nn.Conv2d(128,1,(3,3),padding=1)
        self.fused_feat = nn.Conv2d(128,3,(3,3),padding=1)


    def forward(self,input):
        global vgg_cov1_2,vgg_cov2_2,vgg_cov3_3,vgg_cov4_3,vgg_cov5_3
        self.vgg16(input)
        cpfe_conv3_3 = self.cpfe_conv3_3(vgg_cov3_3)
        cpfe_conv4_3 = self.cpfe_conv4_3(vgg_cov4_3)
        cpfe_conv5_3 = self.cpfe_conv5_3(vgg_cov5_3)

        cpfe_conv4_3 = F.interpolate(cpfe_conv4_3,scale_factor=2,mode='bilinear',align_corners=True)
        cpfe_conv5_3 = F.interpolate(cpfe_conv5_3,scale_factor=4,mode='bilinear',align_corners=True)

        concat_345 = torch.cat((cpfe_conv3_3,cpfe_conv4_3,cpfe_conv5_3),dim=1)
        # 这边时候要返回attention
        ca_345 = self.ca(concat_345)
        ca_345_feat = torch.mul(ca_345,concat_345)
        conv_345 = self.hight_level_conv(ca_345_feat)
        conv_345 = F.relu(self.hight_level_bn(conv_345))
        conv_345 = F.interpolate(conv_345,scale_factor=4,mode='bilinear',align_corners=True)

        # low level feature
        conv1_feat = self.low_level_conv1(vgg_cov1_2)
        conv1_feat = F.relu(self.low_level_bn1(conv1_feat))
        conv2_feat = self.low_level_conv2(vgg_cov2_2)
        conv2_feat = F.relu(self.low_level_bn2(conv2_feat))
        conv2_feat = F.interpolate(conv2_feat,scale_factor=2,mode='bilinear',align_corners=True)
        conv_12 = torch.cat((conv2_feat,conv1_feat),dim=1)
        conv_12 = self.low_level_conv3(conv_12)
        conv_12 = F.relu(self.low_level_bn3(conv_12))

        # spatial attention
        sa_12 = self.sa(conv_12)
        conv_12 = torch.mul(sa_12,conv_12)

        #fused features
        fused_feat = torch.cat((conv_12,conv_345),dim=1)
        # fused_feat = torch.sigmoid(self.fused_feat(fused_feat))
        # 对于着色任务，这里不该用sigmoid，ab_channel是可能为负的
        fused_feat = torch.sigmoid(self.fused_feat(fused_feat))
        return fused_feat

class NoNormDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super(NoNormDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)



if __name__ == "__main__":
    '''
    1:  测试注意力模块
    '''
    # input = torch.randn(2,4,16,8)
    # ca = ChannelAttention(4)
    # weights = ca(input)
    # print('ca weights',weights.shape)
    # sa = SpatialAttention(4)
    # weights = sa(input)
    # print('sa weight',weights.shape)
    '''
    2:  测试cpfe模块
    ''' 
    # input = torch.randn(2,32,64,64)
    # cpfe = Cpfe(32)
    # output = cpfe(input)
    # print(output.shape)
    '''
    3:  测试model 是否能够正常运行
    '''
    input = torch.randn(2,3,256,256)
    model = SCModel()
    out = model(input)
    print(model)
    print('out',out.shape)
    # print(out)

    '''
    4: 测试 NoNormDiscriminator
    '''
    # disc = NoNormDiscriminator(3)
    # # disc = NLayerDiscriminator(3)
    # print(disc)
    # input = torch.randn((1,3,256,256))
    # out = disc(input)
    # print(out.shape)
