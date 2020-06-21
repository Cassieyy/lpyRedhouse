import torch.nn as nn
import torchvision.ops as ops
import torch
import torch.nn.functional as F

class DeformNet(nn.Module):
    def __init__(self):
        super(DeformNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(stride=2,kernel_size=2)
        self.dconv3 = BasicDeformConv2d(64,128,kernel_size=3)
        self.hidden = nn.Linear(14*14*128,1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x):
        # convs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = F.relu(self.dconv3(x))


        x = x.view(-1, 14*14*128)
        x = F.relu(self.hidden(x))
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class BasicDeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, groups=1, offset_groups=1):
        super().__init__()
        offset_channels = 2 * kernel_size * kernel_size
        self.conv2d_offset = nn.Conv2d(
            in_channels,
            offset_channels * offset_groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2d = ops.DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=groups,
            bias=False
        ) 
    def forward(self, x):
        offset = self.conv2d_offset(x)
        return self.conv2d(x, offset)

if __name__ == "__main__":
    input = torch.randn(1,1,28,28)
    net = DeformNet()
    out = net(input)
    print(out.shape)
    print(net)