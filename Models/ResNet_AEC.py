import torch.nn.functional as F
import torch.nn as nn
from Models.CBAM import CBAMBlock



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


def double_convolution_down(in_channels, out_channels):
    """
    In the original paper implementation, the convolution operations were
    not padded but we are padding them here. This is because, we need the
    output result size to be same as input size.
    """
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(2, 4), padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=(2, 4), padding=1),
        nn.ReLU(inplace=True)

    )
    return conv_op

def double_convolution_up(in_channels, out_channels):
    """
    In the original paper implementation, the convolution operations were
    not padded but we are padding them here. This is because, we need the
    output result size to be same as input size.
    """
    conv_op = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 4), padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=(2, 4), padding=1),
        nn.ReLU(inplace=True)

    )
    return conv_op

def double_convolution2(in_channels, out_channels):
    """
    In the original paper implementation, the convolution operations were
    not padded but we are padding them here. This is because, we need the
    output result size to be same as input size.
    """
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)

    )
    return conv_op


def down_linear(in_features, out_features):
    """
    In the original paper implementation, the convolution operations were
    not padded but we are padding them here. This is because, we need the
    output result size to be same as input size.
    """
    conv_op = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, in_features // 4),
        nn.Sigmoid(),
        nn.Linear(in_features // 4, in_features // 16),
        nn.Sigmoid(),
        nn.Linear(in_features // 16, in_features // 64),
        nn.Sigmoid(),
        nn.Linear(in_features // 64, in_features // 256),
        nn.Sigmoid(),
        #nn.Linear(in_features // 64, in_features // 128),
        #nn.Sigmoid(),
        #nn.Linear(in_features // 128, out_features),
        #nn.Sigmoid(),

    )
    return conv_op


def up_linear(in_features, out_features):
    conv_op = nn.Sequential(
        #nn.Linear(in_features, out_features // 128),
        #nn.Sigmoid(),
        #nn.Linear(out_features // 128, out_features // 64),
        #nn.Sigmoid(),
        nn.Linear(in_features, out_features // 256),
        nn.Sigmoid(),
        nn.Linear(out_features // 256, out_features // 64),
        nn.Sigmoid(),
        nn.Linear(out_features // 64, out_features // 16),
        nn.Sigmoid(),
        nn.Linear(out_features // 16, out_features // 4),
        nn.Sigmoid(),
        nn.Linear(out_features // 4, out_features),
        nn.Sigmoid(),
        nn.Unflatten(1, (128, 4, 8))
    )
    return conv_op


class AEC(nn.Module):
    def __init__(self):
        super(AEC, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_convolution_1 = double_convolution_down(1, 16)
        self.res1 = ResidualBlock(16, 16)
        self.cbam1 = CBAMBlock(channel=16)  # CBAM after convolution
        self.down_convolution_2 = double_convolution_down(16, 32)
        self.res2 = ResidualBlock(32, 32)
        self.cbam2 = CBAMBlock(channel=32)
        self.down_convolution_3 = double_convolution_down(32, 64)
        self.res3 = ResidualBlock(64, 64)
        self.cbam3 = CBAMBlock(channel=64)
        self.down_convolution_4 = double_convolution_down(64, 128)
        self.res4 = ResidualBlock(128, 128)
        self.cbam4 = CBAMBlock(channel=128)
        self.down_flatten = down_linear(128*4*8, 16)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_convolution_1 = double_convolution_up(128, 64)
        self.res5 = ResidualBlock(64, 64)
        self.cbam5 = CBAMBlock(channel=64)
        self.up_convolution_2 = double_convolution_up(64, 32)
        self.res6 = ResidualBlock(32, 32)
        self.cbam6 = CBAMBlock(channel=32)
        self.up_convolution_3 = double_convolution_up(32, 16)
        self.res7 = ResidualBlock(16, 16)
        self.cbam7 = CBAMBlock(channel=16)
        self.up_convolution_4 = double_convolution_up(16, 1)
        self.up_unflatten = up_linear(16, 128*4*8)


    def forward(self, x):
        # Encoder
        down_1 = self.down_convolution_1(x)
        down_1 = self.res1(down_1)
        down_1 = self.cbam1(down_1)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_3 = self.res2(down_3)
        down_3 = self.cbam2(down_3)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_5 = self.res3(down_5)
        down_5 = self.cbam3(down_5)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_7 = self.res4(down_7)
        down_7 = self.cbam4(down_7)
        down_8 = self.down_flatten(down_7)
        # Decoder
        up_1 = self.up_unflatten(down_8)
        up_2 = self.up_convolution_1(up_1)
        up_2 = self.res5(up_2)
        up_2 = self.cbam5(up_2)
        up_3 = F.pad(self.upsample(up_2), (0, 1, 0, 0))
        up_4 = self.up_convolution_2(up_3)
        up_4 = self.res6(up_4)
        up_4 = self.cbam6(up_4)
        up_5 = F.pad(self.upsample(up_4), (0, 1, 0, 1))
        up_6 = self.up_convolution_3(up_5)
        up_6 = self.res7(up_6)
        up_6 = self.cbam7(up_6)
        up_7 = F.pad(self.upsample(up_6), (0, 1, 0, 0))
        up_8 = self.up_convolution_4(up_7)
        return up_8


#from torchinfo import summary

#summary(AEC(), (1, 1, 4, 101))