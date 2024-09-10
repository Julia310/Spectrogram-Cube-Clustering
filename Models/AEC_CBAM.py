import torch.nn.functional as F
import torch.nn as nn
from Models.CBAM import CBAMBlock

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
        self.cbam1 = CBAMBlock(channel=16)  # CBAM after convolution
        self.down_convolution_2 = double_convolution_down(16, 32)
        self.cbam2 = CBAMBlock(channel=32)
        self.down_convolution_3 = double_convolution_down(32, 64)
        self.cbam3 = CBAMBlock(channel=64)
        self.down_convolution_4 = double_convolution_down(64, 128)
        self.down_flatten = down_linear(128*4*8, 16)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_convolution_1 = double_convolution_up(128, 64)
        self.cbam4 = CBAMBlock(channel=64)
        self.up_convolution_2 = double_convolution_up(64, 32)
        self.cbam5 = CBAMBlock(channel=32)
        self.up_convolution_3 = double_convolution_up(32, 16)
        self.cbam6 = CBAMBlock(channel=16)
        self.up_convolution_4 = double_convolution_up(16, 1)
        self.up_unflatten = up_linear(16, 128*4*8)


    def forward(self, x):
        # Encoder
        down_1 = self.down_convolution_1(x)
        down_1 = self.cbam1(down_1)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_3 = self.cbam2(down_3)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_5 = self.cbam3(down_5)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.down_flatten(down_7)
        # Decoder
        up_0 = self.up_unflatten(down_8)
        up_1 = self.up_convolution_1(up_0)
        up_1 = self.cbam4(up_1)
        up_2 = F.pad(self.upsample(up_1), (0, 1, 0, 0))
        up_3 = self.up_convolution_2(up_2)
        up_3 = self.cbam5(up_3)
        up_4 = F.pad(self.upsample(up_3), (0, 1, 0, 1))
        up_5 = self.up_convolution_3(up_4)
        up_5 = self.cbam6(up_5)
        up_6 = F.pad(self.upsample(up_5), (0, 1, 0, 0))
        up_7 = self.up_convolution_4(up_6)
        return up_7


#from torchinfo import summary

#summary(AEC(), (1, 1, 4, 101))
