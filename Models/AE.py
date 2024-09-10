import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from Models.CBAM import CBAMBlock

import torch
from torch import nn


class EmbeddingToImage(nn.Module):
    def __init__(self, embed_dim = 1024, channels=1, image_height=4, image_width=101):
        super().__init__()
        self.channels = channels
        self.image_height = image_height
        self.image_width = image_width
        # Reverse of dimensionality reduction: project embeddings back to the original flattened image size
        self.project_to_image = nn.Linear(embed_dim, channels * image_height * image_width)
        # Reshape the flat vector back to the original image dimensions
        self.unflatten = nn.Unflatten(1, (channels, image_height, image_width))

    def forward(self, x):
        x = self.project_to_image(x)  # Project embeddings to flat image vector
        x = x.view(-1, self.channels, self.image_height, self.image_width)
        return x

def down_linear(in_features):
    conv_op = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, in_features // 4),
        nn.Sigmoid(),
        nn.Linear(in_features // 4, in_features // 16),
        nn.Sigmoid(),
        nn.Linear(in_features // 16, in_features // 64),
        nn.Sigmoid(),
        #nn.Linear(in_features // 64, in_features // 128),
        #nn.Sigmoid(),

    )
    return conv_op

def up_linear(out_features):
    conv_op = nn.Sequential(
        #nn.Linear(out_features // 128, out_features // 64),
        #nn.Sigmoid(),
        nn.Linear(out_features // 64, out_features // 16),
        nn.Sigmoid(),
        nn.Linear(out_features // 16, out_features // 4),
        nn.Sigmoid(),
        nn.Linear(out_features // 4, out_features),
        nn.Sigmoid(),
        nn.Unflatten(1, (1, 1024))
    )
    return conv_op

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=2048, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)





class AE(nn.Module):
    def __init__(self, dim = 1024, channels = 1, image_height = 4, image_width = 101):
        super().__init__()
        #self.norm = nn.LayerNorm(dim)
        self.flatten = nn.Flatten(start_dim=1)
        self.embed = nn.Sequential(nn.LayerNorm(channels * image_height * image_width),
                                 nn.Linear(channels * image_height * image_width, dim),
                                 nn.LayerNorm(dim))
        self.layer1 = FeedForward(dim)
        self.layer2 = FeedForward(dim)
        self.layer3 = FeedForward(dim)
        self.layer4 = FeedForward(dim)
        self.layer5 = FeedForward(dim)
        self.layer6 = FeedForward(dim)
        self.layer7 = FeedForward(dim)
        self.layer8 = FeedForward(dim)
        self.layer9 = FeedForward(dim)
        self.layer10 = FeedForward(dim)
        self.layer11 = FeedForward(dim)
        self.layer12 = FeedForward(dim)
        self.down_flatten = down_linear(1024)
        self.up_flatten = up_linear(1024)
        self.resize = EmbeddingToImage()

    def forward(self, x):
        x = self.flatten(x)
        x = self.embed(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.down_flatten(x)
        x = self.up_flatten(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.resize(x)
        return x





#images = torch.randn(8, 1, 4, 101)
#from torchinfo import summary

#summary(AE(), (8, 1, 4, 101))