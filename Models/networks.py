#!/usr/bin/env python3

"""This script contains neural network architectures used in the RIS package.

William Jenkins, wjenkins [at] ucsd [dot] edu
Scripps Institution of Oceanography, UC San Diego
January 2021
"""

import torch
import torch.nn as nn


def double_convolution(in_channels, out_channels):
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


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = double_convolution(1, 16)
        self.down_convolution_2 = double_convolution(16, 32)
        self.down_convolution_3 = double_convolution(32, 64)
        self.down_convolution_4 = double_convolution(64, 128)
        self.down_convolution_5 = double_convolution(128, 256)
        #self.down_flatten = down_linear(128*4*8, 9)
        self.down_flatten = down_linear(128*4*8, 16)
        #self.up_unflatten = up_linear(9, 128*4*8)
        self.up_unflatten = up_linear(16, 128*4*8)
        # Expanding path.
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=(2, 6), padding=(2, 0), output_padding=(0, 1),
            stride=2)
        # Below, `in_channels` again becomes 1024 as we are concatinating.
        self.up_convolution_1 = double_convolution2(256, 128)
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32,
            kernel_size=(2, 6), padding=(2, 0),  output_padding=(1, 1),
            stride=2)
        self.up_convolution_2 = double_convolution2(128, 64)
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16,
            kernel_size=(2, 6), padding=(2, 0), output_padding=(0, 1),
            stride=2)
        self.up_convolution_3 = double_convolution2(64, 32)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2,
            stride=2)
        self.up_convolution_4 = double_convolution2(32, 16)
        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(
            in_channels=16, out_channels=1,
            kernel_size=3, padding=(0, 2),
        )

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.down_flatten(down_7)

        up_0 = self.up_unflatten(down_8)
        y = self.up_convolution_1(torch.cat([down_7, up_0], 1))
        up_1 = self.up_transpose_1(y)
        y = self.up_convolution_2(torch.cat([down_5, up_1], 1))
        up_2 = self.up_transpose_2(y)
        y = self.up_convolution_3(torch.cat([down_3, up_2], 1))
        up_3 = self.up_transpose_3(y)
        y = self.up_convolution_4(torch.cat([down_1, up_3], 1))
        out = self.out(y)
        return out, x

    def encoder(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        out = self.down_flatten(down_7)
        return out


# ======== This network is for data of dimension 100x87 (4 s) =================
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return x * self.sigmoid(x)



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(2, 4), stride=(1, 2), padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 8, kernel_size=(2, 4), stride=(1, 2), padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 8, kernel_size=(2, 4), stride=(1, 2), padding=1),
            nn.ReLU(True),
            SpatialAttentionModule(),
            nn.Flatten(),
            nn.Linear(84, 9),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.encoder(x)




class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(9, 84),  # Further reduction
            nn.ReLU(True),
            nn.Unflatten(1, (1, 7, 12)),
            nn.ConvTranspose2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 8, kernel_size=(2, 4), stride=(1, 2), padding=1, output_padding=(0,1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 8, kernel_size=(2, 4), stride=(1, 2), padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=(2, 4), stride=(1, 2), padding=1,  output_padding=(0,1)),
            nn.ReLU(True),
        )


    def forward(self, x):
        x = self.decoder(x)
        return x


class AEC(nn.Module):
    """
    Description: Autoencoder model; combines encoder and decoder layers.
    Inputs:
        - Input data (spectrograms)
    Outputs:
        - Reconstructed data
        - Latent space data
    """
    def __init__(self):
        super(AEC, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z


def init_weights(m):
    """
    Description: Initializes weights with the Glorot Uniform distribution.
    Inputs:
        - Latent space data
    Outputs:
        - Reconstructed data
    """
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class ClusteringLayer(nn.Module):
    """
    Description: Generates soft cluster assignments using latent features as
    input.
    Arguments:
        - n_clusters: User-defined
        - n_features: Must match output dimension of encoder.
        - alpha: Exponential factor (default: 1.0)
        - weights: Initial values for the cluster centroids
    Inputs:
        Encoded data (output of encoder)
    Outputs:
        Soft cluster assignments
    """
    def __init__(self, n_clusters, n_features=9, alpha=1.0, weights=None):
        super(ClusteringLayer, self).__init__()
        self.n_features = int(n_features)
        self.n_clusters = int(n_clusters)
        self.alpha = alpha
        if weights is None:
            initial_weights = torch.zeros(
                self.n_clusters, self.n_features, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_weights)
        else:
            initial_weights = weights
        self.weights = nn.Parameter(initial_weights)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weights
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x


class DEC(nn.Module):
    """Description: Deep Embedded Clustering Model; combines autoencoder with
    clustering layer to generate end-to-end deep embedded clustering neural
    network model.

    Parameters
    ----------
    n_clusters : int
        Number of clusters

    Returns
    -------
    q : array
        Soft cluster assignments

    x : array
        Reconstructed data

    z : array
        Latent space data
    """
    def __init__(self, n_clusters):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.clustering = ClusteringLayer(self.n_clusters)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        q = self.clustering(z)
        return q, x, z
