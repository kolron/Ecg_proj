import torch
import torch.nn as nn
import math


# A simple but versatile d2 convolutional neural net
class ConvNet2d(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list,
                 kernel_sizes: list, dropout=None, stride=1, dilation=1, batch_norm=False):
        super().__init__()
        assert len(hidden_channels) == len(kernel_sizes)

        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):

            layer_out_channels = hidden_channels[i]
            layers.append(nn.Conv2d(layer_in_channels, layer_out_channels, kernel_size=kernel_sizes[i],
                                    stride=stride, dilation=dilation))
            if batch_norm:
                layers.append(nn.BatchNorm2d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())

            layer_in_channels = layer_out_channels

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# A simple but versatile d1 "deconvolution" neural net
class DeConvNet1d(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list, out_channels: int, out_kernel: int,
                 kernel_lengths: list, dropout=None, stride=1, dilation=1, batch_norm=False, output_padding=1):
        super().__init__()
        assert len(hidden_channels) == len(kernel_lengths)

        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):

            layer_out_channels = hidden_channels[i]
            layers.append(nn.ConvTranspose1d(layer_in_channels, layer_out_channels, kernel_size=kernel_lengths[i],
                                             stride=stride, dilation=dilation, output_padding=output_padding))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.2))

            layer_in_channels = layer_out_channels

        layers.append(nn.ConvTranspose1d(layer_in_channels, out_channels, out_kernel, stride, dilation))

        self.dcnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dcnn(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Ecg12ImageNet(nn.Module):
    """
        This is our model for the image NN, Currently only for Binary classification for AFIB
    """
    def __init__(self, in_channels: int, hidden_channels: list, kernel_sizes: list, in_h: int, in_w: int,
                 fc_hidden_dims: list, dropout=None, stride=1, dilation=1, batch_norm=False, num_of_classes=2):
        super().__init__()
        assert len(hidden_channels) == len(kernel_sizes)

        self.cnn = ConvNet2d(in_channels, hidden_channels, kernel_sizes, dropout, stride, dilation, batch_norm)

        out_channels = hidden_channels[-1]
        out_h = calc_out_length(in_h, kernel_sizes, stride, dilation)
        out_w = calc_out_length(in_w, kernel_sizes, stride, dilation)
        in_dim = out_channels * out_h * out_w

        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        # single score for binary classification, class score for multi-class
        if num_of_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_of_classes))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        out = self.cnn(x)
        out = out.reshape((x.shape[0], -1))
        return self.fc(out)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def calc_out_length(l_in: int, kernel_lengths: list, stride: int, dilation: int, padding=0):
    l_out = l_in
    for kernel in kernel_lengths:
        l_out = math.floor((l_out + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
    return l_out


def calc_out_length_deconv(l_in: int, kernel_lengths: list, out_kernel: int, stride: int, dilation: int,
                           padding=0, output_padding=0):
    l_out = l_in
    for kernel in kernel_lengths:
        l_out = (l_out - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    l_out = (l_out - 1) * stride - 2 * padding + dilation * (out_kernel - 1) + 1
    return l_out
