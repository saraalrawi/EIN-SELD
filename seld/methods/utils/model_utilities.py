import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer, nonlinearity='leaky_relu'):
    '''
    Initialize a layer
    '''
    classname = layer.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                dilation=1, bias=False):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=out_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.init_weights()
        
    def init_weights(self):
        for layer in self.double_conv:
            init_layer(layer)
        
    def forward(self, x):
        x = self.double_conv(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, pos_len, d_model=512, pe_type='t', dropout=0.0):
        """ Positional encoding using sin and cos

        Args:
            pos_len: positional length
            d_model: number of feature maps
            pe_type: 't' | 'f' , time domain, frequency domain
            dropout: dropout probability
        """
        super().__init__()
        
        self.pe_type = pe_type
        pe = torch.zeros(pos_len, d_model)
        pos = torch.arange(0, pos_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = 0.1 * torch.sin(pos * div_term)
        pe[:, 1::2] = 0.1 * torch.cos(pos * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2) # (N, C, T)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # x is (N, C, T, F) or (N, C, T) or (N, C, F)
        if x.ndim == 4:
            if self.pe_type == 't':
                pe = self.pe.unsqueeze(3)
                x += pe[:, :, :x.shape[2]]
            elif self.pe_type == 'f':
                pe = self.pe.unsqueeze(2)
                x += pe[:, :, :, :x.shape[3]]
        elif x.ndim == 3:
            x += self.pe[:, :, :x.shape[2]]
        return self.dropout(x)

class Deconv(nn.Module):
    def __init__(self, inplanes, planes):
        super(Deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(inplanes, planes, 3, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x, is_last=False):
        x = self.deconv(x)
        if not is_last:
            x = self.bn(x)
            x = self.relu(x)
        return x

# Equivelent to function residual_unit in
# https://github.com/deontaepharr/Residual-Attention-Network/blob/master/Code/ResidualAttentionNetwork.py
class BottleNeck(nn.Module):

    def __init__(self, inplanes, planes):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.down_sampler = Downsampler()

    def forward(self, x):
        #x = self.conv4(x)
        residual = x
        # Layer #1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # Layer #2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Layer #3
        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class Downsampler(nn.Module):
    def __init__(self):
        super(Downsampler, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        out, index = self.pool1(x)
        return out

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class AttentionModule(nn.Module):
    def __init__(self, inplanes, planes):
        super(AttentionModule, self).__init__()
        self.bottleneck1_1 = BottleNeck(inplanes, planes //4)
        #self.bottleneck1_2 = BottleNeck(inplanes, planes //4)
        self.downsampler1 = Downsampler()
        self.bottleneck2_1 = BottleNeck(inplanes, planes //4)
        self.downsampler2 = Downsampler()
        self.bottleneck2_2 = BottleNeck(inplanes, planes //4)
        #self.bottleneck2_3 = BottleNeck(inplanes, planes //4)
        self.deconv1 = Deconv(inplanes, planes)
        self.bottleneck2_4 = BottleNeck(inplanes, planes //4)
        self.deconv2 = Deconv(inplanes, planes)
        #self.conv2_1 = nn.Conv2d(inplanes, planes, 1)
        #self.conv2_2 = nn.Conv2d(inplanes, planes, 1)
        self.conv_down = nn.Conv2d(inplanes, inplanes //2 ,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # trunk branch
        x_1 = self.bottleneck1_1(x)
        #x_1 = self.bottleneck1_2(x_1)

        # mask branch
        x_2 = self.downsampler1(x)
        # Perform residual units r
        x_2 = self.bottleneck2_1(x_2)
        x_2 = self.downsampler2(x_2)
        # Perform Middle Residuals - Perform 2*r
        x_2 = self.bottleneck2_2(x_2)
        #x_2 = self.bottleneck2_3(x_2)
        # Upsampling Step Initialization - Top
        x_2 = self.deconv1(x_2)
        x_2 = self.bottleneck2_4(x_2)
        # Last interpolation step - Bottom
        x_2 = self.deconv2(x_2)
        # Conv 1
        #x_2 = self.conv2_1(x_2)
        #  Conv 2
        #x_2 = self.conv2_2(x_2)
        # Sigmoid
        x_2 = self.sigmoid(x_2)


        #x = x_1 * x_2
        #x = x + x_1
        # x_1
        x = self.attention_residual_learning(x_1, x_2)
        x = self.conv_down(x)
        return  x



    def attention_residual_learning(self, mask_input, trunk_input):
        # https://stackoverflow.com/a/53361303/9221241
        Mx = LambdaLayer(lambda x: 1 + x)(mask_input) # 1 + mask
        return (Mx * trunk_input) # M(x) * T(x)

# Github: https://github.com/ai-med/squeeze_and_excitation/
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=4):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv_reduce = nn.Conv2d(num_channels, num_channels//2, kernel_size=1, bias=False)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        output_tensor = self.conv_reduce(output_tensor)
        return output_tensor


