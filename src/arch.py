import os
import numpy as np
import glob
import torch
import torchvision
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Linear, MSELoss
from torch.nn import ConvTranspose2d, Conv2d, MaxPool2d, BatchNorm2d
# For our model
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary
class Nothing(nn.Module):
    def __init__(self):
        super(Nothing,self).__init__()
    def forward(self, radar,satellite):
        return radar, satellite

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.up(x)
        return x
class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""
    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """

        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding='same',bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)

            x1 = self.conv(x+x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding='same')

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding='same',bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self,num_channel=1):
        super(Unet, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = ConvBlock(1, 2*num_channel)
        self.Conv2 = ConvBlock(2*num_channel, 4*num_channel)
        self.Conv3 = ConvBlock(4*num_channel, 8*num_channel)
        self.Conv4 = ConvBlock(8*num_channel, 16*num_channel)
        self.Conv5 = ConvBlock(16*num_channel, 32*num_channel)
        self.mid_conv_1 = single_conv(32*num_channel,32*num_channel)
        self.mid_conv_2 = single_conv(1, 32*num_channel)
        self.MidConv = ConvBlock(64*num_channel, 32*num_channel)
        self.out_conv_S = Conv2d(32*num_channel, 1, (1, 1), padding= 'same')
        self.Up5 = UpConv(64*num_channel, 32*num_channel)
        self.UpConv5 = ConvBlock(64*num_channel, 32*num_channel)
        self.Up4 = UpConv(32*num_channel, 16*num_channel)
        self.UpConv4 = ConvBlock(32*num_channel, 16*num_channel)
        self.Up3 = UpConv(16*num_channel, 8*num_channel)
        self.UpConv3 = ConvBlock(16*num_channel, 8*num_channel)
        self.Up2 = UpConv(8*num_channel, 4*num_channel)
        self.UpConv2 = ConvBlock(8*num_channel, 4*num_channel)
        self.Up1 = UpConv(4*num_channel, 2*num_channel)
        self.UpConv1 = ConvBlock(4*num_channel, 2*num_channel)
        self.out_conv_R = Conv2d(2*num_channel, 1, (1, 1), padding= 'same')
    def forward(self, radar,satellite):
        e1 = self.Conv1(radar)
        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)
        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)
        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)
        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)
        e6 = self.MaxPool(e5)
        X = F.relu(self.mid_conv_1(e6))
        Y = F.relu(self.mid_conv_2(satellite))
        X = torch.cat((X,Y),1)
        Y = self.MidConv(X)
        satellite = self.out_conv_S(Y)
        d5 = self.Up5(X)
        d5 = torch.cat((e5, d5), dim=1)
        d5 = self.UpConv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.UpConv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.UpConv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.UpConv2(d2)
        d1 = self.Up1(d2)
        d0 = torch.cat((e1, d1), dim=1)
        d0 = self.UpConv1(d0)
        radar = self.out_conv_R(d0)
        return radar, satellite

class R2Unet(nn.Module):
    def __init__(self,num_channel=1,t=2):
        super(R2Unet, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.RRCNN1 = RRCNN_block(1,2*num_channel,t=t)
        self.RRCNN2 = RRCNN_block(2*num_channel,4*num_channel,t=t)
        self.RRCNN3 = RRCNN_block(4*num_channel,8*num_channel,t=t)
        self.RRCNN4 = RRCNN_block(8*num_channel,16*num_channel,t=t)
        self.RRCNN5 = RRCNN_block(16*num_channel,32*num_channel,t=t)
        self.mid_conv_1 = single_conv(32*num_channel,32*num_channel)
        self.mid_conv_2 = single_conv(1, 32*num_channel)
        self.MidConv = RRCNN_block(64*num_channel, 32*num_channel)
        self.out_conv_S = Conv2d(32*num_channel, 1, (1, 1), padding= 'same')
        self.Up5 = UpConv(64*num_channel, 32*num_channel)
        self.UpRRCNN5 = RRCNN_block(64*num_channel, 32*num_channel)
        self.Up4 = UpConv(32*num_channel, 16*num_channel)
        self.UpRRCNN4 = RRCNN_block(32*num_channel, 16*num_channel)
        self.Up3 = UpConv(16*num_channel, 8*num_channel)
        self.UpRRCNN3 = RRCNN_block(16*num_channel, 8*num_channel)
        self.Up2 = UpConv(8*num_channel, 4*num_channel)
        self.UpRRCNN2 = RRCNN_block(8*num_channel, 4*num_channel)
        self.Up1 = UpConv(4*num_channel, 2*num_channel)
        self.UpRRCNN1 = RRCNN_block(4*num_channel, 2*num_channel)
        self.out_conv_R = Conv2d(2*num_channel, 1, (1, 1), padding= 'same')
    def forward(self, radar,satellite):
        e1 = self.RRCNN1(radar)
        e2 = self.MaxPool(e1)
        e2 = self.RRCNN2(e2)
        e3 = self.MaxPool(e2)
        e3 = self.RRCNN3(e3)
        e4 = self.MaxPool(e3)
        e4 = self.RRCNN4(e4)
        e5 = self.MaxPool(e4)
        e5 = self.RRCNN5(e5)
        e6 = self.MaxPool(e5)
        X = F.relu(self.mid_conv_1(e6))
        Y = F.relu(self.mid_conv_2(satellite))
        X = torch.cat((X,Y),1)
        Y = self.MidConv(X)
        satellite = self.out_conv_S(Y)
        d5 = self.Up5(X)
        d5 = torch.cat((e5, d5), dim=1)
        d5 = self.UpRRCNN5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.UpRRCNN4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.UpRRCNN3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.UpRRCNN2(d2)
        d1 = self.Up1(d2)
        d0 = torch.cat((e1, d1), dim=1)
        d0 = self.UpRRCNN1(d0)
        radar = self.out_conv_R(d0)
        return radar, satellite
class AttUnet(nn.Module):
    def __init__(self,num_channel=1):
        super(AttUnet, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = ConvBlock(1, 2*num_channel)
        self.Conv2 = ConvBlock(2*num_channel, 4*num_channel)
        self.Conv3 = ConvBlock(4*num_channel, 8*num_channel)
        self.Conv4 = ConvBlock(8*num_channel, 16*num_channel)
        self.Conv5 = ConvBlock(16*num_channel, 32*num_channel)
        self.mid_conv_1 = single_conv(32*num_channel,32*num_channel)
        self.mid_conv_2 = single_conv(1, 32*num_channel)
        self.MidConv = ConvBlock(64*num_channel, 32*num_channel)
        self.out_conv_S = Conv2d(32*num_channel, 1, (1, 1), padding= 'same')
        self.Up5 = UpConv(64*num_channel, 32*num_channel)
        self.Att5 = AttentionBlock(F_g=32*num_channel, F_l=32*num_channel, n_coefficients=16*num_channel)
        self.UpConv5 = ConvBlock(64*num_channel, 32*num_channel)
        self.Up4 = UpConv(32*num_channel, 16*num_channel)
        self.Att4 = AttentionBlock(F_g=16*num_channel, F_l=16*num_channel, n_coefficients=8*num_channel)
        self.UpConv4 = ConvBlock(32*num_channel, 16*num_channel)
        self.Up3 = UpConv(16*num_channel, 8*num_channel)
        self.Att3 = AttentionBlock(F_g=8*num_channel, F_l=8*num_channel, n_coefficients=4*num_channel)
        self.UpConv3 = ConvBlock(16*num_channel, 8*num_channel)
        self.Up2 = UpConv(8*num_channel, 4*num_channel)
        self.Att2 = AttentionBlock(F_g=4*num_channel, F_l=4*num_channel, n_coefficients=2*num_channel)
        self.UpConv2 = ConvBlock(8*num_channel, 4*num_channel)
        self.Up1 = UpConv(4*num_channel, 2*num_channel)
        self.Att1 = AttentionBlock(F_g=2*num_channel, F_l=2*num_channel, n_coefficients=1*num_channel)
        self.UpConv1 = ConvBlock(4*num_channel, 2*num_channel)
        self.out_conv_R = Conv2d(2*num_channel, 1, (1, 1), padding= 'same')
    def forward(self, radar,satellite):
        e1 = self.Conv1(radar)
        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)
        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)
        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)
        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)
        e6 = self.MaxPool(e5)
        X = F.relu(self.mid_conv_1(e6))
        Y = F.relu(self.mid_conv_2(satellite))
        X = torch.cat((X,Y),1)
        Y = self.MidConv(X)
        satellite = self.out_conv_S(Y)
        d5 = self.Up5(X)
        s4 = self.Att5(gate=d5, skip_connection=e5)
        d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)
        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e4)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)
        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e3)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)
        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e2)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)
        d1 = self.Up1(d2)
        s0 = self.Att1(gate=d1, skip_connection=e1)
        d0 = torch.cat((s0, d1), dim=1)
        d0 = self.UpConv1(d0)
        radar = self.out_conv_R(d0)
        return radar, satellite
class AttR2Unet(nn.Module):
    def __init__(self,num_channel=1,t=2):
        super(AttR2Unet, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.RRCNN1 = RRCNN_block(1, 2*num_channel)
        self.RRCNN2 = RRCNN_block(2*num_channel, 4*num_channel)
        self.RRCNN3 = RRCNN_block(4*num_channel, 8*num_channel)
        self.RRCNN4 = RRCNN_block(8*num_channel, 16*num_channel)
        self.RRCNN5 = RRCNN_block(16*num_channel, 32*num_channel)
        self.mid_conv_1 = single_conv(32*num_channel,32*num_channel)
        self.mid_conv_2 = single_conv(1, 32*num_channel)
        self.MidConv = RRCNN_block(64*num_channel, 32*num_channel)
        self.out_conv_S = Conv2d(32*num_channel, 1, (1, 1), padding= 'same')
        self.Up5 = UpConv(64*num_channel, 32*num_channel)
        self.Att5 = AttentionBlock(F_g=32*num_channel, F_l=32*num_channel, n_coefficients=16*num_channel)
        self.UpRRCNN5 = RRCNN_block(64*num_channel, 32*num_channel)
        self.Up4 = UpConv(32*num_channel, 16*num_channel)
        self.Att4 = AttentionBlock(F_g=16*num_channel, F_l=16*num_channel, n_coefficients=8*num_channel)
        self.UpRRCNN4 = RRCNN_block(32*num_channel, 16*num_channel)
        self.Up3 = UpConv(16*num_channel, 8*num_channel)
        self.Att3 = AttentionBlock(F_g=8*num_channel, F_l=8*num_channel, n_coefficients=4*num_channel)
        self.UpRRCNN3 = RRCNN_block(16*num_channel, 8*num_channel)
        self.Up2 = UpConv(8*num_channel, 4*num_channel)
        self.Att2 = AttentionBlock(F_g=4*num_channel, F_l=4*num_channel, n_coefficients=2*num_channel)
        self.UpRRCNN2 = RRCNN_block(8*num_channel, 4*num_channel)
        self.Up1 = UpConv(4*num_channel, 2*num_channel)
        self.Att1 = AttentionBlock(F_g=2*num_channel, F_l=2*num_channel, n_coefficients=1*num_channel)
        self.UpRRCNN1 = RRCNN_block(4*num_channel, 2*num_channel)
        self.out_conv_R = Conv2d(2*num_channel, 1, (1, 1), padding= 'same')
    def forward(self, radar,satellite):
        e1 = self.RRCNN1(radar)
        e2 = self.MaxPool(e1)
        e2 = self.RRCNN2(e2)
        e3 = self.MaxPool(e2)
        e3 = self.RRCNN3(e3)
        e4 = self.MaxPool(e3)
        e4 = self.RRCNN4(e4)
        e5 = self.MaxPool(e4)
        e5 = self.RRCNN5(e5)
        e6 = self.MaxPool(e5)
        X = F.relu(self.mid_conv_1(e6))
        Y = F.relu(self.mid_conv_2(satellite))
        X = torch.cat((X,Y),1)
        Y = self.MidConv(X)
        satellite = self.out_conv_S(Y)
        d5 = self.Up5(X)
        s4 = self.Att5(gate=d5, skip_connection=e5)
        d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpRRCNN5(d5)
        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e4)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpRRCNN4(d4)
        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e3)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpRRCNN3(d3)
        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e2)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpRRCNN2(d2)
        d1 = self.Up1(d2)
        s0 = self.Att1(gate=d1, skip_connection=e1)
        d0 = torch.cat((s0, d1), dim=1)
        d0 = self.UpRRCNN1(d0)
        radar = self.out_conv_R(d0)
        return radar, satellite
