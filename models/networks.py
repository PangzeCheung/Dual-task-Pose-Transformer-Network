import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from .base_function import *
from .PTM import PTM


###############################################################################
# Functions
###############################################################################
def define_G(opt, image_nc, pose_nc, ngf=64, img_f=1024, encoder_layer=3, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False, output_nc=3, num_blocks=3, affine=True, nhead=2, num_CABs=2, num_TTBs=2):
    print(opt.model)
    if opt.model == 'DPTN':
        netG = DPTNGenerator(image_nc, pose_nc, ngf, img_f, encoder_layer, norm, activation, use_spect, use_coord, output_nc, num_blocks, affine, nhead, num_CABs, num_TTBs)
    else:
        raise('generator not implemented!')
    return init_net(netG, opt.init_type, opt.gpu_ids)


def define_D(opt, input_nc=3, ndf=64, img_f=1024, layers=3, norm='none', activation='LeakyReLU', use_spect=True,):
    netD = ResDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect)
    return init_net(netD, opt.init_type, opt.gpu_ids)


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Generator
##############################################################################
class SourceEncoder(nn.Module):
    """
    Source Image Encoder (En_s)
    :param image_nc: number of channels in input image
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param encoder_layer: encoder layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    """
    def __init__(self, image_nc, ngf=64, img_f=1024, encoder_layer=3, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False):
        super(SourceEncoder, self).__init__()

        self.encoder_layer = encoder_layer

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        input_nc = image_nc

        self.block0 = EncoderBlockOptimized(input_nc, ngf, norm_layer,
                                   nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(encoder_layer - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = EncoderBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

    def forward(self, source):
        inputs = source
        out = self.block0(inputs)
        for i in range(self.encoder_layer - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        return out


class DPTNGenerator(nn.Module):
    """
    Dual-task Pose Transformer Network (DPTN)
    :param image_nc: number of channels in input image
    :param pose_nc: number of channels in input pose
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    :param output_nc: number of channels in output image
    :param num_blocks: number of ResBlocks
    :param affine: affine in Pose Transformer Module
    :param nhead: number of heads in attention module
    :param num_CABs: number of CABs
    :param num_TTBs: number of TTBs
    """
    def __init__(self, image_nc, pose_nc, ngf=64, img_f=256, layers=3, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False, output_nc=3, num_blocks=3, affine=True, nhead=2, num_CABs=2, num_TTBs=2):
        super(DPTNGenerator, self).__init__()

        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        input_nc = 2 * pose_nc + image_nc

        # Encoder En_c
        self.block0 = EncoderBlockOptimized(input_nc, ngf, norm_layer,
                                   nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(self.layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = EncoderBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        # ResBlocks
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            block = ResBlock(ngf * mult, ngf * mult, norm_layer=norm_layer,
                             nonlinearity=nonlinearity, use_spect=use_spect, use_coord=use_coord)
            setattr(self, 'mblock' + str(i), block)

        # Pose Transformer Module (PTM)
        self.PTM = PTM(d_model=ngf * mult, nhead=nhead, num_CABs=num_CABs,
                 num_TTBs=num_TTBs, dim_feedforward=ngf * mult,
                 activation="LeakyReLU", affine=affine, norm=norm)

        # Encoder En_s
        self.source_encoder = SourceEncoder(image_nc, ngf, img_f, layers, norm, activation, use_spect, use_coord)

        # Decoder
        for i in range(self.layers):
            mult_prev = mult
            mult = min(2 ** (self.layers - i - 2), img_f // ngf) if i != self.layers - 1 else 1
            up = ResBlockDecoder(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), up)
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)

    def forward(self, source, source_B, target_B, is_train=True):
        # Self-reconstruction Branch
        # Source-to-source Inputs
        input_s_s = torch.cat((source, source_B, source_B), 1)
        # Source-to-source Encoder
        F_s_s = self.block0(input_s_s)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            F_s_s = model(F_s_s)
        # Source-to-source Resblocks
        for i in range(self.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            F_s_s = model(F_s_s)

        # Transformation Branch
        # Source-to-target Inputs
        input_s_t = torch.cat((source, source_B, target_B), 1)
        # Source-to-target Encoder
        F_s_t = self.block0(input_s_t)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            F_s_t = model(F_s_t)
        # Source-to-target Resblocks
        for i in range(self.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            F_s_t = model(F_s_t)

        # Source Image Encoding
        F_s = self.source_encoder(source)

        # Pose Transformer Module for Dual-task Correlation
        F_s_t = self.PTM(F_s_s, F_s_t, F_s)

        # Source-to-source Decoder (only for training)
        out_image_s = None
        if is_train:
            for i in range(self.layers):
                model = getattr(self, 'decoder' + str(i))
                F_s_s = model(F_s_s)
            out_image_s = self.outconv(F_s_s)

        # Source-to-target Decoder
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            F_s_t = model(F_s_t)
        out_image_t = self.outconv(F_s_t)

        return out_image_t, out_image_s


##############################################################################
# Discriminator
##############################################################################
class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    """
    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=3, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False):
        super(ResDiscriminator, self).__init__()

        self.layers = layers

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ndf, ndf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ndf)
            block = ResBlockEncoder(ndf*mult_prev, ndf*mult, ndf*mult_prev, norm_layer, nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)
        self.conv = SpectralNorm(nn.Conv2d(ndf*mult, 1, 1))

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.conv(self.nonlinearity(out))
        return out