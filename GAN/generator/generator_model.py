import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
import math
from torch import Tensor
from torch.nn import Parameter


'''https://github.com/blandocs/Tag2Pix/blob/master/model/pretrained.py'''
'''https://github.com/orashi/AlacGAN/blob/master/models/standard.py'''


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

# Pretrained version
class Selayer(nn.Module):
    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out


class BottleneckX_Origin(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(BottleneckX_Origin, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)

        self.conv2 = nn.Conv2d(planes * 2, planes * 2, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 2)

        self.conv3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.selayer = Selayer(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.selayer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEResNeXt_Origin(nn.Module):
    def __init__(self, block, layers, input_channels=3, cardinality=32, num_classes=1000):
        super(SEResNeXt_Origin, self).__init__()
        self.cardinality = cardinality
        self.inplanes = 64
        self.input_channels = input_channels

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        
        x2 = self.layer1(x1)
        
        x3 = self.layer2(x2)
        
        x4 = self.layer3(x3)
        
        return x1, x2, x3, x4

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Selayer(nn.Module):
    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out
    
class SelayerSpectr(nn.Module):
    def __init__(self, inplanes):
        super(SelayerSpectr, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = SpectralNorm(nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1))
        self.conv2 = SpectralNorm(nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out

class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))
            
        self.selayer = Selayer(out_channels)

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.selayer(bottleneck)
        
        x = self.shortcut.forward(x)
        return x + bottleneck
    
class SpectrResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(SpectrResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = SpectralNorm(nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv_conv = SpectralNorm(nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False))
        self.conv_expand = SpectralNorm(nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))
            
        self.selayer = SelayerSpectr(out_channels)

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.selayer(bottleneck)
        
        x = self.shortcut.forward(x)
        return x + bottleneck
    
class FeatureConv(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super(FeatureConv, self).__init__()

        no_bn = True
        
        seq = []
        seq.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False))
        if not no_bn: seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
        if not no_bn: seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False))
        seq.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(*seq)

    def forward(self, x):
        return self.network(x)
    
class Generator(nn.Module):
    def __init__(self, ngf=64, inplanes=5, out_channels=3):
        super(Generator, self).__init__()

        self.encoder = SEResNeXt_Origin(BottleneckX_Origin, [3, 4, 6, 3], num_classes= 370, input_channels=1)
        
        self.to0 =  self._make_encoder_block_first(inplanes, 32)
        self.to1 = self._make_encoder_block(32, 64)
        self.to2 = self._make_encoder_block(64, 92)
        self.to3 = self._make_encoder_block(92, 128)
        self.to4 = self._make_encoder_block(128, 256)
        
        self.deconv_for_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # output is 64 * 64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # output is 128 * 128
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1, output_padding=0), # output is 256 * 256
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1, output_padding=0), # output is 256 * 256
            nn.Tanh()
        )

        tunnel4 = nn.Sequential(*[ResNeXtBottleneck(512, 512, cardinality=32, dilate=1) for _ in range(20)])

        
        self.tunnel4 = nn.Sequential(nn.Conv2d(1024 + 128, 512, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel4,
                                     nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 64

        depth = 2
        tunnel = [ResNeXtBottleneck(256, 256, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(256, 256, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(256, 256, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(256, 256, cardinality=32, dilate=2),
                   ResNeXtBottleneck(256, 256, cardinality=32, dilate=1)]
        tunnel3 = nn.Sequential(*tunnel)

        self.tunnel3 = nn.Sequential(nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel3,
                                     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 128

        tunnel = [ResNeXtBottleneck(128, 128, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(128, 128, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(128, 128, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(128, 128, cardinality=32, dilate=2),
                   ResNeXtBottleneck(128, 128, cardinality=32, dilate=1)]
        tunnel2 = nn.Sequential(*tunnel)

        self.tunnel2 = nn.Sequential(nn.Conv2d(128 + 256 + 64, 128, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel2,
                                     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        tunnel = [ResNeXtBottleneck(64, 64, cardinality=16, dilate=1)]
        tunnel += [ResNeXtBottleneck(64, 64, cardinality=16, dilate=2)]
        tunnel += [ResNeXtBottleneck(64, 64, cardinality=16, dilate=4)]
        tunnel += [ResNeXtBottleneck(64, 64, cardinality=16, dilate=2),
                   ResNeXtBottleneck(64, 64, cardinality=16, dilate=1)]
        tunnel1 = nn.Sequential(*tunnel)

        self.tunnel1 = nn.Sequential(nn.Conv2d(64 + 32, 64, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel1,
                                     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        self.exit = nn.Sequential(nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(32, out_channels, kernel_size= 1, stride = 1, padding = 0))
        
        
    def _make_encoder_block(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_encoder_block_first(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 1, 1),  # Input shape: (batch_size, inplanes, H, W) -> Output shape: (batch_size, planes, H, W)
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),    # Input shape: (batch_size, planes, H, W) -> Output shape: (batch_size, planes, H, W)
            nn.LeakyReLU(0.2),
        )    
        
    def forward(self, sketch):

        x0 = self.to0(sketch)
        aux_out = self.to1(x0)
        aux_out = self.to2(aux_out)
        aux_out = self.to3(aux_out)
        
        x1, x2, x3, x4 = self.encoder(sketch[:, 0:1])
        
        out = self.tunnel4(torch.cat([x4, aux_out], 1))
        
        
        
        x = self.tunnel3(torch.cat([out, x3], 1))
        
        x = self.tunnel2(torch.cat([x, x2, x1], 1))
        
        
        x = torch.tanh(self.exit(torch.cat([x, x0], 1)))
        
        decoder_output = self.deconv_for_decoder(out)
        x = x*0.5 + 0.5
        decoder_output = decoder_output * 0.5 + 0.5
        
        return x, decoder_output  


class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()
        
        self.generator = Generator()
        
    def forward(self, x, extractor_grad = False):
        fake, guide = self.generator(x)
        return fake, guide