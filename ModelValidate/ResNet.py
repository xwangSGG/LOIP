import torch
import torchvision
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.utils.model_zoo as model_zoo
import math
from torch.hub import load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 kernel
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# get BasicBlock which layers < 50(18, 34)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.BN = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)  # outplane is not in_planes*self.expansion, is planes
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x  # mark the data before BasicBlock
        x = self.conv1(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.BN(x)  # BN operation is before relu operation
        if self.downsample is not None:  # is not None
            residual = self.downsample(residual)  # resize the channel
        x += residual
        x = self.relu(x)
        return x


# get BottleBlock which layers >= 50
class Bottleneck(nn.Module):
    expansion = 4  # the factor of the last layer of BottleBlock and the first layer of it

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.con2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.con2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000
                 ):
        super(ResNet, self).__init__()
        self.inplanes = 64  # the original channel
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 以下构建残差块， 具体参数可以查看resnet参数表
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.average_pool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.block = block
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != block.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, block.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(block.expansion * planes)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample=downsample))  # outplane is planes not planes*block.expansion
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.average_pool(x)
        # x = x.view(x.size(0), -1)  # resize batch-size x H
        # print(x.size())
        # print(512 * self.block.expansion)
        # x = self.fc(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(mode='train',pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_dict = model.state_dict()
        state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
        model.load_state_dict(state_dict, strict=False)

    layers = list(model.children())  # if -2: wu ReLU,Remove Maxpool he ReLU

    for l in layers:
        for p in l.parameters():
            p.requires_grad = False
    if mode == 'cluster':
        layers.append(L2Norm())
    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder', encoder)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        state_dict = load_state_dict_from_url(model_urls['resnet101'],progress=True)
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
        model.load_state_dict(state_dict,strict=False)
    layers = list(model.children())
    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder',encoder)
    return model


def resnet152(mode='train',pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)

        model_dict = model.state_dict()
        state_dict = load_state_dict_from_url(model_urls['resnet152'], progress=True)
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
        model.load_state_dict(state_dict, strict=False)

    layers = list(model.children())  # if -2: wu ReLU,Remove Maxpool he ReLU

    for l in layers:
        for p in l.parameters():
            p.requires_grad = True
    if mode == 'cluster':
        layers.append(L2Norm())
    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder', encoder)

    return model
