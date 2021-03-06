from torch import nn
import math
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

import logging

from collections import OrderedDict

from torchlab.layer.dilated import space2batch, batch2space


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'BasicBlock', 'Bottleneck']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, first_dilation=1, bn=None, use_s2b=False): # NOQA
        super(BasicBlock, self).__init__()

        assert not use_s2b
        self.BatchNorm = bn
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = self.BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=first_dilation, dilation=first_dilation, bias=False) # NOQA
        self.bn2 = self.BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, first_dilation=1, use_s2b=False,
                 bn=None):
        self.BatchNorm = bn
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = self.BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = self.BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = self.BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.space2batch = use_s2b

    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):

        if self.space2batch:
            if type(x) == list or type(x) == tuple:
                x = [space2batch(inp) for inp in x]
            else:
                x = space2batch(x)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if isinstance(out, Variable):
            out += residual
        elif isinstance(out, tuple) or isinstance(out, list):
            out = self._sum_each(out, residual)

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featu

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition."

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by
    """

    def __init__(self, block, layers, num_classes=1000, dilated=True,
                 batched_dilation=False, bn=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.BatchNorm = bn
        self.bn1 = self.BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.dilated = dilated

        self.bdilation = batched_dilation

        if not dilated:
            assert not batched_dilation

        if batched_dilation:
            dilation = 1
            stride = 1
        elif dilated:
            dilation = 2
            stride = 1
        else:
            dilation = 1
            stride = 2

        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=stride, dilation=dilation,
            use_s2b=batched_dilation)

        if batched_dilation:
            dilation = 1
            stride = 1
        elif dilated:
            dilation = 4
            stride = 1
        else:
            dilation = 1
            stride = 2

        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=stride, dilation=dilation,
            use_s2b=batched_dilation)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, self.BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_channel_dict(self):

        bn4 = [m for m in self.layer4.modules()][-2]
        if not isinstance(bn4, self.BatchNorm):
            bn4 = [m for m in self.layer4.modules()][-1]
        assert(isinstance(bn4, self.BatchNorm))
        bn3 = [m for m in self.layer3.modules()][-2]
        if not isinstance(bn3, self.BatchNorm):
            bn3 = [m for m in self.layer4.modules()][-1]
        assert(isinstance(bn3, self.BatchNorm))
        bn2 = [m for m in self.layer2.modules()][-2]
        if not isinstance(bn2, self.BatchNorm):
            bn2 = [m for m in self.layer4.modules()][-1]
        assert(isinstance(bn2, self.BatchNorm))

        if self.dilated:
            df = 8
        else:
            df = 32

        chan_dict = {
            'scale8': bn2.num_features,
            'scale16': bn3.num_features,
            'scale32': bn4.num_features,
            'down_factor': df}

        return chan_dict

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    use_s2b=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.BatchNorm(planes * block.expansion),
            )

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample,
                                first_dilation=dilation, use_s2b=use_s2b,
                                bn=self.BatchNorm))
        elif dilation == 4:
            assert not use_s2b
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample,
                                first_dilation=dilation,
                                bn=self.BatchNorm))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation=dilation, first_dilation=dilation,
                                bn=self.BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, x, verbose=False, return_dict=False):

        wrap = False
        if type(x) is tuple and len(x) == 1:
            x = x[0][0]
            wrap = True

        if isinstance(x, Variable):

            resdict = OrderedDict()

            resdict['image'] = x

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            # Main ResNet Blocks
            x = self.layer1(x)
            if verbose:
                logging.info("{:>7} : {}".format("layer1", x.shape))

            x = self.layer2(x)
            resdict['scale8'] = x
            if verbose:
                logging.info("{:>7} : {}".format("layer2", x.shape))

            x = self.layer3(x)
            resdict['scale16'] = x
            if verbose:
                logging.info("{:>7} : {}".format("layer3", x.shape))

            x = self.layer4(x)

            if self.bdilation:
                x = batch2space(batch2space(x))

            resdict['scale32'] = x
            if verbose:
                logging.info("{:>7} : {}".format("layer4", x.shape))

            if wrap:
                resdict = [resdict]
                x = [x]

            # return all intermediate results
            if return_dict:
                return resdict
            else:
                return x

        elif isinstance(x, tuple) or isinstance(x, list):

            resdicts = [OrderedDict() for i in range(len(x))]

            for x_i, resdict in zip(x, resdicts):
                resdict['image'] = x_i

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            # Main ResNet Blocks
            x = self.layer1(x)
            if verbose:
                logging.info("{:>7} : {}".format("layer1", x[0].shape))

            x = self.layer2(x)

            for x_i, resdict in zip(x, resdicts):
                resdict['scale8'] = x_i

            if verbose:
                logging.info("{:>7} : {}".format("layer2", x[0].shape))

            x = self.layer3(x)
            for x_i, resdict in zip(x, resdicts):
                resdict['scale16'] = x_i
            if verbose:
                logging.info("{:>7} : {}".format("layer3", x[0].shape))

            x = self.layer4(x)

            if self.bdilation:
                x = [batch2space(batch2space(inp)) for inp in x]

            for x_i, resdict in zip(x, resdicts):
                resdict['scale32'] = x_i
            if verbose:
                logging.info("{:>7} : {}".format("layer4", x[0].shape))

            # return all intermediate results
            if return_dict:
                return resdicts
            else:
                return x

        else:
            raise RuntimeError('unknown input type')

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
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


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

if __name__ == "__main__":
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    print(model.layer4)
