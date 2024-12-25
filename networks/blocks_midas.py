import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import Bottleneck
from torchvision.models import resnet



def _resnext(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnext101_32x8d_wsl(**kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    # model1 =  resnet.resnext101_32x8d(pretrained=True)
    # model2 = _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], **kwargs)
    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], **kwargs)


def _make_resnext101_wsl(args):
    resnet = resnext101_32x8d_wsl()
    # 记载与训练权重
    state_dict = torch.load("./weights/resnext101_32x8d-8ba56ff5.pth", map_location=args.device)
    resnet.load_state_dict(state_dict)
    return _make_resnet_backbone(resnet)


def _make_encoder(backbone, features, args, groups=1, expand=False, exportable=True):
    if backbone == "resnext101_wsl":
        pretrained = _make_resnext101_wsl(args)
        scratch = _make_scratch([256, 512, 1024, 2048], features, groups=groups, expand=expand)
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0],
                                  out_shape1,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False,
                                  groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1],
                                  out_shape2,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False,
                                  groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2],
                                  out_shape3,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False,
                                  groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3],
                                  out_shape4,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False,
                                  groups=groups)

    return scratch


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


class Interpolate(nn.Module):
    """Interpolation module.
    """
    def __init__(self, scale_factor, mode):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """
    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """
    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)

        return output


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module.
    """
    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block.
    """
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)

        return output
