import torch
from torch import nn


def conv3x3(in_channels, out_channels, stride=1, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=3, stride=stride, padding=dilation,
                     dilation=dilation, groups=groups, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=1, stride=stride, bias=False)


class Building_Block(nn.Module):
    expansion = 1

    def __init__(self, input_planes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64') # noqa

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock") # noqa

        self.conv1 = conv3x3(input_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.downsample is not None:
            identity = self.downsample(x)
        y += identity
        y = self.relu(y)
        return y


class BottleNeck(nn.Module):
    # parameter to define the number of block in a building block
    # we need this equal to 4 because after pooling the dim will be
    # half of it, so that the channels will be doubled
    expansion = 4

    def __init__(self, input_planes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is not None:
            self.norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width/64.)) * groups
        self.conv1 = conv1x1(in_channels=input_planes,
                             out_channels=planes)
        self.bn1 = self.norm_layer(width)
        self.conv2 = conv3x3(in_channels=width, out_channels=width,
                             stride=stride, groups=groups, dilation=dilation)
        self.bn2 = self.norm_layer(width)
        self.conv3 = conv1x1(in_channels=width,
                             out_channels=planes*self.expansion)
        self.bn3 = self.norm_layer(planes*self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # First layer
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        # Second layer
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        # Last layer
        y = self.conv3(y)
        y = self.bn3(y)

        if self.downsample is not None:
            identity = self.downsample(identity)
        y += identity
        y = self.relu(y)
        return y


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))  # noqa

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512*block.expansion,
                            out_features=num_classes)

        # Initial value for parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero initialize the last batch norm in each residual branch
        # (building_block)
        # so that the residual branch starts with zeros,
        # and each res block behaves like an identity
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, Building_Block):
                    nn.init.constant_(m.bn2.weight, 0)

    # Function to create a layer
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width,
                            previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # Layer after input before res block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Fully-connected and output layer
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block=block, layers=layers, **kwargs)
    return model


def resnet18(**kwargs):
    return _resnet("resnet18", Building_Block, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return _resnet("resnet34", Building_Block, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return _resnet("resnet50", BottleNeck, [3, 4, 6, 3], **kwargs)
