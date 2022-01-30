from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class ABN(nn.Sequential):
    def __init__(self, num_features, activation=nn.LeakyReLU(0.1)):
        super(ABN, self).__init__(OrderedDict([
            ("bn", nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)),
            ("act", activation)
        ]))


class CAB(nn.Module):

    def __init__(self, high_channels, low_channels):
        super(CAB, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(low_channels, high_channels, kernel_size=1),
            ABN(high_channels)
        )
        self.cat = low_channels + high_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(self.cat, int(self.cat // 16), 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(int(self.cat // 16), high_channels, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_h, x_l = x  # high, low
        x_l_1 = self.conv0(x_l)
        x = torch.cat([x_h, x_l], dim=1)
        bahs, chs, _, _ = x.size()
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        attention = self.sigmoid(avg_out + max_out)
        x_l_2 = attention * x_l_1
        return x_h + x_l_2


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class RFBlock(nn.Module):
    def __init__(self, in_chs, out_chs, setting=[3, 6, 12, 18], norm_act=ABN):
        super(RFBlock, self).__init__()

        self.down_chs = nn.Sequential(
            norm_act(in_chs),
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.spatial = nn.Sequential(
            norm_act(out_chs),
            nn.Conv2d(out_chs, out_chs, kernel_size=(3, 3), stride=1, padding=(1, 1), groups=out_chs, bias=False),
            norm_act(out_chs),
            nn.Conv2d(out_chs, 80, kernel_size=(1, 1), stride=1, bias=False)
        )

        self.gap = nn.Sequential(
            norm_act(out_chs),
            nn.AdaptiveAvgPool2d((1, 1)),
            #                               nn.Conv2d(out_chs, out_chs, kernel_size=1, bias=False),
        )

        self.features = []

        for d in setting:
            self.features.append(nn.Sequential(
                norm_act(out_chs),
                nn.Conv2d(out_chs, out_chs, kernel_size=(3, 1), stride=1, padding=(d, 0), dilation=(d, 1),
                          groups=out_chs, bias=False),
                norm_act(out_chs),
                nn.Conv2d(out_chs, out_chs, kernel_size=(1, 3), stride=1, padding=(0, d), dilation=(1, d),
                          groups=out_chs, bias=False),
                norm_act(out_chs),
                nn.Conv2d(out_chs, out_chs, kernel_size=(1, 1), stride=1, bias=False)
            ))

        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        down = self.down_chs(x)
        spatial = self.spatial(down)
        gap = self.gap(down)
        gap = F.upsample(gap, x_size[2:], mode='bilinear')
        out = []
        for f in self.features:
            out.append(f(down.clone()))
        out = torch.cat(out, 1)
        out = torch.cat([out, gap, spatial], 1)
        return out