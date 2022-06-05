import torch
import torch.nn as nn
import torch.nn.functional as F

bn_mom = 0.2
# ====================basic block====================

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 groups=1, bias=False, activation_mode = True):
        super(ConvBNReLU, self).__init__()
        self.activation_mode = activation_mode
        self.conv = nn.Sequential(nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, groups=groups, bias=bias),
            nn.Dropout(0.1))
        self.bn = nn.BatchNorm2d(out_chan, momentum = bn_mom)
        if activation_mode == True:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        if self.activation_mode == True:
            feat = self.relu(feat)
        return feat

def channel_shuffle(x, groups=2):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class dwconv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation_mode=True):
        super(dwconv, self).__init__()
        self.activation_mode = activation_mode
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                      stride=1, padding=0, groups=1, bias=False),
                                            nn.Dropout(0.1))
        self.bn = nn.BatchNorm2d(out_channels, momentum = bn_mom)
        if activation_mode == True:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        if self.activation_mode == True:
            x = self.activation(x)
        return channel_shuffle(x)

class shuffleblock(nn.Module):

    def __init__(self,in_channels):
        super(shuffleblock, self).__init__()
        self.mid_channels = in_channels // 2
        self.conv1 = ConvBNReLU(in_channels // 2, in_channels // 2, ks=1, stride=1, padding=0)
        self.dwconv = dwconv(in_channels // 2, in_channels // 2, 3, 1, 1)
        self.conv2 = ConvBNReLU(in_channels // 2, in_channels // 2, ks=1, stride=1, padding=0)
    def forward(self, x):
        # split
        x_left = x[:, 0:self.mid_channels, :, :]
        x_right = x[:, self.mid_channels:, :, :]
        # right
        x_right = self.conv1(x_right)
        x_right = self.dwconv(x_right)
        x_right = self.conv2(x_right)
        # concat
        x = torch.cat((x_left, x_right), dim=1)
        # shuffle
        x = channel_shuffle(x)
        return x

class shuffleblock_downsample(nn.Module):

    def __init__(self,in_channels):
        super(shuffleblock_downsample, self).__init__()
        self.in_channels = in_channels
        # right
        self.conv1 = ConvBNReLU(in_channels, in_channels, ks=1, stride=1, padding=0)
        self.dwconv1 = dwconv(in_channels, in_channels, 3, 2, 1)
        self.conv2 = ConvBNReLU(in_channels, in_channels, ks=1, stride=1, padding=0)
        # left
        self.dwconv2 = dwconv(in_channels, in_channels, 3, 2, 1)
        self.conv3 = ConvBNReLU(in_channels, in_channels, ks=1, stride=1, padding=0)
    def forward(self, x):
        # left
        x1 = x.clone()
        x1 = self.dwconv2(x1)
        x1 = self.conv3(x1)
        # right
        x_right = self.conv1(x)
        x_right = self.dwconv1(x_right)
        x_right = self.conv2(x_right)
        # concat
        x = torch.cat((x1, x_right), dim=1)
        # shuffle
        x = channel_shuffle(x)
        return x




# ====================basic layer====================

class StemBlock(nn.Module):

    def __init__(self, out_channels):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, out_channels, 3, stride=2, padding=1)
        self.left = nn.Sequential(
            ConvBNReLU(out_channels, out_channels // 2, ks=1, stride=1, padding=0),
            ConvBNReLU(out_channels // 2, out_channels, ks=3, stride=2, padding=1),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(out_channels * 2, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class CEBlock(nn.Module):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(256, momentum = bn_mom)
        self.conv_gap = ConvBNReLU(256, 256, 1, stride=1, padding=0)
        self.conv_last = nn.Conv2d(256, 256, 3, stride=1, padding=1)

    def forward(self, x):
        feat = nn.AdaptiveAvgPool2d(1)(x)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1, padding=1)
        self.dwconv1 = dwconv(in_chan, mid_chan, 3, 1, 1)
        self.conv2 = ConvBNReLU(mid_chan, out_chan, 1, stride=1, padding=0, activation_mode=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1, padding=1)
        self.dwconv1 = dwconv(in_chan, mid_chan, 3, 2, 1, activation_mode = False)
        self.dwconv2 = dwconv(mid_chan, mid_chan, 3, 1, 1, activation_mode = False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.Dropout(0.1),
            nn.BatchNorm2d(out_chan, momentum = bn_mom)
        )
        self.shortcut = nn.Sequential(dwconv(in_chan, in_chan, 3, 2, 1, activation_mode = False),
                                      ConvBNReLU(in_chan, out_chan, activation_mode = False))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat



class BGALayer(nn.Module):

    def __init__(self, mode, channels, ratio):
        super(BGALayer, self).__init__()
        self.mode = mode
        self.left1 = nn.Sequential(dwconv(channels, channels, 3, 1, 1, activation_mode=False),
                                   nn.Conv2d(channels, channels, kernel_size=1, stride=1,padding=0, bias=False),
                                   nn.Dropout(0.1))
        self.left2 = nn.Sequential(ConvBNReLU(channels, channels, 3, 2, 1, activation_mode=False),
                                   nn.Sequential(*[nn.AvgPool2d(kernel_size=3, stride=2, padding=1) for _ in range(ratio-1)]))
        self.right1 = nn.Sequential(ConvBNReLU(channels, channels, 3, 1, 1, activation_mode=False),
                                    nn.Sequential(*[nn.Upsample(scale_factor=2, mode="bilinear")for _ in range(ratio)]))
        self.right2 = nn.Sequential(dwconv(channels, channels, 3, 1, 1, activation_mode=False),
                                    nn.Conv2d(channels, channels, kernel_size=1, stride=1,padding=0, bias=False),
                                    nn.Dropout(0.1))
        if self.mode == "mix":
            self.up = nn.Sequential(*[nn.Upsample(scale_factor=2, mode="bilinear")
                                    for _ in range(ratio)])
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                                  padding=1, bias=False)

    def forward(self, x_d, x_s):
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        if self.mode == "split":
            return left, right # left: detail; right: semantic
        elif self.mode == "mix":
            right = self.up(right)
            out = self.conv(left + right)
            return out



class SegmentHead(nn.Module):

    def __init__(self, inplanes, outplanes, ratio): #, scale_factor=None):
        super(SegmentHead, self).__init__()
        self.ratio = ratio
        modulelist = []
        for i in range(ratio):
            modulelist.append(nn.Sequential(dwconv(inplanes, inplanes, 3, 1, 1),
                                            dwconv(inplanes, inplanes // 2, 1, 1, 0),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)))
            inplanes //= 2
        self.upsample = nn.Sequential(*modulelist)
        self.output = nn.Sequential(dwconv(inplanes, inplanes, 3, 1, 1),
                                    nn.Conv2d(inplanes, outplanes, 1, 1, 0))

    def forward(self, x):
        x = self.upsample(x)
        return self.output(x)

class STDC(nn.Module):
    def __init__(self, in_channels, out_channels, mode=None):
        super(STDC, self).__init__()
        self.block1 = dwconv(in_channels = in_channels, out_channels = out_channels // 2,
                                 kernel_size = 1, stride = 1, padding = 0)
        self.mode = mode
        if mode == "downsample":
            self.block2 = dwconv(in_channels = out_channels // 2, out_channels = out_channels // 4,
                                     kernel_size = 3, stride = 2, padding = 1)
        else:
            self.block2 = dwconv(in_channels=out_channels // 2, out_channels=out_channels // 4,
                                 kernel_size=3, stride=1, padding=1)
        self.block3 = dwconv(in_channels = out_channels // 4, out_channels = out_channels // 8,
                                 kernel_size = 3, stride = 1, padding = 1)
        self.block4 = dwconv(in_channels=out_channels // 8, out_channels=out_channels // 8,
                                 kernel_size = 3, stride = 1, padding = 1)
        if mode == "downsample":
            self.avgpool = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1)
    def forward(self, x):
        x2 = self.block1(x)
        x4 = self.block2(x2)
        x8 = self.block3(x4)
        x8_2 = self.block4(x8)
        if self.mode == "downsample":
            x2 = self.avgpool(x2)
        return torch.cat((x8_2, x8, x4, x2), dim = 1)

# ====================main model====================

class model_one(nn.Module):

    def __init__(self, n_classes=2, aux_mode="train"):
        super(model_one, self).__init__()
        self.mode = aux_mode

        # STDC modules to keep high resolution
        self.stdc = nn.Sequential(STDC(3, 64, "downsample"),
                                  STDC(64, 128),
                                  STDC(128, 128))

        # 1st stage: detail 2x, semantic 4x
        self.d1 = nn.Sequential(ConvBNReLU(3, 64, 3, 2, 1),
                                shuffleblock(64))
        self.stem = StemBlock(32)

        # 2nd stage: detail 4x, semantic 8x
        self.d2 = nn.Sequential(shuffleblock_downsample(64),
                                shuffleblock(128),
                                shuffleblock(128))
        self.s2 = nn.Sequential(GELayerS2(32, 64),
                                GELayerS1(64, 64))

        # 3rd stage: semantic 16x & bga
        self.s3 = nn.Sequential(GELayerS2(64, 128),
                                GELayerS1(128, 128))
        self.bga3 = BGALayer("split", 128, 2)

        # 4th stage: detail 4x, semantic 32x
        self.d4 = nn.Sequential(dwconv(128, 256, kernel_size=3, stride=1, padding=1),
                                dwconv(256, 256, kernel_size=3, stride=1, padding=1))
        self.s4 = nn.Sequential(GELayerS2(128, 256),
                                GELayerS1(256, 256),
                                GELayerS1(256, 256),
                                GELayerS1(256, 256))
        self.ceb = CEBlock()
        # 5th stage: detail 8x, semantic 32x(contextual)
        self.bag5 = BGALayer("mix", 256, 3)
        self.final_head1 = nn.Sequential(dwconv(256, 256, 3, 1, 1),
                                         dwconv(256, 128, 1, 1, 0),
                                         nn.Upsample(scale_factor=2, mode='bilinear'))
        self.final_head2 = nn.Sequential(dwconv(256, 256, 3, 1, 1),
                                         dwconv(256, 128, 1, 1, 0),
                                         nn.Upsample(scale_factor=2, mode='bilinear'))
        # segment head
        if aux_mode == "train":
            self.detail_head = SegmentHead(256, 1, 2)
            self.semantic_head = SegmentHead(256, 2, 5)

        # mix
        self.mix = nn.Sequential(dwconv(128, 64, 3, 1, 1),
                                 dwconv(64, 64, 3, 1, 1),
                                 nn.Conv2d(64, 2, 3, 1, 1))


        self.init_weights()

    def forward(self, x):
        # STDC: high resolution to keep details
        hrdetail = self.stdc(x)

        # 1st stage
        detail1 = self.d1(x) # change the channels 3 to 64, size unchanged
        semantic1 = self.stem(x)

        # 2nd stage
        detail2 = self.d2(detail1)
        semantic2 = self.s2(semantic1)

        # 3rd stage
        semantic3 = self.s3(semantic2)
        detail2, semantic3 = self.bga3(detail2, semantic3)

        # 4th stage
        detail4 = self.d4(detail2)
        semantic4_1 = self.s4(semantic3)
        semantic4 = self.ceb(semantic4_1)

        # 5th stage
        feat = self.bag5(detail4, semantic4)

        # final
        feat = self.final_head1(feat)
        feat = torch.cat([feat, hrdetail], dim = 1) # channels:256
        feat = self.final_head2(feat)
        feat = self.mix(feat)
        # segmentation head

        if self.mode == "train":

            detail4 = self.detail_head(detail4)
            semantic4_1 = self.semantic_head(semantic4_1)

            return detail4, semantic4_1, feat
        else:
            return feat
        # final

    def init_weights(self):
        for name, module in self.named_modules():
             if isinstance(module, (nn.Conv2d, nn.Linear)):
                 nn.init.kaiming_normal_(module.weight, mode='fan_out')
                 if not module.bias is None: nn.init.constant_(module.bias, 0)
             elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                 if hasattr(module, 'last_bn') and module.last_bn:
                     nn.init.zeros_(module.weight)
                 else:
                     nn.init.ones_(module.weight)
                 nn.init.zeros_(module.bias)

