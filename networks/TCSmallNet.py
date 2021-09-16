import torch
import torch.nn as nn
import torch.nn.functional as F

stage_out = {'0.5x': [-1, 24, 48, 96, 192], '1.0x': [-1, 24, 116, 232, 464]}


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride, dilate=1, first=False):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        self.first = first
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2 * dilate
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, dilation=dilate, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if first:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if not self.first:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        else:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    def __init__(self, model_size='0.5x', input_channels=3, stage_out_channels=[-1, 24, 48, 96, 192]):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.stride = [2, 2, 2]
        self.dilate = [1, 1, 1]

        self.model_size = model_size
        self.stage_out_channels = stage_out_channels

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            _stride = self.stride[idxstage]
            _dilate = self.dilate[idxstage]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(
                        ShuffleV2Block(input_channel,
                                       output_channel,
                                       mid_channels=output_channel // 2,
                                       ksize=3,
                                       stride=_stride,
                                       first=True))
                else:
                    self.features.append(
                        ShuffleV2Block(input_channel // 2,
                                       output_channel,
                                       mid_channels=output_channel // 2,
                                       ksize=3,
                                       stride=1,
                                       dilate=_dilate))

                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        self.stage1 = self.features[0:self.stage_repeats[0]]  #4
        self.stage2 = self.features[self.stage_repeats[0]:self.stage_repeats[0] + self.stage_repeats[1]]  #8
        self.stage3 = self.features[self.stage_repeats[0] + self.stage_repeats[1]:self.stage_repeats[0] +
                                    self.stage_repeats[1] + self.stage_repeats[2]]  #16

    def forward(self, x):
        r1 = self.first_conv(x)
        r2 = self.stage1(r1)
        r3 = self.stage2(r2)
        r4 = self.stage3(r3)

        return r1, r2, r3, r4


class SepConvBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(SepConvBlock, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, groups=inplanes, bias=False),
                                  nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True),
                                  nn.Conv2d(inplanes, planes, kernel_size=1, bias=False))

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, low_channels, high_channels, output_channels):
        super(DecoderBlock, self).__init__()
        self.skip_conv = nn.Sequential(
            nn.Conv2d(low_channels, high_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(high_channels // 4),
            nn.ReLU(inplace=True),
        )

        self.last_conv = nn.Sequential(
            SepConvBlock(high_channels + high_channels // 4, output_channels),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_low, x_high):
        x_low = self.skip_conv(x_low)
        x_high = F.interpolate(x_high, size=x_low.size()[2:], mode="bilinear", align_corners=False)
        return self.last_conv(torch.cat((x_high, x_low), dim=1))


class Decoder(nn.Module):
    def __init__(self, mid_channels, num_classes):
        super(Decoder, self).__init__()
        self.decoder3 = DecoderBlock(mid_channels[3], mid_channels[4], mid_channels[3])
        self.decoder2 = DecoderBlock(mid_channels[2], mid_channels[3], mid_channels[2])
        self.decoder1 = DecoderBlock(mid_channels[1], mid_channels[2], mid_channels[1])
        self.final = nn.Conv2d(mid_channels[1], num_classes, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

        self.final2 = nn.Sequential(nn.Conv2d(mid_channels[2], num_classes, kernel_size=3, padding=1, stride=1),
                                    nn.ReLU(inplace=True))

        self.final3 = nn.Sequential(nn.Conv2d(mid_channels[3], num_classes, kernel_size=3, padding=1, stride=1),
                                    nn.ReLU(inplace=True))

    def forward(self, x_enc):
        r1, r2, r3, r4 = x_enc
        d3 = self.decoder3(r3, r4)
        d2 = self.decoder2(r2, d3)
        d1 = self.decoder1(r1, d2)
        p = self.relu(self.final(d1))
        p = F.interpolate(p, scale_factor=2, mode='bilinear', align_corners=False)

        if self.training:
            p2 = self.final2(d2)
            p2 = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)

            p3 = self.final3(d3)
            p3 = F.interpolate(p3, scale_factor=8, mode='bilinear', align_corners=False)

            return p, p2, p3

        return p


class TCSmallNet(nn.Module):
    def __init__(self, args, backbone='0.5x', input_channels=3, num_classes=1):
        super(TCSmallNet, self).__init__()

        mid_channels = stage_out[backbone]
        self.encoder = ShuffleNetV2(backbone, input_channels, mid_channels)
        self.decoder = Decoder(mid_channels, num_classes)

    def forward(self, image):
        image = image / 255.0
        x = self.encoder(image)
        depth = self.decoder(x)

        return depth
