# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn

from networks.blocks_midas import FeatureFusionBlock
from networks.blocks_midas import Interpolate
from networks.blocks_midas import _make_encoder


class MidasNet(torch.nn.Module):

    def __init__(self, args, path=None, features=256, non_negative=True):
        """Init.

        Args:
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnext101_wsl
        """

        super(MidasNet, self).__init__()

        self.__mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        self.__mean = self.__mean.view(1, 3, 1, 1)
        self.__std = torch.tensor([0.229, 0.224, 0.225]).cuda()
        self.__std = self.__std.view(1, 3, 1, 1)

        self.pretrained, self.scratch = _make_encoder("resnext101_wsl", features)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

    def forward(self, image):
        image = image / 255.0
        image = (image - self.__mean) / self.__std

        layer_1 = self.pretrained.layer1(image)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out
