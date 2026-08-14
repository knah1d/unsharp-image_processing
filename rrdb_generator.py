"""
RRDBNet generator (ESRGAN / Real-ESRGAN architecture), for the transfer-learning
alternative to the paper-faithful SRGANGenerator in enhance.py.

This is a SEPARATE, additive path -- enhance.py's SRGANGenerator and the
paper-faithful training recipe in train_srgan.py are untouched. The paper
specifies "16 residual blocks" (Table 4) and a specific 4-layer SRGAN
generator (Section 3.4); RRDB dense blocks are a different, later paper's
architecture (ESRGAN), so this only exists as an explicit opt-in alternative,
not a replacement.

Architecture matches BasicSR's RRDBNet exactly (layer names included) so that
official Real-ESRGAN pretrained weights load directly with no key remapping:
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
(x2plus chosen specifically because it's a native 2x model, matching this
pipeline's 2x scale -- no need to fake a 4x model down to 2x.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_init_weights(module_list, scale=1):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()


def _make_layer(basic_block, num_basic_block, **kwarg):
    return nn.Sequential(*[basic_block(**kwarg) for _ in range(num_basic_block)])


def _pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used inside each RRDB (ESRGAN)."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        _default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block (3 stacked ResidualDenseBlocks)."""

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    ESRGAN/Real-ESRGAN generator. scale=2 matches RealESRGAN_x2plus.pth:
    internally the network always upsamples 4x after a pixel-unshuffle
    pre-shrink, so the net input->output ratio equals the requested scale.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = _make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = _pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = _pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class RRDBGeneratorV(nn.Module):
    """
    Wraps RRDBNet for this pipeline's single-channel V input/output:
    replicate V -> pseudo-RGB going in (same trick already used for the VGG
    perceptual loss), average the 3 output channels coming back out. The
    inner RRDBNet stays exactly the stock 3-channel architecture so official
    pretrained weights load with zero modification.
    """

    def __init__(self, num_block=23):
        super().__init__()
        self.core = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64,
                            num_block=num_block, num_grow_ch=32)

    def load_pretrained(self, path: str, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        state_dict = ckpt.get("params_ema") or ckpt.get("params") or ckpt
        missing, unexpected = self.core.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[RRDBGeneratorV] load_pretrained: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"  missing keys (first 5): {missing[:5]}")
            if unexpected:
                print(f"  unexpected keys (first 5): {unexpected[:5]}")
        else:
            print(f"[RRDBGeneratorV] pretrained weights loaded cleanly from {path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = x.repeat(1, 3, 1, 1)
        out3 = self.core(x3)
        return out3.mean(dim=1, keepdim=True)
