"""
Training script for the SRGAN generator used by enhance.py (_apply_srgan).

Training recipe (paper doesn't publish one explicitly, so this is a standard
SRGAN self-supervised setup adapted to this pipeline):

  1. Take a real endoscopy image at its given resolution -> this is HR ground
     truth (the V channel, size HR x HR).
  2. Downsample it 2x (bicubic) -> LR (size LR x LR = HR/2).
  3. Run it through the SAME classical pre-SRGAN steps used at inference time
     (enhance.pre_srgan_v: normalize -> invert -> gamma -> Hu-WBI -> CLAHE ->
     downsample -> unsharp mask) -> this is the generator's actual input.
  4. Generator upsamples 2x back to HR size; compare against the real HR V
     channel with content + adversarial + high-pass ("sharpness") loss,
     mirroring Eq. 8: L_overall = w1*L_HPF + w2*L_adv + w3*L_content.

Usage (inside Colab, after cloning the repo):
    python train_srgan.py \
        --manifest /content/drive/MyDrive/endoscopy_srgan/data/manifest.json \
        --ckpt_dir /content/drive/MyDrive/endoscopy_srgan/checkpoints \
        --epochs 50 --batch_size 16
"""

import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import Dataset, DataLoader

from enhance import SRGANGenerator, pre_srgan_v

HR_SIZE = 128
LR_SIZE = HR_SIZE // 2
CROP_MARGIN = 16   # extra pixels kept before random-cropping to HR_SIZE, for augmentation


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class EndoscopyVDataset(Dataset):
    """
    Reads real images from the manifest, returns (lr_sharp, hr) V-channel pairs.

    train=True  -> resize-with-margin + random crop + random hflip (augmentation)
    train=False -> deterministic center crop, no flip (stable validation numbers)
    """

    def __init__(self, paths, train: bool = True):
        self.paths = paths
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            # corrupt/unreadable file -> resample a different index instead of crashing the run
            return self[random.randrange(len(self.paths))]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v_ch = hsv[:, :, 2]

        target = HR_SIZE + CROP_MARGIN
        h, w = v_ch.shape
        scale = target / min(h, w)
        v_resized = cv2.resize(v_ch, (max(target, round(w * scale)), max(target, round(h * scale))),
                                interpolation=cv2.INTER_AREA)
        rh, rw = v_resized.shape

        if self.train:
            top = random.randint(0, rh - HR_SIZE)
            left = random.randint(0, rw - HR_SIZE)
        else:
            top, left = (rh - HR_SIZE) // 2, (rw - HR_SIZE) // 2
        hr = v_resized[top:top + HR_SIZE, left:left + HR_SIZE]

        if self.train and random.random() < 0.5:
            hr = np.ascontiguousarray(hr[:, ::-1])

        lr_raw = cv2.resize(hr, (LR_SIZE, LR_SIZE), interpolation=cv2.INTER_AREA)
        lr_sharp = pre_srgan_v(lr_raw)

        hr_t = torch.from_numpy(hr.astype(np.float32) / 255.0).unsqueeze(0)
        lr_t = torch.from_numpy(lr_sharp.astype(np.float32) / 255.0).unsqueeze(0)
        return lr_t, hr_t


# ═══════════════════════════════════════════════════════════════════════════
# Discriminator (paper Sect. 3.4: CNN-like, 8 conv layers, LeakyReLU,
# BatchNorm from the 2nd conv layer onward) -- standard SRGAN discriminator.
# ═══════════════════════════════════════════════════════════════════════════

def _disc_block(in_ch, out_ch, stride, use_bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        layers += _disc_block(1, 64, 1, use_bn=False)     # conv 1 (no BN)
        layers += _disc_block(64, 64, 2)                  # conv 2
        layers += _disc_block(64, 128, 1)                 # conv 3
        layers += _disc_block(128, 128, 2)                # conv 4
        layers += _disc_block(128, 256, 1)                # conv 5
        layers += _disc_block(256, 256, 2)                # conv 6
        layers += _disc_block(256, 512, 1)                # conv 7
        layers += _disc_block(512, 512, 2)                # conv 8
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)   # raw logits (BCEWithLogitsLoss expects this)


# ═══════════════════════════════════════════════════════════════════════════
# Differentiable augmentation (DiffAugment-style) applied only to what the
# discriminator sees. With ~4300 training images the discriminator can
# memorize the real set and stop giving the generator useful signal; randomly
# (and identically, for real vs fake) perturbing its input each step is a
# well-established fix for small-dataset GAN training. Purely a training-time
# technique -- doesn't touch the network architecture the paper specifies.
# ═══════════════════════════════════════════════════════════════════════════

def diff_augment(x: torch.Tensor) -> torch.Tensor:
    x = _rand_brightness(x)
    x = _rand_contrast(x)
    x = _rand_translation(x)
    x = _rand_cutout(x)
    return x


def _rand_brightness(x, strength=0.2):
    offset = (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * strength
    return x + offset


def _rand_contrast(x, strength=0.5):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    factor = 1 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * strength
    return (x - mean) * factor + mean


def _rand_translation(x, ratio=0.125):
    n, c, h, w = x.shape
    max_dx, max_dy = int(h * ratio), int(w * ratio)
    dx = torch.randint(-max_dx, max_dx + 1, (n,), device=x.device)
    dy = torch.randint(-max_dy, max_dy + 1, (n,), device=x.device)
    padded = torch.nn.functional.pad(x, (max_dy, max_dy, max_dx, max_dx), mode="reflect")
    out = torch.empty_like(x)
    for i in range(n):
        out[i] = padded[i, :, max_dx + dx[i]:max_dx + dx[i] + h, max_dy + dy[i]:max_dy + dy[i] + w]
    return out


def _rand_cutout(x, ratio=0.3):
    n, c, h, w = x.shape
    ch, cw = int(h * ratio), int(w * ratio)
    mask = torch.ones_like(x)
    for i in range(n):
        cy = random.randint(0, h - ch)
        cx = random.randint(0, w - cw)
        mask[i, :, cy:cy + ch, cx:cx + cw] = 0
    return x * mask


# ═══════════════════════════════════════════════════════════════════════════
# High-pass ("sharpness") loss -- mirrors Eq. 7's LPF-subtraction, done with a
# fixed (non-trainable) Gaussian depthwise conv so it stays on-GPU and
# differentiable.
# ═══════════════════════════════════════════════════════════════════════════

def _gaussian_kernel(kernel_size=5, sigma=1.0):
    ax = torch.arange(kernel_size) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.sum()).view(1, 1, kernel_size, kernel_size)


class HighPassLoss(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        self.register_buffer("kernel", _gaussian_kernel(kernel_size, sigma))
        self.pad = kernel_size // 2
        self.l1 = nn.L1Loss()

    def _hpf(self, x):
        lpf = torch.nn.functional.conv2d(x, self.kernel, padding=self.pad)
        return x - lpf

    def forward(self, sr, hr):
        return self.l1(self._hpf(sr), self._hpf(hr))


# ═══════════════════════════════════════════════════════════════════════════
# VGG perceptual loss -- this is what the paper's Eq. 8 "L_content" actually
# describes (Sect. 3.4: "content loss measures whether the generated image
# contains the crucial details present in the actual image"), not pixel MSE/L1.
#
# Layer choice: deep VGG layers (relu5_4, the original SRGAN/ESRGAN choice)
# encode ImageNet object-category semantics -- multiple medical-imaging GAN
# studies report this causes hallucinated/distorted anatomical detail when
# applied cross-domain (VGG never saw endoscopy tissue during pretraining).
# relu3_4 is shallow enough to capture generic edges/texture rather than
# object-level semantics, which transfers far more safely to a domain VGG
# was never trained on, while still avoiding the over-smoothing of raw pixel
# loss. The paper never specifies a layer (or VGG at all) -- this is our
# implementation choice, made conservatively given the diagnostic use case.
# ═══════════════════════════════════════════════════════════════════════════

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = tv_models.VGG19_Weights.IMAGENET1K_V1
        vgg = tv_models.vgg19(weights=weights).features[:18].eval()  # up to relu3_4
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.l1 = nn.L1Loss()

    def _prep(self, x):
        x3 = x.repeat(1, 3, 1, 1)          # single-channel V -> pseudo-RGB for VGG
        return (x3 - self.mean) / self.std

    def forward(self, sr, hr):
        return self.l1(self.vgg(self._prep(sr)), self.vgg(self._prep(hr)))


# ═══════════════════════════════════════════════════════════════════════════
# EMA of generator weights -- standard GAN stabilization trick. The raw
# last-iteration generator can be noisy epoch-to-epoch; the EMA shadow copy
# is what actually gets saved as srgan_v.pth.
# ═══════════════════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
                else:
                    self.shadow[k] = v.clone()


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_srgan] device = {device}")

    with open(args.manifest) as f:
        manifest = json.load(f)
    train_ds = EndoscopyVDataset(manifest["train"], train=True)
    val_ds = EndoscopyVDataset(manifest["val"], train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    print(f"[train_srgan] train={len(train_ds)}  val={len(val_ds)}")

    G = SRGANGenerator(n_res=16, scale=2).to(device)
    D = Discriminator().to(device)
    ema = EMA(G, decay=args.ema_decay)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # LR decay at the midpoint of the adversarial phase (standard SRGAN recipe)
    adv_epochs = max(1, args.epochs - args.pretrain_epochs)
    decay_at = args.pretrain_epochs + adv_epochs // 2
    sched_G = torch.optim.lr_scheduler.MultiStepLR(opt_G, milestones=[decay_at], gamma=0.1)
    sched_D = torch.optim.lr_scheduler.MultiStepLR(opt_D, milestones=[decay_at], gamma=0.1)

    pixel_loss = nn.L1Loss()
    content_loss = VGGPerceptualLoss().to(device)
    hpf_loss = HighPassLoss().to(device)
    adv_loss = nn.BCEWithLogitsLoss()

    w_pixel, w_content, w_adv, w_hpf = args.w_pixel, args.w_content, args.w_adv, args.w_hpf

    os.makedirs(args.ckpt_dir, exist_ok=True)
    start_epoch = 0
    resume_path = os.path.join(args.ckpt_dir, "srgan_last.pth")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        sched_G.load_state_dict(ckpt["sched_G"])
        sched_D.load_state_dict(ckpt["sched_D"])
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        start_epoch = ckpt["epoch"] + 1
        print(f"[train_srgan] resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        pretrain = epoch < args.pretrain_epochs
        G.train()
        D.train()
        # Raw (unweighted) per-term averages -- lets us see if one loss term
        # is silently dominating the others despite the configured weights,
        # since pixel/VGG-feature/BCE-logit/HPF losses live on very different
        # natural scales.
        running = {"d": 0.0, "g": 0.0, "pixel": 0.0, "content": 0.0,
                "adv": 0.0, "hpf": 0.0}

        for lr_v, hr_v in train_dl:
            lr_v, hr_v = lr_v.to(device), hr_v.to(device)
            bsz = lr_v.size(0)
            real_labels = torch.ones(bsz, 1, device=device)
            fake_labels = torch.zeros(bsz, 1, device=device)

            if pretrain:
                # ---- Warm-start: content+HPF only, discriminator untouched ----
                # Standard SRGAN practice -- an adversarially-trained-from-scratch
                # generator is unstable; pretrain it to a reasonable reconstruction
                # baseline first (mirrors Table 9's HPF-only config, which alone
                # already reaches SSIM 0.93).
                opt_G.zero_grad()
                sr_v = G(lr_v)
                l_pixel = pixel_loss(sr_v, hr_v)
                l_content = content_loss(sr_v, hr_v)
                l_hpf = hpf_loss(sr_v, hr_v)
                g_loss = w_pixel * l_pixel + w_content * l_content + w_hpf * l_hpf
                g_loss.backward()
                opt_G.step()
                ema.update(G)
                running["d"] += 0.0
                running["g"] += g_loss.item()
                running["pixel"] += l_pixel.item()
                running["content"] += l_content.item()
                running["hpf"] += l_hpf.item()
                continue

            # ---- Discriminator step ----
            # Relativistic average GAN (RaGAN, ESRGAN paper): D judges "is this
            # more realistic than the other one" rather than real/fake in
            # isolation -- sharper edges/textures than a plain SRGAN discriminator
            # loss, same 8-conv-layer network the paper specifies underneath.
            # diff_augment guards against the discriminator memorizing our
            # relatively small (~4300 image) training set.
            with torch.no_grad():
                sr_v = G(lr_v)
            opt_D.zero_grad()
            d_real_logits = D(diff_augment(hr_v))
            d_fake_logits = D(diff_augment(sr_v))
            rel_real = d_real_logits - d_fake_logits.mean()
            rel_fake = d_fake_logits - d_real_logits.mean()
            d_loss = (adv_loss(rel_real, real_labels) +
                    adv_loss(rel_fake, fake_labels)) * 0.5
            d_loss.backward()
            opt_D.step()

            # ---- Generator step ----
            opt_G.zero_grad()
            sr_v = G(lr_v)
            with torch.no_grad():
                d_real_logits = D(diff_augment(hr_v))
            d_fake_logits = D(diff_augment(sr_v))
            rel_fake = d_fake_logits - d_real_logits.mean()
            rel_real = d_real_logits - d_fake_logits.mean()
            g_adv = (adv_loss(rel_fake, real_labels) +
                    adv_loss(rel_real, fake_labels)) * 0.5
            l_pixel = pixel_loss(sr_v, hr_v)
            l_content = content_loss(sr_v, hr_v)
            l_hpf = hpf_loss(sr_v, hr_v)
            g_loss = (w_pixel * l_pixel + w_content * l_content +
                    w_adv * g_adv + w_hpf * l_hpf)
            g_loss.backward()
            opt_G.step()
            ema.update(G)

            running["d"] += d_loss.item()
            running["g"] += g_loss.item()
            running["pixel"] += l_pixel.item()
            running["content"] += l_content.item()
            running["adv"] += g_adv.item()
            running["hpf"] += l_hpf.item()

        sched_G.step()
        sched_D.step()

        n_batches = len(train_dl)
        phase = "pretrain" if pretrain else "adversarial"
        print(f"[epoch {epoch}] ({phase}) D={running['d']/n_batches:.4f}  "
            f"G={running['g']/n_batches:.4f}  lr={sched_G.get_last_lr()[0]:.2e}\n"
            f"           raw terms -> pixel={running['pixel']/n_batches:.4f}  "
            f"content={running['content']/n_batches:.4f}  "
            f"adv={running['adv']/n_batches:.4f}  "
            f"hpf={running['hpf']/n_batches:.4f}")

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            _validate(G, val_dl, device, epoch)

        torch.save({
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "sched_G": sched_G.state_dict(),
            "sched_D": sched_D.state_dict(),
            "ema": ema.shadow,
        }, resume_path)
        torch.save(ema.shadow, os.path.join(args.ckpt_dir, "srgan_v.pth"))

    print(f"[train_srgan] done. Final (EMA) generator weights: "
        f"{os.path.join(args.ckpt_dir, 'srgan_v.pth')}")


@torch.no_grad()
def _validate(G, val_dl, device, epoch):
    G.eval()
    mse_total, n = 0.0, 0
    for lr_v, hr_v in val_dl:
        lr_v, hr_v = lr_v.to(device), hr_v.to(device)
        sr_v = G(lr_v)
        mse_total += torch.nn.functional.mse_loss(sr_v, hr_v, reduction="sum").item()
        n += hr_v.numel()
    mse = mse_total / n
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    print(f"[epoch {epoch}] val_mse={mse:.6f}  val_psnr={psnr:.2f}dB")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--pretrain_epochs", type=int, default=5,
                  help="epochs of content+HPF-only warm-start before adversarial training begins")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    # Weights rebalanced against measured raw-term magnitudes from a real
    # smoke test (raw pixel~0.06, content~0.6, adv~5-7, hpf~0.015), so each
    # term's *weighted* contribution lands in the same rough order of
    # magnitude instead of one term silently dominating the total loss.
    p.add_argument("--w_pixel", type=float, default=1.0,
                  help="pixel-L1 stability anchor alongside the VGG content loss")
    p.add_argument("--w_content", type=float, default=0.05,
                  help="weight on VGG19-relu3_4 perceptual loss (the paper's L_content)")
    p.add_argument("--w_adv", type=float, default=0.01)
    p.add_argument("--w_hpf", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--val_every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
