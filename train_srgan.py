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
from torch.utils.data import Dataset, DataLoader

from enhance import SRGANGenerator, pre_srgan_v

HR_SIZE = 128
LR_SIZE = HR_SIZE // 2


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class EndoscopyVDataset(Dataset):
    """Reads real images from the manifest, returns (lr_sharp, hr) V-channel pairs."""

    def __init__(self, paths):
        self.paths = paths

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

        hr = cv2.resize(v_ch, (HR_SIZE, HR_SIZE), interpolation=cv2.INTER_AREA)
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
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_srgan] device = {device}")

    with open(args.manifest) as f:
        manifest = json.load(f)
    train_ds = EndoscopyVDataset(manifest["train"])
    val_ds = EndoscopyVDataset(manifest["val"])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    print(f"[train_srgan] train={len(train_ds)}  val={len(val_ds)}")

    G = SRGANGenerator(n_res=16, scale=2).to(device)
    D = Discriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.9, 0.999))

    content_loss = nn.L1Loss()
    hpf_loss = HighPassLoss().to(device)
    adv_loss = nn.BCEWithLogitsLoss()

    w_content, w_adv, w_hpf = args.w_content, args.w_adv, args.w_hpf

    os.makedirs(args.ckpt_dir, exist_ok=True)
    start_epoch = 0
    resume_path = os.path.join(args.ckpt_dir, "srgan_last.pth")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[train_srgan] resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        G.train()
        D.train()
        running = {"d": 0.0, "g": 0.0}

        for lr_v, hr_v in train_dl:
            lr_v, hr_v = lr_v.to(device), hr_v.to(device)
            bsz = lr_v.size(0)
            real_labels = torch.ones(bsz, 1, device=device)
            fake_labels = torch.zeros(bsz, 1, device=device)

            # ---- Discriminator step ----
            with torch.no_grad():
                sr_v = G(lr_v)
            opt_D.zero_grad()
            d_loss = (adv_loss(D(hr_v), real_labels) +
                    adv_loss(D(sr_v), fake_labels)) * 0.5
            d_loss.backward()
            opt_D.step()

            # ---- Generator step ----
            opt_G.zero_grad()
            sr_v = G(lr_v)
            g_adv = adv_loss(D(sr_v), real_labels)
            g_content = content_loss(sr_v, hr_v)
            g_hpf = hpf_loss(sr_v, hr_v)
            g_loss = w_content * g_content + w_adv * g_adv + w_hpf * g_hpf
            g_loss.backward()
            opt_G.step()

            running["d"] += d_loss.item()
            running["g"] += g_loss.item()

        n_batches = len(train_dl)
        print(f"[epoch {epoch}] D={running['d']/n_batches:.4f}  "
            f"G={running['g']/n_batches:.4f}")

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            _validate(G, val_dl, device, epoch)

        torch.save({
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
        }, resume_path)
        torch.save(G.state_dict(),
                    os.path.join(args.ckpt_dir, "srgan_v.pth"))

    print(f"[train_srgan] done. Final generator weights: "
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
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--w_content", type=float, default=1.0)
    p.add_argument("--w_adv", type=float, default=1e-3)
    p.add_argument("--w_hpf", type=float, default=0.1)
    p.add_argument("--val_every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
