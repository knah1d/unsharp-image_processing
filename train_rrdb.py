
"""
Training script for the RRDBGeneratorV transfer-learning alternative
(rrdb_generator.py) -- fine-tunes a pretrained Real-ESRGAN generator on our
endoscopy data, instead of training the paper-faithful SRGANGenerator from
scratch (train_srgan.py, untouched by this file).

This is an explicit opt-in "best effort" path, not a replacement for the
paper-faithful pipeline. See rrdb_generator.py's docstring for why.

Reuses every architecture-independent piece from train_srgan.py as-is:
VGG perceptual loss (relu3_4), HPF loss, the paper's 8-conv discriminator +
RaGAN loss, diff_augment, EMA, cosine LR schedule, dataset/manifest loading.
Only the generator and its learning rate differ (fine-tuning regime: much
lower LR so we don't wreck the pretrained prior).

Usage:
    # one-time: download the pretrained base weights
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

    python train_rrdb.py \
        --manifest /content/drive/MyDrive/endoscopy_srgan/data/manifest.json \
        --ckpt_dir /content/drive/MyDrive/endoscopy_srgan/checkpoints_rrdb \
        --pretrained RealESRGAN_x2plus.pth \
        --epochs 30 --pretrain_epochs 2 --batch_size 8
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rrdb_generator import RRDBGeneratorV
from train_srgan import (
    EndoscopyVDataset, Discriminator, HighPassLoss, VGGPerceptualLoss, EMA,
    diff_augment, _validate,
)

HR_SIZE = 128
LR_SIZE = HR_SIZE // 2


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_rrdb] device = {device}")

    with open(args.manifest) as f:
        manifest = json.load(f)
    train_ds = EndoscopyVDataset(manifest["train"], train=True)
    val_ds = EndoscopyVDataset(manifest["val"], train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    print(f"[train_rrdb] train={len(train_ds)}  val={len(val_ds)}")

    G = RRDBGeneratorV(num_block=args.num_block).to(device)
    D = Discriminator().to(device)
    ema = EMA(G, decay=args.ema_decay)

    resume_path = os.path.join(args.ckpt_dir, "rrdb_last.pth")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    start_epoch = 0

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        start_epoch = ckpt["epoch"] + 1
        print(f"[train_rrdb] resumed from epoch {start_epoch}")
    elif args.pretrained:
        # Only load the base pretrained weights on a genuinely fresh run --
        # a resume already has our fine-tuned weights, loading the original
        # pretrained base again would throw away that progress.
        G.load_pretrained(args.pretrained)
    else:
        print("[train_rrdb] WARNING: no --pretrained given and no checkpoint to "
            "resume from -- training RRDBNet from random init defeats the "
            "whole point of this path (that's what train_srgan.py is for).")

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.9, 0.999))
    if os.path.exists(resume_path):
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])

    adv_epochs = max(1, args.epochs - args.pretrain_epochs)
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=adv_epochs, eta_min=args.lr_g * 0.01)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=adv_epochs, eta_min=args.lr_d * 0.01)

    pixel_loss = nn.L1Loss()
    content_loss = VGGPerceptualLoss().to(device)
    hpf_loss = HighPassLoss().to(device)
    adv_loss = nn.BCEWithLogitsLoss()

    w_pixel, w_content, w_adv, w_hpf = args.w_pixel, args.w_content, args.w_adv, args.w_hpf

    for epoch in range(start_epoch, args.epochs):
        pretrain = epoch < args.pretrain_epochs
        G.train()
        D.train()
        running = {"d": 0.0, "g": 0.0, "pixel": 0.0, "content": 0.0, "adv": 0.0, "hpf": 0.0}

        for lr_v, hr_v in train_dl:
            lr_v, hr_v = lr_v.to(device), hr_v.to(device)
            bsz = lr_v.size(0)
            real_labels = torch.ones(bsz, 1, device=device)
            fake_labels = torch.zeros(bsz, 1, device=device)

            if pretrain:
                # Brief warm-start on this data before adversarial training,
                # same rationale as train_srgan.py -- except here it's short
                # (default 2 epochs) since the generator already starts from
                # a competent pretrained prior, not random init.
                opt_G.zero_grad()
                sr_v = G(lr_v)
                l_pixel = pixel_loss(sr_v, hr_v)
                l_content = content_loss(sr_v, hr_v)
                l_hpf = hpf_loss(sr_v, hr_v)
                g_loss = w_pixel * l_pixel + w_content * l_content + w_hpf * l_hpf
                g_loss.backward()
                opt_G.step()
                ema.update(G)
                running["g"] += g_loss.item()
                running["pixel"] += l_pixel.item()
                running["content"] += l_content.item()
                running["hpf"] += l_hpf.item()
                continue

            # ---- Discriminator step (RaGAN, same as train_srgan.py) ----
            with torch.no_grad():
                sr_v = G(lr_v)
            opt_D.zero_grad()
            d_real_logits = D(diff_augment(hr_v))
            d_fake_logits = D(diff_augment(sr_v))
            rel_real = d_real_logits - d_fake_logits.mean()
            rel_fake = d_fake_logits - d_real_logits.mean()
            d_loss = (adv_loss(rel_real, real_labels) + adv_loss(rel_fake, fake_labels)) * 0.5
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
            g_adv = (adv_loss(rel_fake, real_labels) + adv_loss(rel_real, fake_labels)) * 0.5
            l_pixel = pixel_loss(sr_v, hr_v)
            l_content = content_loss(sr_v, hr_v)
            l_hpf = hpf_loss(sr_v, hr_v)
            g_loss = w_pixel * l_pixel + w_content * l_content + w_adv * g_adv + w_hpf * l_hpf
            g_loss.backward()
            opt_G.step()
            ema.update(G)

            running["d"] += d_loss.item()
            running["g"] += g_loss.item()
            running["pixel"] += l_pixel.item()
            running["content"] += l_content.item()
            running["adv"] += g_adv.item()
            running["hpf"] += l_hpf.item()

        if not pretrain:
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
            "ema": ema.shadow,
        }, resume_path)
        torch.save(ema.shadow, os.path.join(args.ckpt_dir, "srgan_v_transfer.pth"))

    print(f"[train_rrdb] done. Final (EMA) weights: "
        f"{os.path.join(args.ckpt_dir, 'srgan_v_transfer.pth')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--pretrained", default=None,
                  help="path to RealESRGAN_x2plus.pth (only used on a genuinely fresh run)")
    p.add_argument("--num_block", type=int, default=23, help="must match the pretrained checkpoint (23 for x2plus)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pretrain_epochs", type=int, default=2,
                  help="short warm-start since the generator already starts from a pretrained prior")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    # Fine-tuning LRs -- much lower than train_srgan.py's from-scratch 1e-4,
    # standard practice to avoid wrecking the pretrained prior.
    p.add_argument("--lr_g", type=float, default=1e-5)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--w_pixel", type=float, default=1.0)
    p.add_argument("--w_content", type=float, default=0.05)
    p.add_argument("--w_adv", type=float, default=0.03)
    p.add_argument("--w_hpf", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--val_every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
