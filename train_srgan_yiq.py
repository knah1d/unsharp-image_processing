"""
Training script for the YIQ-pipeline experiment (enhance_yiq.py) -- a
separate, opt-in alternative to train_srgan.py's paper-faithful HSV training.
Nothing here touches enhance.py, train_srgan.py, or their checkpoints.

Reuses every color-space-independent piece from train_srgan.py as-is:
Discriminator + RaGAN loss, HighPassLoss, VGGPerceptualLoss, EMA,
diff_augment, cosine LR schedule, warm-restart resume logic. Only the
dataset (extracts Y via YIQ instead of V via HSV) and generator input
domain differ.

Usage:
    python train_srgan_yiq.py \
        --manifest /content/data/manifest.json \
        --ckpt_dir /content/drive/MyDrive/endoscopy_srgan/checkpoints_yiq \
        --epochs 50 --pretrain_epochs 5 --batch_size 16
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

from enhance import SRGANGenerator
from enhance_yiq import rgb_to_yiq, pre_srgan_y
from train_srgan import (
    Discriminator, HighPassLoss, VGGPerceptualLoss, EMA, diff_augment, _validate,
)

HR_SIZE = 128
LR_SIZE = HR_SIZE // 2
CROP_MARGIN = 16


class EndoscopyYDataset(Dataset):
    """Same augmentation/crop scheme as train_srgan.py's EndoscopyVDataset,
    but extracts Y (YIQ luma) instead of V (HSV value), and both HR target
    and LR input are built via pre_srgan_y -- same domain-matching fix we
    had to make for the HSV pipeline (see train_srgan.py's EndoscopyVDataset
    docstring): the HR target must live in the same inverted/processed
    domain the generator actually operates in at inference."""

    def __init__(self, paths, train: bool = True):
        self.paths = paths
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            return self[random.randrange(len(self.paths))]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        yiq = rgb_to_yiq(rgb)
        y_ch = np.clip(yiq[..., 0], 0, 255).astype(np.uint8)

        target = HR_SIZE + CROP_MARGIN
        h, w = y_ch.shape
        scale = target / min(h, w)
        y_resized = cv2.resize(y_ch, (max(target, round(w * scale)), max(target, round(h * scale))),
                                interpolation=cv2.INTER_AREA)
        rh, rw = y_resized.shape

        if self.train:
            top = random.randint(0, rh - HR_SIZE)
            left = random.randint(0, rw - HR_SIZE)
        else:
            top, left = (rh - HR_SIZE) // 2, (rw - HR_SIZE) // 2
        hr_raw = y_resized[top:top + HR_SIZE, left:left + HR_SIZE]

        if self.train and random.random() < 0.5:
            hr_raw = np.ascontiguousarray(hr_raw[:, ::-1])

        hr = pre_srgan_y(hr_raw)
        lr_raw = cv2.resize(hr_raw, (LR_SIZE, LR_SIZE), interpolation=cv2.INTER_AREA)
        lr_sharp = pre_srgan_y(lr_raw)

        hr_t = torch.from_numpy(hr.astype(np.float32) / 255.0).unsqueeze(0)
        lr_t = torch.from_numpy(lr_sharp.astype(np.float32) / 255.0).unsqueeze(0)
        return lr_t, hr_t


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_srgan_yiq] device = {device}")

    with open(args.manifest) as f:
        manifest = json.load(f)
    train_ds = EndoscopyYDataset(manifest["train"], train=True)
    val_ds = EndoscopyYDataset(manifest["val"], train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    print(f"[train_srgan_yiq] train={len(train_ds)}  val={len(val_ds)}")

    G = SRGANGenerator(n_res=16, scale=2).to(device)
    D = Discriminator().to(device)
    ema = EMA(G, decay=args.ema_decay)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.9, 0.999))

    pixel_loss = nn.L1Loss()
    content_loss = VGGPerceptualLoss().to(device)
    hpf_loss = HighPassLoss().to(device)
    adv_loss = nn.BCEWithLogitsLoss()

    w_pixel, w_content, w_adv, w_hpf = args.w_pixel, args.w_content, args.w_adv, args.w_hpf

    os.makedirs(args.ckpt_dir, exist_ok=True)
    start_epoch = 0
    resume_path = os.path.join(args.ckpt_dir, "srgan_yiq_last.pth")
    resumed_sched_state = None
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        start_epoch = ckpt["epoch"] + 1
        if not args.fresh_schedule:
            resumed_sched_state = (ckpt.get("sched_G"), ckpt.get("sched_D"))
        print(f"[train_srgan_yiq] resumed from epoch {start_epoch}"
            + ("  (fresh LR schedule -- warm restart)" if args.fresh_schedule else ""))
        if args.fresh_schedule:
            for g in opt_G.param_groups:
                g["lr"] = args.lr
            for g in opt_D.param_groups:
                g["lr"] = args.lr

    remaining_adv_epochs = max(1, args.epochs - max(start_epoch, args.pretrain_epochs))
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=remaining_adv_epochs, eta_min=args.lr * 0.01)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=remaining_adv_epochs, eta_min=args.lr * 0.01)
    if resumed_sched_state and resumed_sched_state[0] is not None:
        sched_G.load_state_dict(resumed_sched_state[0])
        sched_D.load_state_dict(resumed_sched_state[1])

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
            "sched_G": sched_G.state_dict(),
            "sched_D": sched_D.state_dict(),
            "ema": ema.shadow,
        }, resume_path)
        torch.save(ema.shadow, os.path.join(args.ckpt_dir, "srgan_yiq.pth"))

    print(f"[train_srgan_yiq] done. Final (EMA) weights: "
        f"{os.path.join(args.ckpt_dir, 'srgan_yiq.pth')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--pretrain_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--w_pixel", type=float, default=1.0)
    p.add_argument("--w_content", type=float, default=0.05)
    p.add_argument("--w_adv", type=float, default=0.03)
    p.add_argument("--w_hpf", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--fresh_schedule", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
