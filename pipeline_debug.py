"""
Step-by-step pipeline debug tool.

Runs one or more input images through every documented step of enhance.py's
enhance() pipeline (see its docstring, Steps 1-10), saving each step's output
as a viewable image into its own named folder under
pipeline_steps_output/<image_name>/<step_name>/, plus PSNR/SSIM/MSE/brightness
computed against the original input at every step, and a metrics_summary.csv
tying them all together.

The step-by-step math below mirrors enhance.py's pre_srgan_v()/enhance()
exactly, but is reimplemented inline here (rather than calling pre_srgan_v()
as one black-box block) purely so each intermediate stage can be captured --
enhance.py itself is untouched.

Usage:
    python pipeline_debug.py [image1.png image2.png ...]
    (defaults to both paper_original_image_*_clean.png if no args given)
"""

import csv
import os
import sys

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from enhance import hu_wbi_upsample, _apply_srgan

OUTPUT_ROOT = "pipeline_steps_output"

STEP_NAMES = [
    "00_original_input",
    "01_hsv_v_channel",
    "02_normalized",
    "03_inverted",
    "04_gamma_corrected",
    "05_hu_wbi_upsampled",
    "06_clahe",
    "07_downsampled",
    "08_unsharp_mask",
    "09_srgan",
    "10_final_output",
]


def _metrics(original_bgr: np.ndarray, preview_bgr: np.ndarray) -> dict:
    orig_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2GRAY)
    mse = float(np.mean((orig_gray.astype(np.float32) - prev_gray.astype(np.float32)) ** 2))
    psnr = float(cv2.PSNR(original_bgr, preview_bgr))
    ssim_val, _ = ssim(orig_gray, prev_gray, full=True, data_range=255)
    return {
        "psnr": round(psnr, 4),
        "ssim": round(float(ssim_val), 4),
        "mse": round(mse, 4),
        "brightness": round(float(preview_bgr.mean()), 2),
    }


def _save_step(out_dir: str, step_name: str, preview_bgr: np.ndarray,
              original_bgr: np.ndarray, orig_size: tuple) -> dict:
    step_dir = os.path.join(out_dir, step_name)
    os.makedirs(step_dir, exist_ok=True)

    # Resize back to the original spatial size first, so every step's image
    # and metric is directly comparable (some steps run at 2x resolution).
    if preview_bgr.shape[:2] != orig_size:
        preview_bgr = cv2.resize(preview_bgr, (orig_size[1], orig_size[0]))

    cv2.imwrite(os.path.join(step_dir, "image.png"), preview_bgr)

    m = _metrics(original_bgr, preview_bgr)
    with open(os.path.join(step_dir, "metrics.txt"), "w") as f:
        for k, v in m.items():
            f.write(f"{k}: {v}\n")
    return m


def run_pipeline_debug(image_path: str, gamma=0.8, clip_limit=2.0, cn=0.85, sigma=1.0):
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.resize(img, (512, 512))
    orig_size = img.shape[:2]  # (h, w)

    summary = []

    def record(step_idx, preview_bgr):
        m = _save_step(out_dir, STEP_NAMES[step_idx], preview_bgr, img, orig_size)
        summary.append((STEP_NAMES[step_idx], m))
        return m

    # Step 00: original input (baseline reference -- PSNR/SSIM against itself)
    record(0, img)

    # Step 1: RGB -> HSV, extract V (visualized as grayscale)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    record(1, cv2.cvtColor(v_ch, cv2.COLOR_GRAY2BGR))

    def reconstruct(v_state_uint8: np.ndarray) -> np.ndarray:
        """Merge a V-channel state back with the ORIGINAL H,S for a viewable
        RGB preview -- resizes H,S up/down to match if this stage is at a
        different resolution (e.g. the 2x Hu-WBI/CLAHE stages)."""
        v = v_state_uint8
        h_r, s_r = h_ch, s_ch
        if v.shape != h_ch.shape:
            h_r = cv2.resize(h_ch, (v.shape[1], v.shape[0]))
            s_r = cv2.resize(s_ch, (v.shape[1], v.shape[0]))
        merged_hsv = cv2.merge([h_r, s_r, v])
        merged_rgb = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB)
        return cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR)

    # Step 2: normalize (V/255) -- visually near-identical to step 1, since
    # normalizing then rescaling back to 0-255 for display returns the same
    # appearance; the normalization only matters for the math that follows.
    v_norm = v_ch.astype(np.float32) / 255.0
    v_norm_u8 = np.clip(v_norm * 255, 0, 255).astype(np.uint8)
    record(2, reconstruct(v_norm_u8))

    # Step 3: invert (1 - normalized V) -- expect a big PSNR/SSIM drop here,
    # this is intentionally a near-negative image at this stage.
    v_inv = 1.0 - v_norm
    v_inv_u8 = np.clip(v_inv * 255, 0, 255).astype(np.uint8)
    record(3, reconstruct(v_inv_u8))

    # Step 4: gamma correction (V'^gamma)
    v_gamma = np.power(np.clip(v_inv, 0.0, 1.0), gamma)
    v_u8 = np.clip(v_gamma * 255, 0, 255).astype(np.uint8)
    record(4, reconstruct(v_u8))

    # Step 5: Hu-WBI 2x upsample
    v_up = hu_wbi_upsample(v_u8)
    record(5, reconstruct(v_up))

    # Step 6: CLAHE on the upsampled image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v_up)
    record(6, reconstruct(v_clahe))

    # Step 7: downsample back to original size
    h_orig, w_orig = v_u8.shape
    v_down = cv2.resize(v_clahe, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    record(7, reconstruct(v_down))

    # Step 8: unsharp mask (HPF): PS = (I - LPF(I)) * Cn
    lpf = cv2.GaussianBlur(v_down.astype(np.float32), (0, 0), sigma)
    ps = (v_down.astype(np.float32) - lpf) * cn
    v_sharp = np.clip(v_down.astype(np.float32) + ps, 0, 255).astype(np.uint8)
    record(8, reconstruct(v_sharp))

    # Step 9: SRGAN super-resolution (uses whatever srgan_v.pth is present
    # locally; falls back to the Lanczos approximation if not)
    v_sr = _apply_srgan(v_sharp)
    record(9, reconstruct(v_sr))

    # Step 10: final -- invert back, merge HSV, convert to RGB
    v_final = 255 - v_sr
    final_hsv = cv2.merge([h_ch, s_ch, v_final])
    final_rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
    record(10, final_bgr)

    with open(os.path.join(out_dir, "metrics_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "psnr", "ssim", "mse", "brightness"])
        for step_name, m in summary:
            writer.writerow([step_name, m["psnr"], m["ssim"], m["mse"], m["brightness"]])

    print(f"[pipeline_debug] {image_path} -> {out_dir}/  ({len(summary)} steps saved)")
    return summary


if __name__ == "__main__":
    images = sys.argv[1:] or [
        "paper_original_image_1_clean.png",
        "paper_original_image_2_clean.png",
    ]
    for img_path in images:
        run_pipeline_debug(img_path)
