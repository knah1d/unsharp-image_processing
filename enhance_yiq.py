"""
YIQ-based variant of the enhancement pipeline -- an explicit, separate
experiment, NOT a replacement for enhance.py's paper-faithful HSV pipeline.

The paper specifies HSV (Fig. 1, Table 8's "RGB to HSV conversion" component).
This file exists purely to test the hypothesis discussed in conversation:
Y (perceptually-weighted luma, 0.299R+0.587G+0.114B) may give a more
"honest" brightness signal than HSV's V (=max(R,G,B), dominated by whichever
channel is brightest) for gamma/CLAHE to work on -- at the cost of a real
risk HSV doesn't have: YIQ->RGB is a linear transform, so modifying Y while
keeping the original I,Q fixed can produce out-of-gamut RGB values that
need clipping (HSV's V->RGB conversion never needs this).

Everything else mirrors enhance.py's pipeline exactly (same Hu-WBI, CLAHE,
unsharp mask, SRGAN architecture) -- only the color-space split differs.
Uses its own weights file (srgan_yiq.pth) since a V-channel-trained
generator cannot be reused on Y-channel data.
"""

import os

import cv2
import numpy as np
import torch

from enhance import ResidualBlock, SRGANGenerator, hu_wbi_upsample  # noqa: F401  (re-exported for train_srgan_yiq.py)

# ═══════════════════════════════════════════════════════════════════════════════
# YIQ <-> RGB conversion (OpenCV has no built-in YIQ, unlike HSV)
# ═══════════════════════════════════════════════════════════════════════════════

_RGB2YIQ = np.array([
    [0.299, 0.587, 0.114],
    [0.596, -0.274, -0.322],
    [0.211, -0.523, 0.312],
], dtype=np.float32)
_YIQ2RGB = np.linalg.inv(_RGB2YIQ).astype(np.float32)


def rgb_to_yiq(rgb_uint8: np.ndarray) -> np.ndarray:
    """RGB [0,255] uint8 -> YIQ float32. Y in [0,255]; I,Q roughly [-150,150]."""
    rgb = rgb_uint8.astype(np.float32)
    return rgb @ _RGB2YIQ.T


def yiq_to_rgb(yiq: np.ndarray) -> np.ndarray:
    """YIQ float32 -> RGB [0,255] uint8, clipped (the real risk HSV doesn't have --
    modifying Y independently of I,Q can push RGB out of the valid [0,255] gamut)."""
    rgb = yiq @ _YIQ2RGB.T
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Classical pipeline (mirrors enhance.py's pre_srgan_v exactly, on Y instead of V)
# ═══════════════════════════════════════════════════════════════════════════════

def pre_srgan_y(y_ch: np.ndarray,
                gamma: float = 0.8,
                clip_limit: float = 2.0,
                tile_size: tuple = (8, 8),
                cn: float = 0.85,
                sigma: float = 1.0) -> np.ndarray:
    """Steps 2-8, identical math to enhance.py's pre_srgan_v, applied to Y."""
    y_norm = np.clip(y_ch, 0, 255).astype(np.float32) / 255.0
    y_inv = 1.0 - y_norm
    y_gamma = np.power(np.clip(y_inv, 0.0, 1.0), gamma)
    y_uint8 = np.clip(y_gamma * 255, 0, 255).astype(np.uint8)

    y_up = hu_wbi_upsample(y_uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    y_clahe = clahe.apply(y_up)

    h_orig, w_orig = y_uint8.shape
    y_down = cv2.resize(y_clahe, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    lpf = cv2.GaussianBlur(y_down.astype(np.float32), (0, 0), sigma)
    ps = (y_down.astype(np.float32) - lpf) * cn
    y_sharp = np.clip(y_down.astype(np.float32) + ps, 0, 255).astype(np.uint8)
    return y_sharp


_srgan_model_yiq: SRGANGenerator | None = None
_WEIGHTS_PATH_YIQ = os.path.join(os.path.dirname(__file__), "srgan_yiq.pth")


def srgan_yiq_is_trained() -> bool:
    return os.path.exists(_WEIGHTS_PATH_YIQ)


def _load_srgan_yiq() -> SRGANGenerator | None:
    global _srgan_model_yiq
    if _srgan_model_yiq is not None:
        return _srgan_model_yiq
    if not os.path.exists(_WEIGHTS_PATH_YIQ):
        return None
    try:
        m = SRGANGenerator()
        m.load_state_dict(torch.load(_WEIGHTS_PATH_YIQ, map_location="cpu", weights_only=True))
        m.eval()
        _srgan_model_yiq = m
        print(f"[SRGAN-YIQ] loaded weights from {_WEIGHTS_PATH_YIQ}")
        return m
    except Exception as e:
        print(f"[SRGAN-YIQ] weight load failed: {e}")
        return None


def _apply_srgan_yiq(y_uint8: np.ndarray) -> np.ndarray:
    h, w = y_uint8.shape
    model = _load_srgan_yiq()
    if model is not None:
        with torch.no_grad():
            t = torch.from_numpy(y_uint8.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            sr = model(t).squeeze().numpy()
        y_sr = (sr * 255).clip(0, 255).astype(np.uint8)
        return cv2.resize(y_sr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        y_big = cv2.resize(y_uint8, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        y_sharp = cv2.filter2D(y_big.astype(np.float32), -1, kernel)
        y_sharp = np.clip(y_sharp, 0, 255).astype(np.uint8)
        y_blend = cv2.addWeighted(y_sharp, 0.80, y_big, 0.20, 0)
        return cv2.resize(y_blend, (w, h), interpolation=cv2.INTER_LANCZOS4)


def enhance_yiq(image_bgr: np.ndarray,
                gamma: float = 0.8,
                clip_limit: float = 2.0,
                cn: float = 0.85,
                sigma: float = 1.0,
                use_srgan: bool = True) -> np.ndarray:
    """Full pipeline, YIQ instead of HSV. Same 10 steps as enhance.enhance()."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    yiq = rgb_to_yiq(rgb)
    y_ch, i_ch, q_ch = yiq[..., 0], yiq[..., 1], yiq[..., 2]

    y_sharp = pre_srgan_y(y_ch, gamma, clip_limit, cn=cn, sigma=sigma)
    y_sr = _apply_srgan_yiq(y_sharp) if use_srgan else y_sharp

    y_final = 255 - y_sr.astype(np.float32)
    final_yiq = np.stack([y_final, i_ch, q_ch], axis=-1)
    final_rgb = yiq_to_rgb(final_yiq)
    return cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
