from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import cv2
import numpy as np


class ObfuscationMethod(str, Enum):
    BLUR = "blur"
    PIXELATE = "pixelate"
    MASK_SQUARE = "mask_square"
    MASK_CIRCLE = "mask_circle"


@dataclass
class ObfuscationParams:
    margin_ratio: float = 0.15
    blur_kernel: int = 31
    blur_sigma: float = 0.0
    pixelate_factor: int = 12
    mask_color_bgr: Tuple[int, int, int] = (0, 0, 0)


def _odd_kernel(k: int) -> int:
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    return k


def expand_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    margin_ratio: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    if margin_ratio < 0:
        margin_ratio = 0.0
    cx = x + w / 2.0
    cy = y + h / 2.0
    nw = w * (1.0 + 2.0 * margin_ratio)
    nh = h * (1.0 + 2.0 * margin_ratio)
    nx = int(round(cx - nw / 2.0))
    ny = int(round(cy - nh / 2.0))
    nw_i = int(round(nw))
    nh_i = int(round(nh))
    nx = max(0, nx)
    ny = max(0, ny)
    if nx + nw_i > img_w:
        nw_i = img_w - nx
    if ny + nh_i > img_h:
        nh_i = img_h - ny
    nw_i = max(1, nw_i)
    nh_i = max(1, nh_i)
    return nx, ny, nw_i, nh_i


def apply_obfuscation(
    image_bgr: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    method: ObfuscationMethod,
    params: ObfuscationParams,
) -> None:
    """Mutate ``image_bgr`` in place for the given axis-aligned bbox."""
    img_h, img_w = image_bgr.shape[:2]
    ex, ey, ew, eh = expand_bbox(x, y, w, h, params.margin_ratio, img_w, img_h)

    if method is ObfuscationMethod.BLUR:
        roi = image_bgr[ey : ey + eh, ex : ex + ew]
        if roi.size == 0:
            return
        k = _odd_kernel(params.blur_kernel)
        blurred = cv2.GaussianBlur(roi, (k, k), params.blur_sigma)
        image_bgr[ey : ey + eh, ex : ex + ew] = blurred
        return

    if method is ObfuscationMethod.PIXELATE:
        roi = image_bgr[ey : ey + eh, ex : ex + ew]
        if roi.size == 0:
            return
        f = max(2, int(params.pixelate_factor))
        small_w = max(1, ew // f)
        small_h = max(1, eh // f)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (ew, eh), interpolation=cv2.INTER_NEAREST)
        image_bgr[ey : ey + eh, ex : ex + ew] = pixelated
        return

    color = tuple(int(c) for c in params.mask_color_bgr)

    if method is ObfuscationMethod.MASK_SQUARE:
        cv2.rectangle(
            image_bgr,
            (ex, ey),
            (ex + ew - 1, ey + eh - 1),
            color,
            thickness=-1,
        )
        return

    if method is ObfuscationMethod.MASK_CIRCLE:
        cx = ex + ew // 2
        cy = ey + eh // 2
        r = int(max(ew, eh) / 2.0)
        r = max(1, r)
        cv2.circle(image_bgr, (cx, cy), r, color, thickness=-1)
        return

    raise ValueError(f"Unknown obfuscation method: {method}")
