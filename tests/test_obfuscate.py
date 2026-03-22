import numpy as np
import pytest

from minor_masker.obfuscate import (
    ObfuscationMethod,
    ObfuscationParams,
    apply_obfuscation,
    expand_bbox,
)


def test_expand_bbox_clips_to_image():
    x, y, w, h = expand_bbox(10, 10, 20, 20, 0.0, 100, 100)
    assert (x, y, w, h) == (10, 10, 20, 20)
    x, y, w, h = expand_bbox(0, 0, 10, 10, 1.0, 15, 15)
    assert x >= 0 and y >= 0
    assert x + w <= 15
    assert y + h <= 15


@pytest.mark.parametrize(
    "method",
    [
        ObfuscationMethod.BLUR,
        ObfuscationMethod.PIXELATE,
        ObfuscationMethod.MASK_SQUARE,
        ObfuscationMethod.MASK_CIRCLE,
    ],
)
def test_apply_obfuscation_mutates_roi(method: ObfuscationMethod):
    img = np.full((80, 80, 3), 200, dtype=np.uint8)
    before = img[20:60, 20:60].copy()
    apply_obfuscation(img, 20, 20, 40, 40, method, ObfuscationParams())
    after = img[20:60, 20:60]
    assert not np.array_equal(before, after)
