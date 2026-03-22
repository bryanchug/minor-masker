from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from minor_masker.obfuscate import (
    ObfuscationMethod,
    ObfuscationParams,
    apply_obfuscation,
)


class RunMode(str, Enum):
    AGE_GATED = "age_gated"
    MASK_ALL = "mask_all"


@dataclass
class FaceRecord:
    x: int
    y: int
    w: int
    h: int
    age: Optional[float]
    obfuscated: bool


def _normalize_facial_area(fa: Any) -> Optional[Dict[str, int]]:
    if fa is None:
        return None
    if isinstance(fa, dict):
        keys = {k.lower(): k for k in fa}
        def g(name: str) -> int:
            k = keys.get(name) or keys.get(name[0])
            if k is None:
                raise KeyError(name)
            return int(fa[k])

        return {"x": g("x"), "y": g("y"), "w": g("w"), "h": g("h")}
    if isinstance(fa, (list, tuple)) and len(fa) >= 4:
        return {
            "x": int(fa[0]),
            "y": int(fa[1]),
            "w": int(fa[2]),
            "h": int(fa[3]),
        }
    return None


def _age_from_analyze_item(item: Dict[str, Any]) -> Optional[float]:
    age = item.get("age")
    if age is None:
        return None
    try:
        return float(age)
    except (TypeError, ValueError):
        return None


def _region_from_analyze_item(item: Dict[str, Any]) -> Optional[Dict[str, int]]:
    r = item.get("region") or item.get("facial_area")
    return _normalize_facial_area(r)


def detect_faces_with_ages(
    img_rgb: np.ndarray,
    detector_backend: str = "opencv",
) -> List[Dict[str, Any]]:
    """
    Return list of dicts: {"region": {x,y,w,h}, "age": float}.

    Uses DeepFace: extract all face regions, then run age on each crop.
    """
    from deepface import DeepFace

    faces_out: List[Dict[str, Any]] = []

    try:
        extracted = DeepFace.extract_faces(
            img_path=img_rgb,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=False,
        )
    except Exception:
        return []

    if not isinstance(extracted, list):
        extracted = [extracted]

    for obj in extracted:
        if not isinstance(obj, dict):
            continue
        fa = _normalize_facial_area(obj.get("facial_area"))
        if fa is None:
            continue
        x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
        ih, iw = img_rgb.shape[:2]
        x = max(0, min(x, iw - 1))
        y = max(0, min(y, ih - 1))
        w = max(1, min(w, iw - x))
        h = max(1, min(h, ih - y))
        crop = img_rgb[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        try:
            analyzed = DeepFace.analyze(
                img_path=crop,
                actions=["age"],
                enforce_detection=False,
                detector_backend=detector_backend,
                silent=True,
            )
        except Exception:
            continue
        if isinstance(analyzed, dict):
            analyzed = [analyzed]
        age_val: Optional[float] = None
        for it in analyzed:
            if isinstance(it, dict):
                age_val = _age_from_analyze_item(it)
                if age_val is not None:
                    break
        if age_val is None:
            continue
        faces_out.append(
            {
                "region": {"x": x, "y": y, "w": w, "h": h},
                "age": age_val,
            }
        )

    if faces_out:
        return faces_out

    # Fallback: single-face analyze on full frame (some images/layouts work better).
    try:
        analyzed = DeepFace.analyze(
            img_path=img_rgb,
            actions=["age"],
            enforce_detection=True,
            detector_backend=detector_backend,
            silent=True,
        )
    except Exception:
        return []

    if isinstance(analyzed, dict):
        analyzed = [analyzed]
    out2: List[Dict[str, Any]] = []
    for it in analyzed:
        if not isinstance(it, dict):
            continue
        reg = _region_from_analyze_item(it)
        age_val = _age_from_analyze_item(it)
        if reg is None or age_val is None:
            continue
        out2.append({"region": reg, "age": age_val})
    return out2


def process_image(
    image_bgr: np.ndarray,
    *,
    mode: RunMode,
    age_threshold: float,
    obfuscation: ObfuscationMethod,
    ob_params: ObfuscationParams,
    detector_backend: str = "opencv",
) -> tuple[np.ndarray, List[FaceRecord]]:
    """Return obfuscated image copy and per-face records."""
    out = image_bgr.copy()
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detected = detect_faces_with_ages(img_rgb, detector_backend=detector_backend)
    records: List[FaceRecord] = []

    for face in detected:
        r = face["region"]
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        age = face.get("age")
        age_f = float(age) if age is not None else None

        should_obfuscate = mode is RunMode.MASK_ALL or (
            mode is RunMode.AGE_GATED
            and age_f is not None
            and age_f < age_threshold
        )

        if should_obfuscate:
            apply_obfuscation(out, x, y, w, h, obfuscation, ob_params)

        records.append(
            FaceRecord(
                x=x,
                y=y,
                w=w,
                h=h,
                age=age_f,
                obfuscated=should_obfuscate,
            )
        )

    return out, records


def run_file(
    input_path: Path,
    output_path: Path,
    *,
    mode: RunMode,
    age_threshold: float,
    obfuscation: ObfuscationMethod,
    ob_params: ObfuscationParams,
    detector_backend: str = "opencv",
    json_path: Optional[Path] = None,
) -> None:
    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    out, records = process_image(
        image_bgr,
        mode=mode,
        age_threshold=age_threshold,
        obfuscation=obfuscation,
        ob_params=ob_params,
        detector_backend=detector_backend,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), out):
        raise OSError(f"Failed to write image: {output_path}")

    if json_path is not None:
        oparams = asdict(ob_params)
        oparams["mask_color_bgr"] = list(oparams["mask_color_bgr"])
        payload = {
            "input": str(input_path.resolve()),
            "output": str(output_path.resolve()),
            "mode": mode.value,
            "age_threshold": age_threshold,
            "obfuscation": obfuscation.value,
            "obfuscation_params": oparams,
            "detector_backend": detector_backend,
            "faces": [
                {
                    "bbox": {"x": fr.x, "y": fr.y, "w": fr.w, "h": fr.h},
                    "age": fr.age,
                    "obfuscated": fr.obfuscated,
                }
                for fr in records
            ],
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
