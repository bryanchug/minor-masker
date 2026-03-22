from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minor_masker.obfuscate import ObfuscationMethod, ObfuscationParams
from minor_masker.pipeline import RunMode, run_file


def _parse_bgr(color: str) -> tuple[int, int, int]:
    parts = color.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "mask-color must be three integers: B G R (OpenCV order), space or comma separated"
        )
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Detect faces, estimate apparent age with DeepFace, and obfuscate "
            "regions (blur, pixelate, or solid mask)."
        )
    )
    p.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input image file path(s).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output image path when a single input is given. "
            "For multiple inputs, use --output-dir instead."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for outputs when multiple inputs are provided (mirrors basenames).",
    )
    p.add_argument(
        "--mode",
        choices=[m.value for m in RunMode],
        default=RunMode.AGE_GATED.value,
        help="age_gated: obfuscate when predicted age < threshold; mask_all: obfuscate every face.",
    )
    p.add_argument(
        "--age-threshold",
        type=float,
        default=18.0,
        help="Apparent-age threshold for age_gated mode (default: 18).",
    )
    p.add_argument(
        "--obfuscation",
        choices=[m.value for m in ObfuscationMethod],
        default=ObfuscationMethod.BLUR.value,
        help="Visual obfuscation applied to each selected face region.",
    )
    p.add_argument(
        "--margin-ratio",
        type=float,
        default=0.15,
        help="Expand each face bbox by this fraction before obfuscating (default: 0.15).",
    )
    p.add_argument(
        "--blur-kernel",
        type=int,
        default=31,
        help="Gaussian blur kernel size (odd integer; default: 31).",
    )
    p.add_argument(
        "--blur-sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma (0 = auto from kernel; default: 0).",
    )
    p.add_argument(
        "--pixelate-factor",
        type=int,
        default=12,
        help="Downscale factor for pixelate (default: 12).",
    )
    p.add_argument(
        "--mask-color",
        default="0 0 0",
        help="Solid mask color as 'B G R' (OpenCV order), e.g. '0 0 0' (default: black).",
    )
    p.add_argument(
        "--detector-backend",
        default="opencv",
        help="DeepFace detector backend (default: opencv). Try retinaface for harder images.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Write a JSON sidecar next to each output (<output>.meta.json).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs: list[Path] = args.inputs
    mode = RunMode(args.mode)
    obfuscation = ObfuscationMethod(args.obfuscation)
    ob_params = ObfuscationParams(
        margin_ratio=args.margin_ratio,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        pixelate_factor=args.pixelate_factor,
        mask_color_bgr=_parse_bgr(str(args.mask_color)),
    )

    if len(inputs) == 1:
        inp = inputs[0]
        if args.output_dir is not None:
            print("error: use --output (file) for a single input, not --output-dir", file=sys.stderr)
            return 2
        if args.output is None:
            out = inp.with_name(f"{inp.stem}_masked{inp.suffix}")
        else:
            out = args.output
        outputs = [(inp, out)]
    else:
        if args.output is not None:
            print(
                "error: multiple inputs require --output-dir, not --output",
                file=sys.stderr,
            )
            return 2
        if args.output_dir is None:
            print("error: multiple inputs require --output-dir", file=sys.stderr)
            return 2
        od = args.output_dir
        outputs = [(p, od / p.name) for p in inputs]

    for inp, out in outputs:
        json_path = out.with_suffix(out.suffix + ".meta.json") if args.json else None
        try:
            run_file(
                inp,
                out,
                mode=mode,
                age_threshold=args.age_threshold,
                obfuscation=obfuscation,
                ob_params=ob_params,
                detector_backend=args.detector_backend,
                json_path=json_path,
            )
        except Exception as e:
            print(f"error: {inp}: {e}", file=sys.stderr)
            return 1
        print(f"{inp} -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
