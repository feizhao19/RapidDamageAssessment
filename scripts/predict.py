#!/usr/bin/env python3
"""Run inference on a pre/post disaster image pair and save the damage mask."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vipde.models import ViPDE
from vipde.utils import (
    describe_device,
    load_image_array,
    load_image_resized_padded,
    normalize_to_tensor,
    predict_with_tta,
    resolve_device,
    restore_to_original_size,
    save_damage_overlays,
    save_image_lossless,
    set_seed,
    supports_fp16,
)

TOTAL_STEPS = 6


def log_step(step: int, title: str, detail: str = "") -> None:
    message = f"[{step}/{TOTAL_STEPS}] {title}"
    if detail:
        message = f"{message} — {detail}"
    print(message, flush=True)


def format_resize_meta(meta) -> str:
    orig_w, orig_h = meta.original_size
    scaled_w, scaled_h = meta.scaled_size
    return (
        f"{orig_w}x{orig_h} -> {scaled_w}x{scaled_h} "
        f"(pad left={meta.pad_left}, top={meta.pad_top}, method={meta.resample})"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Predict damage mask from pre/post images")
    parser.add_argument("--pre-image", required=True)
    parser.add_argument("--post-image", required=True)
    parser.add_argument(
        "--weights",
        default="checkpoints/vipde_vitb_damage_v1.pth",
        help="ViPDE checkpoint (.pth). Defaults to checkpoints/vipde_vitb_damage_v1.pth",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=None,
        help="Optional SAM backbone weights. Only needed if --weights is a partial checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for all outputs. Created automatically if it does not exist.",
    )
    parser.add_argument("--model-name", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=1024, help="Longest side after resize; image is padded to a square")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index (Linux/NVIDIA only)")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device. auto prefers cuda, then Apple mps, then cpu.",
    )
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp16", "auto"],
        help="Inference precision. fp16/auto only apply on CUDA.",
    )
    parser.add_argument(
        "--resample",
        default="lanczos",
        choices=["lanczos", "cubic", "area", "nearest"],
        help="Interpolation for RGB resize. lanczos (default) is highest quality for downscaling.",
    )
    parser.add_argument(
        "--tta-rotate",
        action="store_true",
        help="Also run TTA (soft voting) and save results under <output-dir>/tta/.",
    )
    parser.add_argument(
        "--tta-mode",
        default="rotate",
        choices=["rotate", "d4"],
        help="TTA strategy when --tta-rotate is set: rotate (4 views) or d4 (8 flip+rotate views).",
    )
    return parser.parse_args()


def run_forward(model, pre, post, *, use_fp16: bool, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        if device.type == "cuda" and use_fp16:
            pre_in, post_in = pre.half(), post.half()
            model.half()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model(pre_in, post_in)
        pre_in, post_in = pre.float(), post.float()
        model.float()
        return model(pre_in, post_in)


def save_prediction_outputs(
    pred: np.ndarray,
    output_dir: Path,
    *,
    prefix: str,
    pre_orig: np.ndarray,
    post_orig: np.ndarray,
    pre_meta,
    post_meta,
) -> None:
    """Save damage mask and pre/post overlays for one prediction."""
    mask_path = output_dir / (f"damage_mask{prefix}.png" if prefix else "damage_mask.png")
    Image.fromarray(pred).save(mask_path)
    print(f"         damage mask{prefix or ''}: {mask_path}", flush=True)

    pre_mask_orig = restore_to_original_size(pred, pre_meta, is_mask=True)
    post_mask_orig = restore_to_original_size(pred, post_meta, is_mask=True)
    overlay_dir = output_dir / f"overlays{prefix}" if prefix else output_dir
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_paths = save_damage_overlays(
        pre_orig,
        post_orig,
        pre_mask_orig,
        str(overlay_dir),
        post_mask_np=post_mask_orig,
    )
    if prefix:
        print(f"         overlays{prefix}: {overlay_dir}", flush=True)
    for key, path in overlay_paths.items():
        print(f"           {key}: {path}", flush=True)


def main():
    args = parse_args()
    started_at = time.time()

    log_step(1, "Setup", "preparing runtime and output directory")
    set_seed(28)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = resolve_device(args.device)
    pixel_mean = [0.5] * 3
    pixel_std = [0.5] * 3

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = supports_fp16(device, args.precision)
    runtime_precision = "fp16" if use_fp16 else "fp32"
    print(f"         device: {describe_device(device)}, precision={runtime_precision}", flush=True)
    print(f"         output directory: {output_dir.resolve()}", flush=True)

    pre_input_path = output_dir / "pre_input.png"
    post_input_path = output_dir / "post_input.png"

    log_step(2, "Model init", f"backbone={args.model_name}, weights={args.weights}")
    model = ViPDE.from_pretrained(
        weights_path=args.weights,
        backbone_name=args.model_name,
        num_classes=args.num_classes,
        sam_checkpoint_path=args.sam_checkpoint,
    )
    model.to(device)
    model.eval()
    print(f"         model ready on {device}", flush=True)

    log_step(3, "Preprocess images", f"resize longest side to {args.img_size} and pad to square")
    pre_orig = load_image_array(args.pre_image)
    post_orig = load_image_array(args.post_image)
    save_image_lossless(pre_orig, str(pre_input_path))
    save_image_lossless(post_orig, str(post_input_path))
    print(f"         saved originals: {pre_input_path.name}, {post_input_path.name}", flush=True)

    pre_np, pre_meta = load_image_resized_padded(args.pre_image, args.img_size, method=args.resample)
    print(f"         pre-image:  {Path(args.pre_image).name} | {format_resize_meta(pre_meta)}", flush=True)

    post_np, post_meta = load_image_resized_padded(args.post_image, args.img_size, method=args.resample)
    print(f"         post-image: {Path(args.post_image).name} | {format_resize_meta(post_meta)}", flush=True)

    pre = normalize_to_tensor(pre_np, pixel_mean, pixel_std).to(device)
    post = normalize_to_tensor(post_np, pixel_mean, pixel_std).to(device)

    log_step(4, "Predicting", f"standard forward pass ({runtime_precision})")
    infer_started = time.time()
    logits = run_forward(model, pre, post, use_fp16=use_fp16, device=device)
    infer_seconds = time.time() - infer_started
    print(f"         logits shape: {tuple(logits.shape)} | elapsed: {infer_seconds:.2f}s", flush=True)

    tta_pred = None
    if args.tta_rotate:
        tta_views = 8 if args.tta_mode == "d4" else 4
        print(
            f"         also running TTA mode={args.tta_mode} ({tta_views} views, soft voting)...",
            flush=True,
        )
        tta_started = time.time()
        tta_pred = predict_with_tta(
            model,
            pre,
            post,
            mode=args.tta_mode,
            use_fp16=use_fp16,
            device=device,
        )
        tta_seconds = time.time() - tta_started
        print(f"         TTA soft voting done | elapsed: {tta_seconds:.2f}s", flush=True)

    log_step(5, "Post-process", "argmax over class logits to build damage mask")
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    unique_classes = sorted(int(v) for v in np.unique(pred))
    print(f"         predicted classes in mask: {unique_classes}", flush=True)
    if tta_pred is not None:
        tta_classes = sorted(int(v) for v in np.unique(tta_pred))
        print(f"         TTA predicted classes in mask: {tta_classes}", flush=True)

    log_step(6, "Save outputs", "writing mask, originals, and overlay visualizations")
    save_prediction_outputs(
        pred,
        output_dir,
        prefix="",
        pre_orig=pre_orig,
        post_orig=post_orig,
        pre_meta=pre_meta,
        post_meta=post_meta,
    )

    if tta_pred is not None:
        tta_dir = output_dir / "tta"
        tta_dir.mkdir(parents=True, exist_ok=True)
        print(f"         saving TTA results to: {tta_dir.resolve()}", flush=True)
        save_prediction_outputs(
            tta_pred,
            tta_dir,
            prefix="",
            pre_orig=pre_orig,
            post_orig=post_orig,
            pre_meta=pre_meta,
            post_meta=post_meta,
        )

    total_seconds = time.time() - started_at
    print(f"\nDone. All outputs written to {output_dir.resolve()} ({total_seconds:.2f}s total)", flush=True)


if __name__ == "__main__":
    main()
