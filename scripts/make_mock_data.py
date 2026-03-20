#!/usr/bin/env python3
"""Create a mock DAAM run for the HTML viewer (no model required)."""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def gaussian_blob(h, w, cx, cy, sigma=0.12):
    y, x = np.mgrid[0:h, 0:w]
    dx = (x / w) - cx
    dy = (y / h) - cy
    blob = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
    blob = (blob - blob.min()) / (blob.max() - blob.min() + 1e-6)
    return blob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default="monkey with hat walking")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    heat_dir = os.path.join(args.out, "heatmaps")
    os.makedirs(heat_dir, exist_ok=True)

    size = args.size
    img = Image.new("RGB", (size, size), (16, 20, 28))
    draw = ImageDraw.Draw(img)

    # Background gradient blocks
    draw.rectangle([0, 0, size, size], fill=(18, 24, 36))
    draw.rectangle([0, int(size * 0.55), size, size], fill=(24, 36, 48))

    # Simple shapes for visual anchors
    draw.ellipse([int(size * 0.15), int(size * 0.25), int(size * 0.45), int(size * 0.55)], fill=(160, 140, 110))
    draw.ellipse([int(size * 0.22), int(size * 0.18), int(size * 0.38), int(size * 0.32)], fill=(120, 90, 70))
    draw.rectangle([int(size * 0.55), int(size * 0.35), int(size * 0.8), int(size * 0.5)], fill=(90, 120, 160))
    draw.rectangle([int(size * 0.58), int(size * 0.31), int(size * 0.76), int(size * 0.36)], fill=(70, 90, 120))

    # Add a subtle path
    draw.rectangle([int(size * 0.05), int(size * 0.7), int(size * 0.95), int(size * 0.78)], fill=(50, 60, 75))

    img.save(os.path.join(args.out, "image.png"))

    tokens = [t for t in args.prompt.split(" ") if t.strip()]
    token_entries = []

    centers = [
        (0.3, 0.4),
        (0.5, 0.45),
        (0.65, 0.32),
        (0.5, 0.72),
    ]

    for idx, token in enumerate(tokens):
        cx, cy = centers[idx % len(centers)]
        blob = gaussian_blob(size, size, cx, cy, sigma=0.12 + 0.03 * idx)
        img_map = (blob * 255).astype(np.uint8)
        filename = f"t{idx:02d}_{token}.png"
        Image.fromarray(img_map).save(os.path.join(heat_dir, filename))
        token_entries.append({"id": idx, "text": token, "file": f"heatmaps/{filename}"})

    run = {
        "prompt": args.prompt,
        "image": "image.png",
        "width": size,
        "height": size,
        "tokens": token_entries,
    }

    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)

    print(f"Mock run written to {args.out}")


if __name__ == "__main__":
    main()
