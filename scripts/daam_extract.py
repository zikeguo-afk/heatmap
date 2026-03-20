#!/usr/bin/env python3
"""Generate DAAM-style cross-attention heatmaps from Stable Diffusion.

This script records cross-attention maps during sampling, aggregates them
across layers and timesteps, and exports per-token heatmaps plus the
synthesized image for a lightweight HTML viewer.
"""
import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    from diffusers import StableDiffusionPipeline
    from diffusers.models.attention_processor import AttnProcessor
except Exception as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependencies. Install with: pip install -r requirements.txt\n"
        f"Original error: {exc}"
    )


@dataclass
class RunConfig:
    prompt: str
    model: str
    steps: int
    guidance: float
    seed: int
    width: int
    height: int
    device: str


class AttentionAggregator:
    """Aggregate cross-attention maps across layers and timesteps."""

    def __init__(self, token_count: int, out_h: int, out_w: int, device: str):
        self.token_count = token_count
        self.out_h = out_h
        self.out_w = out_w
        self.device = device
        self.maps = torch.zeros((token_count, out_h, out_w), device=device)
        self.count = 0

    def add(self, attn_probs: torch.Tensor) -> None:
        """attn_probs shape: (batch, heads, query, key)."""
        if attn_probs.ndim != 4:
            return
        batch, heads, query, key = attn_probs.shape
        if batch == 0 or key != self.token_count:
            return

        side = int(math.sqrt(query))
        if side * side != query:
            return

        attn = attn_probs[0].mean(dim=0)  # (query, key)
        attn = attn.transpose(0, 1).contiguous()  # (key, query)
        attn = attn.view(key, 1, side, side)
        attn = F.interpolate(attn, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        self.maps += attn.squeeze(1)
        self.count += 1

    def finalize(self) -> torch.Tensor:
        if self.count == 0:
            return self.maps
        return self.maps / float(self.count)


class StoreAttnProcessor(AttnProcessor):
    """Wraps the default attention processor to capture cross-attention maps."""

    def __init__(self, aggregator: AttentionAggregator):
        self.aggregator = aggregator

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Only store cross-attention (text conditioning)
        if encoder_hidden_states is not hidden_states:
            # reshape to (batch, heads, query, key)
            head_dim = attn.heads
            b = attention_probs.shape[0] // head_dim
            attn_probs = attention_probs.view(b, head_dim, attention_probs.shape[1], attention_probs.shape[2])
            self.aggregator.add(attn_probs)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def tokenize(pipe, prompt: str):
    ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    tokens = pipe.tokenizer.convert_ids_to_tokens(ids)
    specials = set(pipe.tokenizer.all_special_ids)
    return ids, tokens, specials


def write_heatmaps(out_dir: str, tokens, maps: torch.Tensor, specials) -> list:
    heat_dir = os.path.join(out_dir, "heatmaps")
    os.makedirs(heat_dir, exist_ok=True)

    maps = maps.detach().float().cpu()
    maps = maps - maps.amin(dim=(1, 2), keepdim=True)
    denom = maps.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    maps = maps / denom

    token_entries = []
    for idx, token in enumerate(tokens):
        if idx in specials:
            continue
        token_safe = token.replace("/", "_").replace("<", "").replace(">", "")
        filename = f"t{idx:02d}_{token_safe}.png"
        path = os.path.join(heat_dir, filename)
        img = (maps[idx].numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(path)
        token_entries.append({"id": idx, "text": token, "file": f"heatmaps/{filename}"})
    return token_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DAAM-style attention heatmaps.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True, help="Output directory (e.g., web/data/run1)")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = RunConfig(
        prompt=args.prompt,
        model=args.model,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        width=args.width,
        height=args.height,
        device=args.device,
    )

    os.makedirs(args.out, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model,
        torch_dtype=torch.float16 if cfg.device.startswith("cuda") else torch.float32,
    )
    pipe = pipe.to(cfg.device)

    ids, tokens, specials = tokenize(pipe, cfg.prompt)
    aggregator = AttentionAggregator(len(tokens), cfg.height, cfg.width, cfg.device)

    processors = {}
    for name in pipe.unet.attn_processors.keys():
        processors[name] = StoreAttnProcessor(aggregator)
    pipe.unet.set_attn_processor(processors)

    generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)

    image = pipe(
        cfg.prompt,
        width=cfg.width,
        height=cfg.height,
        num_inference_steps=cfg.steps,
        guidance_scale=cfg.guidance,
        generator=generator,
    ).images[0]

    image_path = os.path.join(args.out, "image.png")
    image.save(image_path)

    maps = aggregator.finalize()
    token_entries = write_heatmaps(args.out, tokens, maps, specials)

    run = {
        "prompt": cfg.prompt,
        "image": "image.png",
        "width": cfg.width,
        "height": cfg.height,
        "tokens": token_entries,
    }

    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)

    print(f"Saved run to {args.out}")


if __name__ == "__main__":
    main()
