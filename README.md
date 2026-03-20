# Car Prompt Heatmap Studio

This repo now has two parts:

- A practical DAAM extraction script for real Stable Diffusion runs.
- A browser-only simulation mode for car design prompts, which does not run Stable Diffusion and instead generates rule-based token heatmaps on top of an uploaded car image.

## Recommended Use

If you do not want to run Stable Diffusion locally, use the browser simulation mode:

```bash
cd /Users/hongwifi/Documents/热力图的复刻/web
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

In the page you can:

1. Upload a car image.
2. Enter a Chinese or English prompt.
3. Choose the vehicle view.
4. Generate token heatmaps.
5. Use the time-step slider or `Play` to simulate a diffusion-like reveal.

## What The Browser Mode Does

- Parses prompt tokens and matches car-design keywords.
- Maps tokens such as `headlights`, `roof`, `grille`, `wheel`, `SUV`, `electric`, `futuristic` to geometric regions on the car image.
- Uses a time-step animation to mimic how a heatmap might emerge during generation.
- Runs fully in the browser with no backend and no model inference.

This mode is useful for concept demos, prompt analysis, and interaction design. It is not true model attention.

## Real DAAM Extraction

If you later want true cross-attention maps from Stable Diffusion, the script is still available:

- [daam_extract.py](/Users/hongwifi/Documents/热力图的复刻/scripts/daam_extract.py)
- [requirements.txt](/Users/hongwifi/Documents/热力图的复刻/requirements.txt)

Example:

```bash
cd /Users/hongwifi/Documents/热力图的复刻
pip install -r requirements.txt

python3 scripts/daam_extract.py \
  --prompt "monkey with hat walking" \
  --out web/data/run1 \
  --steps 30 \
  --guidance 7.5 \
  --seed 42
```

Then refresh the static viewer if you want to inspect exported heatmaps.
