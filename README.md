# Warhammer Paint Finder

Take a photo of a painted miniature, get the closest matching paints from Citadel, Vallejo, AK, and other brands.

The tool extracts dominant colors from an image using K-means clustering, removes the background with rembg, then matches each color against a database of 1700+ miniature paints using ΔE distance in CIE LAB color space.

## How it works

1. rembg isolates the miniature from the background
2. K-means clustering pulls out the dominant colors (near-white and low-saturation pixels are filtered out)
3. Each color is matched against the paint database by perceptual distance (ΔE CIE LAB)

You can also pass in your own paint collection. The matcher will try your paints first and fall back to the full catalog when nothing is close enough.

## Quick start

```bash
pip install -r requirements.txt

# Build the paint database (fetches from Arcturus5404/miniature-paints)
python build_dataset.py

# Match paints from a photo
python colour_matcher.py photo.jpg
```

## Usage

```bash
# 5 colors, top 3 matches each (default)
python colour_matcher.py miniature.jpg

# More colors, more suggestions
python colour_matcher.py miniature.jpg --colors 8 --top 5

# Filter by brand
python colour_matcher.py miniature.jpg --brand "Citadel Colour"

# Use your own collection (prioritizes paints you own)
python colour_matcher.py miniature.jpg --collection my_collection.json

# Adjust the fallback threshold (default: 15.0 ΔE)
python colour_matcher.py miniature.jpg --collection my_collection.json --fallback-threshold 10

# Save a debug image (original + segmented + color swatches)
python colour_matcher.py miniature.jpg --debug
```

### Collection file format

A JSON array of paint objects:

```json
[
  { "name": "Abaddon Black", "brand": "Citadel Colour", "set": "Base", "hex": "#231F20", "r": 35, "g": 31, "b": 32 },
  { "name": "Mephiston Red", "brand": "Citadel Colour", "set": "Base", "hex": "#9A1115", "r": 154, "g": 17, "b": 21 }
]
```

## Example output

```
Couleur 1 — RGB(142, 35, 28)
  ΔE   3.2 | #8B2500 | Citadel Colour — Mephiston Red (Base)
  ΔE   5.7 | #922A1E | Vallejo — Cavalry Brown (Model Color)
  ΔE   8.1 | #7B2D26 | AK Interactive — Red Primer (3rd Gen)

Couleur 2 — RGB(45, 48, 51)
  ΔE   1.8 | #2B2B2B | Citadel Colour — Abaddon Black (Base)
  ...
```

## Paint database

Built from [Arcturus5404/miniature-paints](https://github.com/Arcturus5404/miniature-paints) (MIT, Rick Fleuren 2022). Covers Citadel, Vallejo, AK Interactive, Army Painter, Scale75, and more.

Run `python build_dataset.py` to fetch and rebuild `data/paints.json`.

## Requirements

- Python 3.10+
- Pillow, scikit-learn, numpy, requests, rembg

## License

MIT
