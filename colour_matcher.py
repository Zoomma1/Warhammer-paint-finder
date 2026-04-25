# colour_matcher.py
# K-means colour extraction + ΔE LAB matching against paint database
# Usage:
#   python colour_matcher.py --mode figurine <image.jpg>
#   python colour_matcher.py --mode reference <image.jpg>

import json
import argparse
from cv2 import kmeans
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans

DATASET = Path("data/paints.json")

def rgb_to_lab(rgb):
    rgb = np.array(rgb, dtype=float) / 255.0
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] /= 12.92
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = rgb @ M.T / np.array([0.95047, 1.00000, 1.08883])
    f = np.where(xyz > 0.008856, xyz ** (1/3), (903.3 * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)

def load_dataset():
    paints = json.loads(DATASET.read_text(encoding="utf-8"))
    rgb = np.array([[p["r"], p["g"], p["b"]] for p in paints], dtype=float)
    lab = rgb_to_lab(rgb)
    return paints, lab

def extract_colors(image_path, n_colors=5, sat_threshold=0.15):
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img).reshape(-1, 3).astype(float)

    r, g, b = pixels[:, 0]/255, pixels[:, 1]/255, pixels[:, 2]/255
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    with np.errstate(invalid='ignore'):
        saturation = np.where(cmax == 0, 0, (cmax - cmin) / cmax)
    colored = pixels[(saturation > 0.2) & (cmax/255 < 0.85)]
    source = colored if len(colored) > n_colors * 10 else pixels

    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(source)
    return kmeans.cluster_centers_.astype(int), kmeans, img


def save_debug(img, kmeans, colors, path="debug.png"):
    pixels = np.array(img).reshape(-1, 3).astype(float)
    labels = kmeans.predict(pixels)
    segmented = colors[labels].reshape(np.array(img).shape).astype(np.uint8)

    swatch_h = 60
    swatches = np.zeros((swatch_h, img.width, 3), dtype=np.uint8)
    w = img.width // len(colors)
    for i, c in enumerate(colors):
        swatches[:, i*w:(i+1)*w] = c

    combined = np.vstack([np.array(img), segmented, swatches])
    Image.fromarray(combined).save(path)
    print(f"Debug sauvegardé → {path}")

def match_colors(dominant_colors, paints, paints_lab, top_n=3, brand=None):
    results = []
    filtered = [(i, p) for i, p in enumerate(paints)
                if brand is None or p["brand"].lower() == brand.lower()]
    indices = [i for i, _ in filtered]
    subset_lab = paints_lab[indices]

    for color in dominant_colors:
        lab = rgb_to_lab(color.astype(float))
        deltas = np.sqrt(np.sum((subset_lab - lab) ** 2, axis=1))
        top = np.argsort(deltas)[:top_n]
        matches = []
        for rank in top:
            p = filtered[rank][1]
            matches.append({
                "name": p["name"],
                "brand": p["brand"],
                "set": p["set"],
                "hex": p["hex"],
                "delta_e": round(float(deltas[rank]), 2)
            })
        results.append({"input_rgb": color.tolist(), "matches": matches})
    return results

def main():
    parser = argparse.ArgumentParser(description="Warhammer Paint Finder")
    parser.add_argument("image", help="Chemin vers l'image")
    parser.add_argument("--mode", choices=["figurine", "reference"], default="figurine")
    parser.add_argument("--colors", type=int, default=5, help="Nombre de couleurs à extraire")
    parser.add_argument("--top", type=int, default=3, help="Nombre de suggestions par couleur")
    parser.add_argument("--brand", help="Filtrer par marque (ex: 'Citadel Colour')")
    parser.add_argument("--debug", action="store_true", help="Sauvegarder l'image de debug")
    args = parser.parse_args()

    print("Chargement du dataset...")
    paints, paints_lab = load_dataset()
    print(f"{len(paints)} peintures chargées")

    print(f"Extraction des couleurs ({args.colors} clusters)...")
    dominant, kmeans, img = extract_colors(args.image, args.colors)

    print("Matching...\n")
    results = match_colors(dominant, paints, paints_lab, args.top, args.brand)

    if args.debug:
        save_debug(img, kmeans, dominant)

    for i, r in enumerate(results):
        rgb = r["input_rgb"]
        print(f"Couleur {i+1} — RGB({rgb[0]}, {rgb[1]}, {rgb[2]})")
        for m in r["matches"]:
            print(f"  ΔE {m['delta_e']:5.1f} | {m['hex']} | {m['brand']} — {m['name']} ({m['set']})")
        print()

if __name__ == "__main__":
    main()