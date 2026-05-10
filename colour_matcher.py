# colour_matcher.py
# K-means colour extraction + ΔE LAB matching against paint database
# Usage:
#   python colour_matcher.py --mode figurine <image.jpg>
#   python colour_matcher.py --mode reference <image.jpg>

import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans

DATASET = Path("data/paints.json")
DEFAULT_FALLBACK_THRESHOLD = 15.0

SAT_THRESHOLD = 0.2
LUMINOSITY_MAX = 0.85
COLORED_MIN_RATIO = 10
KMEANS_N_INIT = 10
KMEANS_RANDOM_STATE = 42

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array (0-255) to CIE LAB. Accepts any shape (..., 3)."""
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

def load_paints(path: Path) -> tuple[list[dict], np.ndarray]:
    """Load a paints JSON file and return (paints list, LAB array).
    Skips entries with missing r/g/b values."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    paints = [p for p in raw if p.get("r") is not None and p.get("g") is not None and p.get("b") is not None]
    rgb = np.array([[p["r"], p["g"], p["b"]] for p in paints], dtype=float)
    lab = rgb_to_lab(rgb)
    return paints, lab

def compute_saturation_mask(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pixel (saturation, cmax) arrays from a float RGB pixel array."""
    r, g, b = pixels[:, 0]/255, pixels[:, 1]/255, pixels[:, 2]/255
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    with np.errstate(invalid='ignore'):
        saturation = np.where(cmax == 0, 0, (cmax - cmin) / cmax)
    return saturation, cmax

def remove_background(img: Image.Image) -> Image.Image:
    """Remove background using rembg. Returns RGBA (transparent bg) or original RGB on failure."""
    try:
        from rembg import remove
        return remove(img)
    except Exception as e:
        print(f"⚠️  rembg indisponible ({e}) — analyse sur l'image complète")
        return img

def extract_colors(
    image_path: str | Path,
    n_colors: int = 5,
    sat_threshold: float = SAT_THRESHOLD,
) -> tuple[np.ndarray, KMeans, Image.Image, Image.Image]:
    """Cluster dominant colours in image_path; returns (centers, kmeans, img, masked_img)."""
    img = Image.open(image_path).convert("RGB")
    masked_img = remove_background(img)

    if masked_img.mode == "RGBA":
        rgba = np.array(masked_img)
        alpha_flat = rgba[:, :, 3].reshape(-1) > 10
        pixels = rgba[:, :, :3].reshape(-1, 3).astype(float)[alpha_flat]
    else:
        pixels = np.array(masked_img).reshape(-1, 3).astype(float)

    saturation, cmax = compute_saturation_mask(pixels)
    colored = pixels[(saturation > sat_threshold) & (cmax/255 < LUMINOSITY_MAX)]
    source = colored if len(colored) > n_colors * COLORED_MIN_RATIO else pixels

    kmeans = KMeans(n_clusters=n_colors, n_init=KMEANS_N_INIT, random_state=KMEANS_RANDOM_STATE)
    kmeans.fit(source)
    return kmeans.cluster_centers_.astype(int), kmeans, img, masked_img

def save_debug(img: Image.Image, kmeans: KMeans, colors: np.ndarray, path: str = "debug.png", masked_img: Image.Image | None = None) -> None:
    """Save segmented image + colour swatches side-by-side to path. If masked_img provided, also saves debug_masked.png."""
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

    if masked_img is not None and masked_img.mode == "RGBA":
        masked_path = path.replace(".png", "_masked.png") if path.endswith(".png") else path + "_masked.png"
        masked_img.save(masked_path)
        print(f"Debug masqué sauvegardé → {masked_path}")

def match_colors(
    dominant_colors: np.ndarray,
    paints: list[dict],
    paints_lab: np.ndarray,
    top_n: int = 3,
    brand: str | None = None,
    collection: tuple[list[dict], np.ndarray] | None = None,
    fallback_threshold: float = DEFAULT_FALLBACK_THRESHOLD,
) -> list[dict]:
    """Match each dominant colour against the paint database by ΔE; returns ranked results.

    If collection is provided, matches against it first. When no collection match
    satisfies fallback_threshold, adds best catalog matches tagged [hors collection].
    """
    results = []
    filtered_catalog = [(i, p) for i, p in enumerate(paints)
                        if brand is None or p["brand"].lower() == brand.lower()]
    catalog_indices = [i for i, _ in filtered_catalog]
    catalog_lab = paints_lab[catalog_indices]

    if collection is not None:
        coll_paints, coll_lab = collection
        filtered_coll = [(i, p) for i, p in enumerate(coll_paints)
                         if brand is None or p["brand"].lower() == brand.lower()]
        coll_indices = [i for i, _ in filtered_coll]
        coll_subset_lab = coll_lab[coll_indices] if coll_indices else np.empty((0, 3))

    for color in dominant_colors:
        lab = rgb_to_lab(color.astype(float))
        matches = []

        if collection is not None and len(coll_subset_lab) > 0:
            deltas = np.sqrt(np.sum((coll_subset_lab - lab) ** 2, axis=1))
            top = np.argsort(deltas)[:top_n]
            for rank in top:
                p = filtered_coll[rank][1]
                matches.append({
                    "name": p["name"],
                    "brand": p["brand"],
                    "set": p.get("set", ""),
                    "hex": p["hex"],
                    "delta_e": round(float(deltas[rank]), 2),
                    "source": "collection",
                })

        best_collection_de = matches[0]["delta_e"] if matches else float("inf")
        needs_fallback = collection is None or best_collection_de > fallback_threshold

        if needs_fallback:
            deltas = np.sqrt(np.sum((catalog_lab - lab) ** 2, axis=1))
            top = np.argsort(deltas)[:top_n]
            for rank in top:
                p = filtered_catalog[rank][1]
                matches.append({
                    "name": p["name"],
                    "brand": p["brand"],
                    "set": p["set"],
                    "hex": p["hex"],
                    "delta_e": round(float(deltas[rank]), 2),
                    "source": "catalog" if collection is not None else "catalog_only",
                })

        results.append({"input_rgb": color.tolist(), "matches": matches})
    return results

def main():
    parser = argparse.ArgumentParser(description="Warhammer Paint Finder")
    parser.add_argument("image", help="Chemin vers l'image")
    parser.add_argument("--mode", choices=["figurine", "reference"], default="figurine",
                        help="--mode reference|figurine : flag réservé, comportement différencié dans un ticket à venir (cf. WPF-05)")
    parser.add_argument("--colors", type=int, default=5, help="Nombre de couleurs à extraire")
    parser.add_argument("--top", type=int, default=3, help="Nombre de suggestions par couleur")
    parser.add_argument("--brand", help="Filtrer par marque (ex: 'Citadel Colour')")
    parser.add_argument("--debug", action="store_true", help="Sauvegarder l'image de debug")
    parser.add_argument("--collection", type=Path, help="Fichier JSON de peintures possédées (priorité au matching)")
    parser.add_argument("--fallback-threshold", type=float, default=DEFAULT_FALLBACK_THRESHOLD,
                        help=f"Seuil ΔE au-delà duquel le catalogue complet est consulté (défaut: {DEFAULT_FALLBACK_THRESHOLD})")
    args = parser.parse_args()

    print("Chargement du dataset...")
    paints, paints_lab = load_paints(DATASET)
    print(f"{len(paints)} peintures chargées")

    collection = None
    if args.collection:
        if not args.collection.exists():
            print(f"❌ Collection introuvable : {args.collection}")
            return
        raw_count = len(json.loads(args.collection.read_text(encoding="utf-8")))
        coll_paints, coll_lab = load_paints(args.collection)
        collection = (coll_paints, coll_lab)
        skipped = raw_count - len(coll_paints)
        msg = f"📦 Collection chargée : {len(coll_paints)} peintures (seuil fallback ΔE = {args.fallback_threshold})"
        if skipped:
            msg += f" — {skipped} ignorées (pas de valeur RGB)"
        print(msg)

    print(f"Extraction des couleurs ({args.colors} clusters)...")
    dominant, kmeans, img, masked_img = extract_colors(args.image, args.colors)

    print("Matching...\n")
    results = match_colors(dominant, paints, paints_lab, args.top, args.brand,
                           collection=collection, fallback_threshold=args.fallback_threshold)

    if args.debug:
        save_debug(img, kmeans, dominant, masked_img=masked_img)

    for i, r in enumerate(results):
        rgb = r["input_rgb"]
        print(f"Couleur {i+1} — RGB({rgb[0]}, {rgb[1]}, {rgb[2]})")
        for m in r["matches"]:
            tag = " [hors collection]" if m.get("source") == "catalog" else (" [collection ✓]" if m.get("source") == "collection" else "")
            set_info = f" ({m['set']})" if m.get("set") else ""
            print(f"  ΔE {m['delta_e']:5.1f} | {m['hex']} | {m['brand']} — {m['name']}{set_info}{tag}")
        print()

if __name__ == "__main__":
    main()
