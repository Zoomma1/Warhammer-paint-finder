# colour_matcher.py
# CLI du Paint Finder : wrapper argparse + affichage autour de wpf.core.
# La logique pure (extraction, matching, recettes) vit dans wpf/core.py (WPF-06).
# Usage:
#   python colour_matcher.py --mode figurine <image.jpg>
#   python colour_matcher.py --mode reference <image.jpg>

import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans

from wpf.core import (
    DEFAULT_FALLBACK_THRESHOLD,
    build_recipe,
    extract_colors,
    load_paints,
    match_colors,
    resolve_n_colors,
    rgb_to_lab,
)

DATASET = Path("data/paints.json")

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

def _format_recipe_line(label: str, step: dict) -> str:
    """Ligne CLI alignée pour une étape de recette. Type du set affiché brut."""
    set_label = f"{step['brand']} — {step['set']}" if step.get("set") else step["brand"]
    tag = "  [→ basecoat : rien dans les seuils]" if step.get("fallback") else ""
    return f"  {label:<9}: {step['name']:<24} ({set_label})  ΔE {step['delta_e']:5.1f}{tag}"

def main():
    parser = argparse.ArgumentParser(description="Warhammer Paint Finder")
    parser.add_argument("image", help="Chemin vers l'image")
    parser.add_argument("--mode", choices=["figurine", "reference"], default="figurine",
                        help="figurine : photo de figurine (filtrage fond/saturation). "
                             "reference : artwork/concept art (toutes les couleurs intentionnelles, défaut 8 clusters)")
    parser.add_argument("--colors", type=int, default=None,
                        help="Nombre de couleurs à extraire (défaut : 8 en reference, 5 en figurine)")
    parser.add_argument("--top", type=int, default=3, help="Nombre de suggestions par couleur")
    parser.add_argument("--brand", help="Filtrer par marque (ex: 'Citadel Colour')")
    parser.add_argument("--debug", action="store_true", help="Sauvegarder l'image de debug")
    parser.add_argument("--recipe", action="store_true",
                        help="Afficher une recette 3 étapes (basecoat/shade/highlight) par couleur au lieu de la liste plate")
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

    n_colors = resolve_n_colors(args.mode, args.colors)
    print(f"Extraction des couleurs ({n_colors} clusters, mode {args.mode})...")
    dominant, kmeans, img, masked_img = extract_colors(args.image, n_colors, mode=args.mode)

    if args.debug:
        save_debug(img, kmeans, dominant, masked_img=masked_img)

    if args.recipe:
        print("Recettes...\n")
        for i, color in enumerate(dominant):
            color_lab = rgb_to_lab(color.astype(float))
            recipe = build_recipe(color_lab, paints, paints_lab, collection=collection)
            rgb = color.tolist()
            print(f"Couleur {i+1} — RGB({rgb[0]}, {rgb[1]}, {rgb[2]})")
            for label, role in (("Basecoat", "basecoat"), ("Shade", "shade"), ("Highlight", "highlight")):
                print(_format_recipe_line(label, recipe[role]))
            print()
        return

    print("Matching...\n")
    results = match_colors(dominant, paints, paints_lab, args.top, args.brand,
                           collection=collection, fallback_threshold=args.fallback_threshold)

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
