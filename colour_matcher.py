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

# Recipe (WPF-04): seuils de luminosité LAB (L*) relatifs au basecoat.
SHADOW_L_DELTA = 15
HIGHLIGHT_L_DELTA = 10

SAT_THRESHOLD = 0.2
LUMINOSITY_MAX = 0.85
COLORED_MIN_RATIO = 10
KMEANS_N_INIT = 10
KMEANS_RANDOM_STATE = 42

# Nombre de clusters par défaut selon le mode (WPF-05).
# reference (artwork) : palette plus large, toutes les couleurs sont intentionnelles.
DEFAULT_COLORS_FIGURINE = 5
DEFAULT_COLORS_REFERENCE = 8

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

def select_source_pixels(
    pixels: np.ndarray,
    n_colors: int,
    mode: str = "figurine",
    sat_threshold: float = SAT_THRESHOLD,
) -> np.ndarray:
    """Pixels envoyés au K-means selon le mode (WPF-05).

    'reference' (artwork) : toutes les couleurs sont intentionnelles → aucun filtre.
    'figurine' : filtrage saturation/luminosité, avec fallback sur tous les pixels
    si trop peu passent le filtre (image très désaturée)."""
    if mode == "reference":
        return pixels
    saturation, cmax = compute_saturation_mask(pixels)
    colored = pixels[(saturation > sat_threshold) & (cmax/255 < LUMINOSITY_MAX)]
    return colored if len(colored) > n_colors * COLORED_MIN_RATIO else pixels

def extract_colors(
    image_path: str | Path,
    n_colors: int = 5,
    sat_threshold: float = SAT_THRESHOLD,
    mode: str = "figurine",
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

    source = select_source_pixels(pixels, n_colors, mode=mode, sat_threshold=sat_threshold)

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

def _nearest(target_lab: np.ndarray, paints_lab: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, float] | None:
    """Index + ΔE de la peinture la plus proche en LAB, optionnellement restreint à un masque booléen.
    Retourne None si le masque ne sélectionne aucune peinture."""
    indices = np.where(mask)[0] if mask is not None else np.arange(len(paints_lab))
    if len(indices) == 0:
        return None
    deltas = np.sqrt(np.sum((paints_lab[indices] - target_lab) ** 2, axis=1))
    best = int(np.argmin(deltas))
    return int(indices[best]), float(deltas[best])

def _recipe_step(paint: dict, delta_e: float, lab: np.ndarray) -> dict:
    """Construit une étape de recette à partir d'une peinture. Type du set affiché brut."""
    return {
        "name": paint["name"],
        "brand": paint["brand"],
        "set": paint.get("set", ""),
        "hex": paint["hex"],
        "delta_e": round(delta_e, 2),
        "L": round(float(lab[0]), 2),
    }

def _role_step(target_lab: np.ndarray, paints: list[dict], paints_lab: np.ndarray,
               mask: np.ndarray, basecoat: dict) -> dict:
    """Étape shade/highlight : la plus proche dans le masque, sinon fallback = basecoat."""
    found = _nearest(target_lab, paints_lab, mask=mask)
    if found is None:
        step = dict(basecoat)
        step["fallback"] = True
        return step
    idx, delta_e = found
    step = _recipe_step(paints[idx], delta_e, paints_lab[idx])
    step["fallback"] = False
    return step

def build_recipe(
    color_lab: np.ndarray,
    paints: list[dict],
    paints_lab: np.ndarray,
    collection: tuple[list[dict], np.ndarray] | None = None,
) -> dict:
    """Recette 3 étapes (basecoat → shade → highlight) pour une couleur cible LAB.

    Le rôle de chaque peinture découle de sa luminosité L* relative au basecoat,
    pas de son type Citadel. Fonction pure, séparable du CLI (prépare WPF-06).

    - basecoat  : peinture au ΔE min (toutes peintures)
    - shade     : ΔE min parmi L* < L*(basecoat) - SHADOW_L_DELTA, sinon fallback=basecoat
    - highlight : ΔE min parmi L* > L*(basecoat) + HIGHLIGHT_L_DELTA, sinon fallback=basecoat

    Si une collection est fournie, elle devient l'espace de recherche prioritaire (WPF-03).
    """
    color_lab = np.asarray(color_lab, dtype=float)
    if collection is not None:
        paints, paints_lab = collection
    paints_lab = np.asarray(paints_lab, dtype=float)

    base_idx, base_de = _nearest(color_lab, paints_lab)
    basecoat = _recipe_step(paints[base_idx], base_de, paints_lab[base_idx])
    base_L = basecoat["L"]

    L = paints_lab[:, 0]
    shade = _role_step(color_lab, paints, paints_lab, L < base_L - SHADOW_L_DELTA, basecoat)
    highlight = _role_step(color_lab, paints, paints_lab, L > base_L + HIGHLIGHT_L_DELTA, basecoat)

    return {"basecoat": basecoat, "shade": shade, "highlight": highlight}

def _format_recipe_line(label: str, step: dict) -> str:
    """Ligne CLI alignée pour une étape de recette. Type du set affiché brut."""
    set_label = f"{step['brand']} — {step['set']}" if step.get("set") else step["brand"]
    tag = "  [→ basecoat : rien dans les seuils]" if step.get("fallback") else ""
    return f"  {label:<9}: {step['name']:<24} ({set_label})  ΔE {step['delta_e']:5.1f}{tag}"

def resolve_n_colors(mode: str, requested: int | None) -> int:
    """Nombre de clusters (WPF-05) : valeur explicite si fournie, sinon défaut
    conditionnel au mode (reference → 8, figurine → 5)."""
    if requested is not None:
        return requested
    return DEFAULT_COLORS_REFERENCE if mode == "reference" else DEFAULT_COLORS_FIGURINE

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
