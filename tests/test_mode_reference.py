# tests/test_mode_reference.py
# WPF-05 — Mode reference (artwork → palette → recipe).
#
# Couvre les 4 done testables du ticket :
#   1. select_source_pixels : 'reference' ne filtre aucun pixel (≥ figurine, > si
#      l'image contient des pixels désaturés/clairs).
#   2. resolve_n_colors : défaut conditionnel 8 (reference) / 5 (figurine).
#   3. resolve_n_colors : une valeur --colors explicite gagne sur le défaut.
#   4. extract_colors : le mode est bien branché bout-en-bout (test comportemental
#      sur une image synthétique, remove_background neutralisé).
#
# Aucune dépendance à data/paints.json ni à rembg : pixels en mémoire + image
# temporaire (tmp_path), remove_background monkeypatché en identité.

import numpy as np
from PIL import Image

from wpf.core import (
    DEFAULT_COLORS_FIGURINE,
    DEFAULT_COLORS_REFERENCE,
    extract_colors,
    resolve_n_colors,
    select_source_pixels,
)


# --- Helpers ------------------------------------------------------------------

def _mixed_pixels(n_red=100, n_white=50, n_gray=50):
    """Pixels float (N, 3) : rouge saturé + blanc (clair) + gris (désaturé).
    Le rouge passe le filtre figurine ; blanc et gris sont filtrés."""
    red = np.tile([255.0, 0.0, 0.0], (n_red, 1))
    white = np.tile([250.0, 250.0, 250.0], (n_white, 1))
    gray = np.tile([128.0, 128.0, 128.0], (n_gray, 1))
    return np.vstack([red, white, gray])


# --- Done 1 : select_source_pixels filtre selon le mode -----------------------

def test_reference_keeps_all_pixels():
    pixels = _mixed_pixels()
    out = select_source_pixels(pixels, n_colors=2, mode="reference")
    assert len(out) == len(pixels)


def test_figurine_filters_desaturated_and_bright():
    # 100 rouges saturés > 2 * COLORED_MIN_RATIO (20) → pas de fallback ;
    # blanc (luminosité) et gris (saturation) sont retirés.
    pixels = _mixed_pixels(n_red=100, n_white=50, n_gray=50)
    out = select_source_pixels(pixels, n_colors=2, mode="figurine")
    assert len(out) == 100


def test_reference_strictly_more_than_figurine_when_desaturated_present():
    pixels = _mixed_pixels(n_red=100, n_white=50, n_gray=50)
    ref = select_source_pixels(pixels, n_colors=2, mode="reference")
    fig = select_source_pixels(pixels, n_colors=2, mode="figurine")
    assert len(ref) >= len(fig)
    assert len(ref) > len(fig)


def test_reference_ge_figurine_without_desaturated():
    # Image 100% saturée : aucun pixel à filtrer → les deux modes égaux.
    pixels = np.tile([255.0, 0.0, 0.0], (100, 1))
    ref = select_source_pixels(pixels, n_colors=2, mode="reference")
    fig = select_source_pixels(pixels, n_colors=2, mode="figurine")
    assert len(ref) >= len(fig)


# --- Done 2 : resolve_n_colors — défaut conditionnel au mode ------------------

def test_default_colors_reference_is_8():
    assert resolve_n_colors("reference", None) == DEFAULT_COLORS_REFERENCE == 8


def test_default_colors_figurine_is_5():
    assert resolve_n_colors("figurine", None) == DEFAULT_COLORS_FIGURINE == 5


# --- Done 3 : une valeur explicite gagne sur le défaut conditionnel -----------

def test_explicit_colors_wins_in_reference():
    assert resolve_n_colors("reference", 4) == 4


def test_explicit_colors_wins_in_figurine():
    assert resolve_n_colors("figurine", 12) == 12


# --- Done 4 : extract_colors branche le mode bout-en-bout ---------------------

def _save_red_white_image(tmp_path):
    """Image 10x20 : moitié gauche rouge saturé, moitié droite blanc (100 px chacun)."""
    arr = np.zeros((10, 20, 3), dtype=np.uint8)
    arr[:, :5] = [255, 0, 0]       # rouge vif saturé
    arr[:, 5:10] = [200, 0, 0]     # rouge sombre saturé (2 clusters distincts en figurine)
    arr[:, 10:] = [250, 250, 250]  # blanc clair
    path = tmp_path / "art.png"
    Image.fromarray(arr).save(path)
    return path


def test_extract_colors_mode_wired(tmp_path, monkeypatch):
    # remove_background neutralisé : pas de rembg, image RGB inchangée.
    monkeypatch.setattr("wpf.core.remove_background", lambda img: img)
    path = _save_red_white_image(tmp_path)

    ref_centers, *_ = extract_colors(path, n_colors=2, mode="reference")
    fig_centers, *_ = extract_colors(path, n_colors=2, mode="figurine")

    # reference garde le blanc → un cluster quasi-blanc doit apparaître.
    assert any(c.min() > 200 for c in ref_centers)
    # figurine filtre le blanc → il ne reste que du rouge, aucun cluster clair.
    assert all(c.min() <= 200 for c in fig_centers)
