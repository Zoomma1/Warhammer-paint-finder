# tests/test_colour_matcher.py
# WPF-09 — Tests unitaires des fonctions core de colour_matcher.py.
#
# Couvre les 6 cas du ticket : rgb_to_lab, compute_saturation_mask, match_colors
# (nominal, filtre brand, collection + fallback) et load_paints.
#
# Aucune dépendance à data/paints.json : datasets synthétiques en mémoire
# (np.array) ; load_paints est testé via un fichier temporaire pytest (tmp_path),
# la seule fonction core qui lit réellement le disque.

import json

import numpy as np
import pytest

from colour_matcher import (
    compute_saturation_mask,
    load_paints,
    match_colors,
    rgb_to_lab,
)


# --- Helpers ------------------------------------------------------------------

def _paint(name, rgb, brand="Citadel Colour", set_name="Base"):
    """Dict peinture minimal (clés attendues par match_colors) + valeurs RGB."""
    r, g, b = rgb
    return {"name": name, "brand": brand, "set": set_name,
            "hex": "#000000", "r": r, "g": g, "b": b}


def _dataset(paints):
    """(paints, paints_lab) : LAB calculé depuis les RGB pour des ΔE maîtrisés."""
    rgb = np.array([[p["r"], p["g"], p["b"]] for p in paints], dtype=float)
    return paints, rgb_to_lab(rgb)


# --- Cas 1 : rgb_to_lab (valeurs CIE de référence, tolérance ±0.5) ------------

def test_rgb_to_lab_white():
    lab = rgb_to_lab(np.array([255.0, 255.0, 255.0]))
    assert lab[0] == pytest.approx(100.0, abs=0.5)
    assert lab[1] == pytest.approx(0.0, abs=0.5)
    assert lab[2] == pytest.approx(0.0, abs=0.5)


def test_rgb_to_lab_black():
    lab = rgb_to_lab(np.array([0.0, 0.0, 0.0]))
    assert lab[0] == pytest.approx(0.0, abs=0.5)
    assert lab[1] == pytest.approx(0.0, abs=0.5)
    assert lab[2] == pytest.approx(0.0, abs=0.5)


def test_rgb_to_lab_pure_red():
    # Référence CIE Lab (sRGB, D65) du rouge pur : L≈53.24, a≈80.09, b≈67.20.
    lab = rgb_to_lab(np.array([255.0, 0.0, 0.0]))
    assert lab[0] == pytest.approx(53.24, abs=0.5)
    assert lab[1] == pytest.approx(80.09, abs=0.5)
    assert lab[2] == pytest.approx(67.20, abs=0.5)


def test_rgb_to_lab_preserves_shape():
    # Accepte un lot de pixels (..., 3) → renvoie (..., 3).
    lab = rgb_to_lab(np.array([[255.0, 255.0, 255.0], [0.0, 0.0, 0.0]]))
    assert lab.shape == (2, 3)


# --- Cas 2 : compute_saturation_mask (saturation, cmax) -----------------------

def test_saturation_gray_is_zero():
    sat, cmax = compute_saturation_mask(np.array([[128.0, 128.0, 128.0]]))
    assert sat[0] == pytest.approx(0.0)
    assert cmax[0] == pytest.approx(128 / 255)


def test_saturation_black_is_zero():
    # cmax == 0 : la division est gardée → saturation 0, pas de NaN.
    sat, cmax = compute_saturation_mask(np.array([[0.0, 0.0, 0.0]]))
    assert sat[0] == pytest.approx(0.0)
    assert cmax[0] == pytest.approx(0.0)


def test_saturation_pure_red_is_one():
    sat, cmax = compute_saturation_mask(np.array([[255.0, 0.0, 0.0]]))
    assert sat[0] == pytest.approx(1.0)
    assert cmax[0] == pytest.approx(1.0)


def test_saturation_midpoint_hand_computed():
    # [255,128,128] → cmax=1, cmin=128/255 → sat = 1 - 128/255.
    sat, _ = compute_saturation_mask(np.array([[255.0, 128.0, 128.0]]))
    assert sat[0] == pytest.approx(1 - 128 / 255, abs=1e-6)


# --- Cas 3 : match_colors nominal (top-1 ΔE ≈ 0) ------------------------------

@pytest.fixture
def catalog():
    """Petit catalogue de couleurs pures, deux marques."""
    return _dataset([
        _paint("Pure Red", [255, 0, 0]),
        _paint("Pure Green", [0, 255, 0]),
        _paint("Pure Blue", [0, 0, 255]),
        _paint("White", [255, 255, 255]),
        _paint("Vallejo Red", [250, 10, 10], brand="Vallejo"),
    ])


def test_match_top1_is_exact_color(catalog):
    paints, paints_lab = catalog
    results = match_colors(np.array([[255, 0, 0]]), paints, paints_lab)
    top = results[0]["matches"][0]
    assert top["name"] == "Pure Red"
    assert top["delta_e"] == pytest.approx(0.0, abs=0.01)
    assert top["source"] == "catalog_only"


def test_match_one_result_per_input_color(catalog):
    paints, paints_lab = catalog
    dominant = np.array([[255, 0, 0], [0, 0, 255]])
    results = match_colors(dominant, paints, paints_lab)
    assert len(results) == 2
    assert results[0]["matches"][0]["name"] == "Pure Red"
    assert results[1]["matches"][0]["name"] == "Pure Blue"


# --- Cas 4 : match_colors avec filtre brand -----------------------------------

def test_brand_filter_restricts_pool(catalog):
    paints, paints_lab = catalog
    # Query = rouge pur. Sans filtre le top-1 serait "Pure Red" (Citadel).
    results = match_colors(np.array([[255, 0, 0]]), paints, paints_lab,
                           brand="Vallejo")
    matches = results[0]["matches"]
    assert all(m["brand"] == "Vallejo" for m in matches)
    assert matches[0]["name"] == "Vallejo Red"


def test_brand_filter_is_case_insensitive(catalog):
    paints, paints_lab = catalog
    results = match_colors(np.array([[255, 0, 0]]), paints, paints_lab,
                           brand="vallejo")
    assert results[0]["matches"][0]["brand"] == "Vallejo"


# --- Cas 5 : load_paints (skip r/g/b manquants, cohérence list/array) ---------

def test_load_paints_skips_entries_without_rgb(tmp_path):
    data = [
        {"name": "Has RGB", "brand": "X", "set": "S", "hex": "#ff0000",
         "r": 255, "g": 0, "b": 0},
        {"name": "No RGB", "brand": "X", "set": "S", "hex": "#000000"},
        {"name": "Partial", "brand": "X", "set": "S", "hex": "#111111",
         "r": 1, "g": 2},  # b manquant → ignorée
    ]
    path = tmp_path / "my_collection.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    paints, lab = load_paints(path)

    assert len(paints) == 1
    assert paints[0]["name"] == "Has RGB"
    # Cohérence entre la liste de peintures et l'array LAB retourné.
    assert lab.shape == (len(paints), 3)


def test_load_paints_keeps_zero_rgb(tmp_path):
    # r/g/b = 0 est valide (noir) : le filtre teste `is not None`, pas la véracité.
    data = [{"name": "Black", "brand": "X", "set": "S", "hex": "#000000",
             "r": 0, "g": 0, "b": 0}]
    path = tmp_path / "paints.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    paints, lab = load_paints(path)

    assert len(paints) == 1
    assert lab.shape == (1, 3)


# --- Cas 6 : match_colors avec collection (priorité + fallback_threshold) ------
# source ∈ {collection, catalog, catalog_only} selon que le meilleur ΔE
# collection passe ou non le seuil de fallback.

def test_source_catalog_only_without_collection(catalog):
    paints, paints_lab = catalog
    results = match_colors(np.array([[255, 0, 0]]), paints, paints_lab)
    assert results[0]["matches"][0]["source"] == "catalog_only"


def test_source_collection_when_within_threshold(catalog):
    paints, paints_lab = catalog
    # Collection contenant exactement la couleur cible → ΔE 0 ≤ seuil.
    collection = _dataset([_paint("Coll Red", [255, 0, 0])])
    results = match_colors(np.array([[255, 0, 0]]), paints, paints_lab,
                           collection=collection, fallback_threshold=15.0)
    matches = results[0]["matches"]
    assert matches[0]["source"] == "collection"
    assert matches[0]["name"] == "Coll Red"
    # ΔE collection sous le seuil → pas de fallback catalogue.
    assert all(m["source"] == "collection" for m in matches)


def test_source_catalog_fallback_when_collection_too_far(catalog):
    paints, paints_lab = catalog
    # Collection ne contenant qu'un bleu : très loin du rouge cible → ΔE > seuil.
    collection = _dataset([_paint("Coll Blue", [0, 0, 255])])
    results = match_colors(np.array([[255, 0, 0]]), paints, paints_lab,
                           collection=collection, fallback_threshold=15.0)
    sources = [m["source"] for m in results[0]["matches"]]
    # L'entrée collection (hors seuil) reste listée...
    assert "collection" in sources
    # ...et le catalogue complet est consulté en fallback.
    assert "catalog" in sources
    # Aucune entrée "catalog_only" quand une collection est fournie.
    assert "catalog_only" not in sources
    # Le meilleur match catalogue est bien le rouge exact.
    catalog_matches = [m for m in results[0]["matches"] if m["source"] == "catalog"]
    assert catalog_matches[0]["name"] == "Pure Red"
    assert catalog_matches[0]["delta_e"] == pytest.approx(0.0, abs=0.01)
