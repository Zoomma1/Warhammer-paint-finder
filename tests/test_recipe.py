# tests/test_recipe.py
# TDD — contrat de build_recipe (WPF-04).
# Le rôle d'une peinture (basecoat/shade/highlight) dépend de sa luminosité LAB (L*)
# relative au basecoat — pas de son type Citadel.
#
# Palette synthétique déterministe : on travaille directement en LAB pour contrôler
# précisément L* et ΔE, sans dépendre de data/paints.json (1700 peintures, fragile).

import numpy as np
import pytest

from wpf.core import build_recipe

# Couleur cible : un vert. L* = 50.
TARGET_LAB = np.array([50.0, -40.0, 30.0])


def _paint(name, lab, set_name="Citadel Colour — Base", brand="Citadel Colour"):
    """Construit (dict peinture, vecteur LAB) avec un L*/ΔE contrôlés."""
    return (
        {"name": name, "brand": brand, "set": set_name, "hex": "#000000"},
        np.array(lab, dtype=float),
    )


def make_palette(entries):
    """Transforme une liste de (dict, lab) en (paints, paints_lab)."""
    paints = [e[0] for e in entries]
    paints_lab = np.array([e[1] for e in entries], dtype=float)
    return paints, paints_lab


@pytest.fixture
def full_palette():
    """Palette riche couvrant tous les rôles + une peinture en zone morte.

    Cible L*=50 → basecoat L*=50 → seuils : shade L*<35, highlight L*>60.
    """
    return make_palette([
        _paint("Base Green", [50.0, -40.0, 30.0]),   # ΔE 0  → basecoat
        _paint("Dark Green", [25.0, -40.0, 30.0]),   # L*25<35, ΔE 25 → shade
        _paint("Light Green", [75.0, -40.0, 30.0]),  # L*75>60, ΔE 25 → highlight
        _paint("Mid Green", [45.0, -40.0, 30.0]),    # L*45 zone morte, ΔE 5 → jamais shade/highlight
        _paint("Dark Teal", [25.0, -10.0, 30.0]),    # L*25<35 mais ΔE 39 → moins bon shade que Dark Green
    ])


@pytest.fixture
def no_dark_no_bright_palette():
    """Aucune peinture sous le seuil shade ni au-dessus du seuil highlight → fallback."""
    return make_palette([
        _paint("Base Green", [50.0, -40.0, 30.0]),   # basecoat
        _paint("Mid Green", [45.0, -40.0, 30.0]),    # zone morte uniquement
    ])


# --- Basecoat -----------------------------------------------------------------

def test_basecoat_is_closest_delta_e(full_palette):
    paints, paints_lab = full_palette
    recipe = build_recipe(TARGET_LAB, paints, paints_lab)
    assert recipe["basecoat"]["name"] == "Base Green"
    assert recipe["basecoat"]["delta_e"] == pytest.approx(0.0, abs=0.01)


# --- Shade --------------------------------------------------------------------

def test_shade_is_darker_within_threshold(full_palette):
    paints, paints_lab = full_palette
    recipe = build_recipe(TARGET_LAB, paints, paints_lab)
    assert recipe["shade"]["name"] == "Dark Green"
    assert recipe["shade"]["fallback"] is False
    # Le shade est plus sombre que le basecoat.
    assert recipe["shade"]["L"] < recipe["basecoat"]["L"]


def test_shade_picks_min_delta_e_among_dark(full_palette):
    """Parmi les peintures sombres, on prend le ΔE min — pas n'importe quelle sombre."""
    paints, paints_lab = full_palette
    recipe = build_recipe(TARGET_LAB, paints, paints_lab)
    # Dark Teal est aussi sous le seuil mais plus loin en ΔE → ne doit pas être choisi.
    assert recipe["shade"]["name"] != "Dark Teal"


# --- Highlight ----------------------------------------------------------------

def test_highlight_is_brighter_within_threshold(full_palette):
    paints, paints_lab = full_palette
    recipe = build_recipe(TARGET_LAB, paints, paints_lab)
    assert recipe["highlight"]["name"] == "Light Green"
    assert recipe["highlight"]["fallback"] is False
    assert recipe["highlight"]["L"] > recipe["basecoat"]["L"]


# --- Zone morte ---------------------------------------------------------------

def test_deadzone_paint_never_used_as_shade_or_highlight(full_palette):
    paints, paints_lab = full_palette
    recipe = build_recipe(TARGET_LAB, paints, paints_lab)
    assert recipe["shade"]["name"] != "Mid Green"
    assert recipe["highlight"]["name"] != "Mid Green"


# --- Fallback (Option A : flag + valeur = basecoat) ---------------------------

def test_shade_fallback_when_no_dark_paint(no_dark_no_bright_palette):
    paints, paints_lab = no_dark_no_bright_palette
    recipe = build_recipe(TARGET_LAB, paints, paints_lab)
    assert recipe["shade"]["fallback"] is True
    assert recipe["shade"]["name"] == recipe["basecoat"]["name"]


def test_highlight_fallback_when_no_bright_paint(no_dark_no_bright_palette):
    paints, paints_lab = no_dark_no_bright_palette
    recipe = build_recipe(TARGET_LAB, paints, paints_lab)
    assert recipe["highlight"]["fallback"] is True
    assert recipe["highlight"]["name"] == recipe["basecoat"]["name"]


# --- Structure du dict --------------------------------------------------------

def test_recipe_dict_structure(full_palette):
    paints, paints_lab = full_palette
    recipe = build_recipe(TARGET_LAB, paints, paints_lab)
    assert set(recipe.keys()) == {"basecoat", "shade", "highlight"}
    for role in ("basecoat", "shade", "highlight"):
        step = recipe[role]
        assert {"name", "brand", "set", "hex", "delta_e", "L"} <= set(step.keys())
    # Le type du set est affiché brut, sans interprétation de consistance.
    assert recipe["basecoat"]["set"] == "Citadel Colour — Base"
    # Le fallback n'existe que pour shade/highlight.
    assert "fallback" in recipe["shade"]
    assert "fallback" in recipe["highlight"]


# --- Collection (WPF-03) : la collection est l'espace de recherche prioritaire -

def test_collection_used_as_search_space(full_palette):
    """Si une collection est fournie, le basecoat vient de la collection."""
    paints, paints_lab = full_palette  # catalogue : contient un ΔE 0
    collection = make_palette([
        _paint("Coll Green", [50.0, -35.0, 30.0]),  # ΔE 5 — moins bon que le catalogue
    ])
    recipe = build_recipe(TARGET_LAB, paints, paints_lab, collection=collection)
    assert recipe["basecoat"]["name"] == "Coll Green"
