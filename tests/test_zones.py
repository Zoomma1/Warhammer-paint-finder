# tests/test_zones.py
# WPF-07a — Segmentation spatiale en zones.
#
# Une zone = un cluster K-means sur [L, a, b, λ·x, λ·y] + ses îlots (régions
# connexes) — décisions du 2026-09-15 : la définition « zone = composante
# connexe d'un cluster couleur » du ticket éclatait une vraie photo en 40+
# zones, et la couleur seule fusionnait cornes et chair (ΔE 8.7). Couvre :
#   1. Aplats francs → nombre de zones, aires, bbox, tri par aire décroissante.
#   2. Le filtre d'aire élimine le bruit (îlot minuscule) sans perdre sa surface.
#   3. Lissage par vote majoritaire.
#   4. Les pixels transparents ne forment jamais de zone ; tout pixel opaque
#      appartient à une zone.
#   5. La position sépare deux surfaces de même couleur éloignées.
#   6. Image RGB sans alpha (fallback rembg) → tout est opaque.
#
# Aucune dépendance à rembg ni à extract_colors : segment_zones est pure.

import numpy as np
import pytest
from PIL import Image

from wpf.core import segment_zones

RED = (255, 0, 0)
BLUE = (0, 0, 255)


# --- Helpers ------------------------------------------------------------------

def _rgba(height=100, width=100):
    """Image RGBA entièrement transparente (fond détouré)."""
    return np.zeros((height, width, 4), dtype=np.uint8)


def _paint(arr, top, bottom, left, right, color):
    arr[top:bottom, left:right, :3] = color
    arr[top:bottom, left:right, 3] = 255


def _to_img(arr):
    return Image.fromarray(arr, "RGBA")


@pytest.fixture
def two_flats():
    """60 % rouge à gauche, 40 % bleu à droite, bordure transparente 10 px."""
    arr = _rgba()
    _paint(arr, 10, 90, 10, 58, RED)    # 80 x 48 = 3840 px
    _paint(arr, 10, 90, 58, 90, BLUE)   # 80 x 32 = 2560 px
    return _to_img(arr)


# --- Done 1 : aplats → zones, aires, tri --------------------------------------

def test_two_flats_give_two_zones_sorted_by_area(two_flats):
    zones = segment_zones(two_flats, n_zones=2)
    assert [z["zone_id"] for z in zones] == [1, 2]
    assert zones[0]["area_px"] == 3840
    assert zones[1]["area_px"] == 2560
    assert zones[0]["rgb"] == list(RED)
    assert zones[1]["rgb"] == list(BLUE)


def test_area_ratio_is_relative_to_opaque_surface(two_flats):
    zones = segment_zones(two_flats, n_zones=2)
    assert zones[0]["area_ratio"] == pytest.approx(0.6)
    assert zones[1]["area_ratio"] == pytest.approx(0.4)


def test_bbox_is_left_top_right_bottom(two_flats):
    zones = segment_zones(two_flats, n_zones=2)
    assert zones[0]["bbox"] == (10, 10, 58, 90)
    assert zones[1]["bbox"] == (58, 10, 90, 90)


def test_zone_dict_structure(two_flats):
    zone = segment_zones(two_flats, n_zones=2)[0]
    assert set(zone) == {"zone_id", "cluster_id", "rgb", "lab", "area_px", "area_ratio", "bbox", "regions"}
    assert zone["lab"].shape == (3,)
    assert set(zone["regions"][0]) == {"area_px", "area_ratio", "bbox"}


def test_n_zones_is_capped_by_available_pixels():
    arr = _rgba()
    _paint(arr, 0, 2, 0, 2, RED)  # 4 pixels opaques
    zones = segment_zones(_to_img(arr), n_zones=12, min_area_ratio=0, smooth_ratio=0)
    assert 1 <= len(zones) <= 4


@pytest.mark.parametrize("n_zones", [0, -3])
def test_n_zones_below_one_is_clamped_to_one(two_flats, n_zones):
    zones = segment_zones(two_flats, n_zones=n_zones)
    assert len(zones) == 1
    assert zones[0]["area_px"] == 6400


# --- Done 2 : filtre d'aire ---------------------------------------------------

def test_noise_speck_is_filtered_from_islands_but_counted_in_area(two_flats):
    # 4 px rouges isolés dans le bleu (~0.06 %) : avec n_zones=2 ils rejoignent
    # le cluster rouge, comptent dans son aire mais pas dans ses îlots.
    arr = np.array(two_flats)
    _paint(arr, 50, 52, 70, 72, RED)
    red = segment_zones(_to_img(arr), n_zones=2, smooth_ratio=0)[0]
    assert red["area_px"] == 3844
    assert len(red["regions"]) == 1
    assert red["bbox"] == (10, 10, 58, 90)  # le speck n'étire pas la bbox


def test_noise_speck_kept_as_island_when_threshold_disabled(two_flats):
    arr = np.array(two_flats)
    _paint(arr, 50, 52, 70, 72, RED)
    red = segment_zones(_to_img(arr), n_zones=2, min_area_ratio=0, smooth_ratio=0)[0]
    assert [r["area_px"] for r in red["regions"]] == [3840, 4]


def test_cluster_below_threshold_is_dropped_entirely():
    # 4 px rouges pour 6400 opaques → le cluster rouge n'est pas une zone.
    # spatial_weight=0 : sinon, à 2 clusters, couper le bleu en deux moitiés
    # coûte moins d'inertie qu'isoler 4 pixels — ici on teste le seuil d'aire.
    arr = _rgba()
    _paint(arr, 10, 90, 10, 90, BLUE)
    _paint(arr, 50, 52, 70, 72, RED)
    zones = segment_zones(_to_img(arr), n_zones=2, spatial_weight=0, smooth_ratio=0)
    assert [z["rgb"] for z in zones] == [list(BLUE)]


def test_largest_island_is_kept_even_below_threshold():
    # Cluster à 2 % éclaté en 4 îlots de 0.5 % : la zone existe, et garde au
    # moins son plus grand îlot pour porter une bbox.
    arr = _rgba()
    _paint(arr, 0, 100, 0, 100, BLUE)
    for x in (0, 25, 50, 75):
        _paint(arr, 0, 5, x, x + 10, RED)  # 4 × 50 px = 200 px = 2 %
    red = segment_zones(_to_img(arr), n_zones=2, smooth_ratio=0)[-1]
    assert red["rgb"] == list(RED)
    assert red["area_px"] == 200
    assert len(red["regions"]) == 1


# --- Done 3 : lissage par vote majoritaire ------------------------------------

def test_speckle_is_absorbed_by_majority_vote(two_flats):
    # Un pixel rouge sur trois dans la moitié bleue : sans lissage, des centaines
    # d'îlots ; avec, le bleu absorbe tout et les deux aplats restent entiers.
    arr = np.array(two_flats)
    for y in range(20, 80):
        for x in range(65, 85):
            if (x + y) % 3 == 0:
                arr[y, x, :3] = RED
    raw = segment_zones(_to_img(arr), n_zones=2, min_area_ratio=0, smooth_ratio=0)
    smoothed = segment_zones(_to_img(arr), n_zones=2, min_area_ratio=0, smooth_ratio=0.05)
    assert len(raw[0]["regions"]) > 100
    assert [len(z["regions"]) for z in smoothed] == [1, 1]
    assert [z["area_px"] for z in smoothed] == [3840, 2560]


# --- Done 4 : transparents hors zone, opaques tous zonés ----------------------

def test_every_opaque_pixel_and_no_transparent_one_is_zoned(two_flats):
    zones = segment_zones(two_flats, n_zones=2)
    assert sum(z["area_px"] for z in zones) == 6400  # 80 x 80 opaques sur 100 x 100


def test_fully_transparent_image_gives_no_zone():
    assert segment_zones(_to_img(_rgba()), n_zones=2) == []


# --- Done 5 : la position sépare deux surfaces de même couleur ----------------

@pytest.fixture
def split_red():
    """Rouge à gauche, bleu au centre, rouge à droite."""
    arr = _rgba()
    _paint(arr, 0, 100, 0, 30, RED)
    _paint(arr, 0, 100, 30, 70, BLUE)
    _paint(arr, 0, 100, 70, 100, RED)
    return _to_img(arr)


def test_same_colour_far_apart_becomes_two_zones_when_asked(split_red):
    zones = segment_zones(split_red, n_zones=3)
    assert len(zones) == 3
    assert sorted(z["rgb"] for z in zones) == [list(BLUE), list(RED), list(RED)]
    reds = [z for z in zones if z["rgb"] == list(RED)]
    assert sorted(r["bbox"] for r in reds) == [(0, 0, 30, 100), (70, 0, 100, 100)]


def test_same_colour_far_apart_is_one_zone_with_two_islands_otherwise(split_red):
    red = segment_zones(split_red, n_zones=2)[0]
    assert red["rgb"] == list(RED)
    assert red["area_px"] == 6000
    assert [r["area_px"] for r in red["regions"]] == [3000, 3000]
    assert red["bbox"] == (0, 0, 100, 100)


# --- Done 6 : image RGB sans alpha --------------------------------------------

def test_rgb_image_treats_every_pixel_as_opaque():
    arr = np.zeros((50, 50, 3), dtype=np.uint8)
    arr[:, :25] = RED
    arr[:, 25:] = BLUE
    zones = segment_zones(Image.fromarray(arr, "RGB"), n_zones=2)
    assert len(zones) == 2
    assert sum(z["area_px"] for z in zones) == 2500
