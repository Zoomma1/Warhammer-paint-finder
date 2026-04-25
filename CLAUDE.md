# CLAUDE.md — Warhammer Paint Finder

## Commandes

| Commande | Dossier | Action |
|----------|---------|--------|
| `python build_dataset.py` | `./` | Fetcher + parser Arcturus5404 → `data/paints.json` |
| `python colour_matcher.py --mode figurine <image>` | `./` | Matcher une mini peinte |
| `python colour_matcher.py --mode reference <image>` | `./` | Générer palette depuis image de référence |
| `jupyter notebook` | `./` | Lancer les notebooks d'exploration |

## Architecture

Script CLI Python standalone — pas de serveur, pas de DB.
- `colour_matcher.py` — point d'entrée principal (K-means + matching ΔE LAB)
- `build_dataset.py` — fetch + parse MD Arcturus5404 → `data/paints.json`
- `data/paints.json` — dataset local généré, ne pas éditer manuellement

## Règles obligatoires

- Matching toujours en espace **LAB** (pas RGB) — distance ΔE uniquement
- `data/paints.json` est généré — relancer `build_dataset.py` pour mettre à jour
- Mode détection automatique figurine/référence = **V2**, ne pas implémenter en V1
