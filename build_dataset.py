# build_dataset.py
# Fetch + parse Arcturus5404/miniature-paints MD files → data/paints.json
import requests
import json
import re
from pathlib import Path

REPO_API = "https://api.github.com/repos/Arcturus5404/miniature-paints/contents/paints"
OUTPUT = Path("data/paints.json")

def get_paint_files():
    response = requests.get(REPO_API)
    response.raise_for_status()
    files = response.json()
    return [f for f in files if f["name"].endswith(".md")]

def parse_md(content, brand):
    paints = []
    has_code = False
    for line in content.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if parts[1] == "Name":
            has_code = parts[2] == "Code"
            continue
        if "---" in parts[1]:
            continue
        offset = 1 if has_code else 0
        hex_match = re.search(r"`#([A-Fa-f0-9]{6})`", parts[6 + offset])
        if not hex_match:
            continue
        try:
            paints.append({
                "name": parts[1],
                "brand": brand,
                "set": parts[2 + offset],
                "r": int(parts[3 + offset]),
                "g": int(parts[4 + offset]),
                "b": int(parts[5 + offset]),
                "hex": "#" + hex_match.group(1).upper()
            })
        except ValueError:
            continue
    return paints

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    files = get_paint_files()
    print(f"{len(files)} fichiers trouvés")

    all_paints = []
    for f in files:
        brand = f["name"].replace(".md", "").replace("_", " ")
        content = requests.get(f["download_url"]).text
        paints = parse_md(content, brand)
        print(f"  {brand}: {len(paints)} couleurs")
        all_paints.extend(paints)

    OUTPUT.write_text(json.dumps(all_paints, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTotal : {len(all_paints)} peintures → {OUTPUT}")

if __name__ == "__main__":
    main()