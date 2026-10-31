#!/usr/bin/env python3
"""Inline stages, characters and sound effects into index.html (one self-contained file).

Stages      every image in stages/ (except preview_*) becomes a stage. stages/stages.json sets the
            order, names and descriptions; images not listed there are added after it, named by file.
Characters  every folder in characters/ (and every lone image directly in characters/) becomes a
            playable character. characters/<id>/character.json is optional — without it the folder's
            sprite.* (or its first image) is used as a still, and the game animates it by itself.
"""
import base64
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "game.src.html"
OUT = ROOT / "index.html"
STAGES = ROOT / "stages"
CHARACTERS = ROOT / "characters"

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
BIG = 3 * 1024 * 1024  # warn above this: every byte is inlined into index.html

# Text files inserted verbatim (inline <script> bodies).
TEXTS = {
    "{{SFX_JS}}": "parts/sfx.js",
}


def is_image(p: pathlib.Path) -> bool:
    return p.is_file() and p.suffix.lower() in MIME


def data_uri(path: pathlib.Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing image: {path.relative_to(ROOT)}")
    if path.stat().st_size > BIG:
        print(f"warning: {path.relative_to(ROOT)} is {path.stat().st_size / 1e6:.1f} MB (inlined as-is)")
    return f"data:{MIME[path.suffix.lower()]};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def read_json(path: pathlib.Path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path.relative_to(ROOT)}: {e}")


def stage_defs() -> list:
    manifest = read_json(STAGES / "stages.json", [])
    listed = {e["file"] for e in manifest}
    for p in sorted(STAGES.iterdir()):
        if is_image(p) and not p.name.startswith("preview_") and p.name not in listed:
            manifest.append({"id": p.stem, "name": p.stem, "file": p.name})
    out = []
    for e in manifest:
        out.append({"id": e.get("id") or pathlib.Path(e["file"]).stem, "name": e.get("name") or e["file"],
                    "desc": e.get("desc", ""), "src": data_uri(STAGES / e["file"])})
    return out


def pick_sprite(folder: pathlib.Path):
    imgs = [p for p in sorted(folder.iterdir()) if is_image(p)]
    for p in imgs:
        if p.stem.lower() == "sprite":
            return p
    rest = [p for p in imgs if not p.stem.lower().startswith(("preview", "portrait", "walk", "attack"))]
    return rest[0] if rest else None


def character_def(cid: str, folder: pathlib.Path, meta: dict, sprite: pathlib.Path) -> dict:
    d = {"id": cid, "name": meta.get("name") or cid, "desc": meta.get("desc", ""), "alt": meta.get("alt", ""),
         "order": meta.get("order", 100), "sprite": data_uri(sprite)}
    portrait = meta.get("portrait") or next((p.name for p in sorted(folder.iterdir()) if is_image(p) and p.stem.lower() == "portrait"), None)
    if portrait:
        d["portrait"] = data_uri(folder / portrait)
    for key in ("walk", "attack"):
        if key in meta:
            sheet = meta[key]
            frames = int(sheet.get("frames", 1))
            if frames < 1:
                raise SystemExit(f"characters/{cid}: {key}.frames must be ≥ 1")
            d[key] = {"src": data_uri(folder / sheet["file"]), "frames": frames}
            for opt in ("loop", "dt"):
                if opt in sheet:
                    d[key][opt] = sheet[opt]
    for opt in ("feet", "height", "pixelated", "faces"):
        if opt in meta:
            d[opt] = meta[opt]
    return d


def character_defs() -> list:
    out = []
    if not CHARACTERS.is_dir():
        return out
    for p in sorted(CHARACTERS.iterdir()):
        if p.is_dir():
            meta = read_json(p / "character.json", {})
            sprite = p / meta["sprite"] if meta.get("sprite") else pick_sprite(p)
            if sprite is None:
                print(f"skipped characters/{p.name}: no image")
                continue
            out.append(character_def(p.name, p, meta, sprite))
        elif is_image(p):
            out.append(character_def(p.stem, CHARACTERS, {}, p))
    out.sort(key=lambda d: (d["order"], d["id"]))
    return out


def js_json(value) -> str:
    # "</" would end the inline <script>; "<\/" reads the same to JSON.parse and to JS
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def main() -> None:
    html = SRC.read_text(encoding="utf-8")
    for key, rel in TEXTS.items():
        if key not in html:
            raise SystemExit(f"placeholder {key} missing from source")
        text = (ROOT / rel).read_text(encoding="utf-8")
        if "</script" in text.lower():
            raise SystemExit(f"{rel} contains a closing script tag")
        html = html.replace(key, text)
    stages, chars = stage_defs(), character_defs()
    if not chars:
        raise SystemExit("no characters found in characters/")
    for key, value in (("/*{{STAGE_DEFS}}*/[]", stages), ("/*{{CHARACTER_DEFS}}*/[]", chars)):
        if key not in html:
            raise SystemExit(f"placeholder {key} missing from source")
        html = html.replace(key, js_json(value))
    OUT.write_text(html, encoding="utf-8")
    print(f"stages: {', '.join(s['name'] for s in stages)}")
    print(f"characters: {', '.join(c['name'] for c in chars)}")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
