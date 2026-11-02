#!/usr/bin/env python3
"""Inline stages, characters and sound effects into index.html (one self-contained file).

Stages      every image in stages/ (except preview_*, *_mask, *_ink and *_map) becomes a stage. stages/stages.json
            sets the order, names, descriptions and the default 型 ("template": full / meisho / fuuin / shio);
            images not listed there are added after it, named by file. <stem>_mask.png (or "mask" in
            stages.json) is the gimmick mask: red 墨溜まり, yellow 名所, blue low ground, white high
            ground, green start (tools/make_mask.py writes a starter one).
            Detailed art: "hd": true draws the picture at full detail instead of as 128-px pixel art.
            <stem>_ink.* (or "ink") is the picture before its colour returns (made from the colour if
            missing); <stem>_map.png (or "map") is the terrain in four flat colours — floor (141,136,123),
            water (143,201,184), 朱 (208,83,58), wall (11,15,28) — used instead of reading it from colours.
Characters  every folder in characters/ (and every lone image directly in characters/) becomes a
            playable character. characters/<id>/character.json is optional — without it the folder's
            sprite.* (or its first image) is used as a still, and the game animates it by itself.
            With "pxo" in character.json the actions come straight from a Pixelorama project: each
            animation tag (idle / walk / attack / hurt / win) becomes that action, with its frame timing.
"""
import base64
import json
import pathlib
import re
import struct
import zipfile
import zlib

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "game.src.html"
OUT = ROOT / "index.html"
STAGES = ROOT / "stages"
CHARACTERS = ROOT / "characters"

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
BIG = 3 * 1024 * 1024  # warn above this: every byte is inlined into index.html
TEMPLATES = ("full", "meisho", "fuuin", "shio")  # 型 ids the game knows

# Text files inserted verbatim (inline <script> bodies).
TEXTS = {
    "{{SFX_JS}}": "parts/sfx.js",
}


def is_image(p: pathlib.Path) -> bool:
    return p.is_file() and p.suffix.lower() in MIME


def is_stage_image(p: pathlib.Path) -> bool:
    return is_image(p) and not p.name.startswith("preview_") and not p.stem.endswith(("_mask", "_ink", "_map"))


def companion(e: dict, key: str, suffix: str):
    """A stage's extra picture: named in stages.json, or <stem>_<suffix>.* next to it."""
    if e.get(key):
        return STAGES / e[key]
    stem = pathlib.Path(e["file"]).stem
    for ext in MIME:
        p = STAGES / f"{stem}_{suffix}{ext}"
        if p.is_file():
            return p
    return None


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
        if is_stage_image(p) and p.name not in listed:
            manifest.append({"id": p.stem, "name": p.stem, "file": p.name})
    out = []
    for e in manifest:
        d = {"id": e.get("id") or pathlib.Path(e["file"]).stem, "name": e.get("name") or e["file"],
             "desc": e.get("desc", ""), "src": data_uri(STAGES / e["file"])}
        if e.get("template"):
            if e["template"] not in TEMPLATES:
                raise SystemExit(f"stages.json: unknown template {e['template']!r} (use one of {', '.join(TEMPLATES)})")
            d["template"] = e["template"]
        mask = STAGES / e["mask"] if e.get("mask") else STAGES / f"{pathlib.Path(e['file']).stem}_mask.png"
        if e.get("mask") or mask.is_file():
            d["mask"] = data_uri(mask)
        if e.get("hd"):
            d["hd"] = True
            ink = companion(e, "ink", "ink")
            if ink:
                d["ink"] = data_uri(ink)
        tmap = companion(e, "map", "map")
        if tmap:
            d["map"] = data_uri(tmap)
        out.append(d)
    return out


# ── Pixelorama (.pxo) ──
# A .pxo (Pixelorama 1.0+) is a ZIP: data.json describes the canvas, layers, frames and tags, and every cel's
# pixels are a raw RGBA8 buffer at image_data/frames/<frame>/layer_<layer> (both counted from 1). Since
# Pixelorama 1.2.4 a cel may be smaller than the canvas ("image_size") and placed at "offset".
ANIMS = ("idle", "idle2", "walk", "walk_up", "walk_down", "attack", "attack2", "kick", "kick2",
         "hurt", "hurt_s", "down", "getup", "ko", "win")


def _vec(text, default):
    if isinstance(text, (list, tuple)) and len(text) == 2:
        return int(text[0]), int(text[1])
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", str(text or ""))
    return (int(float(m.group(1))), int(float(m.group(2)))) if m else default


def read_pxo(path: pathlib.Path) -> dict:
    try:
        z = zipfile.ZipFile(path)
        d = json.loads(z.read("data.json").decode("utf-8"))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as e:
        raise SystemExit(f"{path.relative_to(ROOT)}: not a Pixelorama 1.0+ project ({e})")
    w, h = int(d["size_x"]), int(d["size_y"])
    if int(d.get("color_mode", 5)) not in (5, -1, 40, 48):  # RGBA8, or indexed (stored as RGBA8)
        raise SystemExit(f"{path.relative_to(ROOT)}: only RGBA / indexed colour projects are supported")
    layers, names = d.get("layers", []), set(z.namelist())

    def shown(i):  # a layer and all of its parent groups visible
        while i >= 0:
            if not layers[i].get("visible", True):
                return False
            i = int(layers[i].get("parent", -1))
        return True

    frames, durations = [], []
    for fi, fr in enumerate(d.get("frames", [])):
        canvas = bytearray(w * h * 4)
        for li, cel in enumerate(fr.get("cels", [])):
            if li >= len(layers) or int(layers[li].get("type", 0)) != 0 or not shown(li):
                continue
            name = f"image_data/frames/{fi + 1}/layer_{li + 1}"
            if name not in names:
                continue
            iw, ih = _vec(cel.get("image_size"), (w, h))
            ox, oy = _vec(cel.get("offset"), (0, 0))
            data = z.read(name)
            if len(data) != iw * ih * 4:
                raise SystemExit(f"{path.relative_to(ROOT)}: frame {fi + 1} layer {li + 1} has an unexpected size")
            op = float(cel.get("opacity", 1.0)) * float(layers[li].get("opacity", 1.0))
            if (iw, ih, ox, oy) == (w, h, 0, 0) and op >= 1 and not any(canvas):
                canvas[:] = data  # the usual case: one full-size layer
                continue
            for y in range(ih):
                cy = y + oy
                if not 0 <= cy < h:
                    continue
                for x in range(iw):
                    cx = x + ox
                    s = (y * iw + x) * 4
                    a = data[s + 3] * op / 255
                    if a <= 0 or not 0 <= cx < w:
                        continue
                    t = (cy * w + cx) * 4
                    da = canvas[t + 3] / 255
                    oa = a + da * (1 - a)
                    for c in range(3):  # straight-alpha "over"
                        canvas[t + c] = round((data[s + c] * a + canvas[t + c] * da * (1 - a)) / oa)
                    canvas[t + 3] = round(oa * 255)
        frames.append(bytes(canvas))
        durations.append(float(fr.get("duration", 1.0)))
    fps = float(d.get("fps", 6.0)) or 6.0
    tags = {t["name"]: (int(t["from"]) - 1, int(t["to"]) - 1) for t in d.get("tags", [])}
    return {"w": w, "h": h, "frames": frames, "secs": [v / fps for v in durations], "tags": tags}


def png_uri(w: int, h: int, rgba: bytes) -> str:
    raw = b"".join(b"\x00" + rgba[y * w * 4:(y + 1) * w * 4] for y in range(h))
    chunk = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def strip(px: dict, a: int, b: int) -> str:
    """Frames a..b of a project side by side, as one PNG."""
    w, h, n = px["w"], px["h"], b - a + 1
    rows = []
    for y in range(h):
        rows.append(b"".join(px["frames"][i][y * w * 4:(y + 1) * w * 4] for i in range(a, b + 1)))
    return png_uri(w * n, h, b"".join(rows))


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
    if meta.get("pxo"):  # actions from a Pixelorama project, one animation tag each
        px = read_pxo(folder / meta["pxo"])
        for key in ANIMS:
            spec = meta.get(key) if isinstance(meta.get(key), dict) else {}
            tag = spec.get("tag", key)
            if tag not in px["tags"]:
                if key == "walk":
                    raise SystemExit(f"characters/{cid}/{meta['pxo']}: no animation tag named {tag!r}")
                continue
            a, b = px["tags"][tag]
            d[key] = {"src": strip(px, a, b), "frames": b - a + 1, "durs": [round(v, 3) for v in px["secs"][a:b + 1]]}
            for opt in ("loop", "dt", "windup"):
                if opt in spec:
                    d[key][opt] = spec[opt]
    for key in ("walk", "attack"):
        if key in meta and not meta.get("pxo"):
            sheet = meta[key]
            frames = int(sheet.get("frames", 1))
            if frames < 1:
                raise SystemExit(f"characters/{cid}: {key}.frames must be ≥ 1")
            d[key] = {"src": data_uri(folder / sheet["file"]), "frames": frames}
            for opt in ("loop", "dt"):
                if opt in sheet:
                    d[key][opt] = sheet[opt]
    for opt in ("feet", "height", "body", "pixelated", "faces"):
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
