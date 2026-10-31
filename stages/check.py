#!/usr/bin/env python3
"""Reproduce the game's image -> terrain classification for a stage image.

usage:
  python3 check.py path/to/image.png [--preview out.png]
  python3 check.py --all          # every stage image here: (re)writes preview_*.png and the terrain in stages.json

Pipeline (same as the game):
  1. downscale so the long side is 128 art px (BOX filter, aspect kept)
  2. terrain cell = 2x2 art px, average RGB -> hue, lightness L, chroma C=max-min
  3. classify: wall (L<0.17) > water (hue 170-255, C>=0.15, L<=0.80)
     > warm (red hue & C>=0.28, or orange hue & C>=0.40; L<=0.78) > floor
  4. 4-neighbour flood fill over non-wall cells; largest region is playable,
     everything else (closed pockets) counts as wall.
"""
import json
import os
import sys
from collections import deque

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

FLOOR, WALL, WATER, WARM, CLOSED = 0, 1, 2, 3, 4
PREVIEW_COLORS = {
    FLOOR: (0xC8, 0xC2, 0xB0),
    WALL: (0x0B, 0x0F, 0x1C),
    WATER: (0x8F, 0xC9, 0xB8),
    WARM: (0xD0, 0x53, 0x3A),
    CLOSED: (0x33, 0x3A, 0x50),
}


def art_grid(img, long_side=128):
    """Downscale to the art grid (long side = 128, BOX filter)."""
    w, h = img.size
    s = long_side / max(w, h)
    aw, ah = max(1, round(w * s)), max(1, round(h * s))
    return img.convert("RGB").resize((aw, ah), Image.BOX)


def rgb_to_hsl(rgb):
    """rgb: float array (...,3) in 0..1 -> (h 0..360, s 0..1, l 0..1)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    d = mx - mn
    l = (mx + mn) / 2
    den = 1 - np.abs(2 * l - 1)
    s = np.where(d < 1e-9, 0.0, d / np.where(den < 1e-9, 1, den))
    dd = np.where(d < 1e-9, 1, d)
    h = np.where(
        mx == r, ((g - b) / dd) % 6,
        np.where(mx == g, (b - r) / dd + 2, (r - g) / dd + 4))
    h = np.where(d < 1e-9, 0.0, h * 60.0)
    return h, s, l


def classify_cells(img):
    """Return (kind[ch,cw], hsl) per terrain cell before flood fill."""
    a = np.asarray(art_grid(img), dtype=np.float64) / 255.0
    ah, aw = a.shape[:2]
    ch, cw = ah // 2, aw // 2
    a = a[: ch * 2, : cw * 2]
    cell = a.reshape(ch, 2, cw, 2, 3).mean(axis=(1, 3))
    h, s, l = rgb_to_hsl(cell)
    c = cell.max(-1) - cell.min(-1)  # chroma (same rule as the game)
    wall = l < 0.17
    water = (~wall) & (h >= 170) & (h <= 255) & (c >= 0.15) & (l <= 0.80)
    warm = (~wall) & (~water) & (l <= 0.78) & (
        (((h <= 25) | (h >= 340)) & (c >= 0.28)) | ((h > 25) & (h <= 45) & (c >= 0.40)))
    kind = np.full((ch, cw), FLOOR, dtype=np.int8)
    kind[wall] = WALL
    kind[water] = WATER
    kind[warm] = WARM
    return kind, (h, s, l)


def flood(kind):
    """Largest 4-connected non-wall region; the rest becomes CLOSED."""
    ch, cw = kind.shape
    seen = np.zeros((ch, cw), bool)
    best = []
    for y in range(ch):
        for x in range(cw):
            if seen[y, x] or kind[y, x] == WALL:
                continue
            comp = []
            q = deque([(y, x)])
            seen[y, x] = True
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < ch and 0 <= nx < cw and not seen[ny, nx] and kind[ny, nx] != WALL:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(comp) > len(best):
                best = comp
    main = np.zeros((ch, cw), bool)
    for y, x in best:
        main[y, x] = True
    final = kind.copy()
    final[(kind != WALL) & (~main)] = CLOSED
    return final


def analyse(path):
    img = Image.open(path)
    kind, hsl = classify_cells(img)
    final = flood(kind)
    n = final.size
    nonwall = int((kind != WALL).sum())
    main = int(((final != WALL) & (final != CLOSED)).sum())
    pct = lambda k: round(100.0 * int((final == k).sum()) / n, 2)
    res = {
        "floor": pct(FLOOR),
        "water": pct(WATER),
        "warm": pct(WARM),
        "wall": round(100.0 * int(((final == WALL) | (final == CLOSED)).sum()) / n, 2),
        "closed": pct(CLOSED),
    }
    extra = {
        "cells": f"{final.shape[1]}x{final.shape[0]}",
        "largest_region_share_of_nonwall": round(100.0 * main / max(1, nonwall), 2),
        "unique_colors": int(len(np.unique(np.asarray(img.convert('RGB')).reshape(-1, 3), axis=0))),
    }
    return res, extra, final


def make_preview(img_path, final, out_path, scale=3):
    img = Image.open(img_path).convert("RGB")
    left = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    ch, cw = final.shape
    t = np.zeros((ch, cw, 3), np.uint8)
    for k, c in PREVIEW_COLORS.items():
        t[final == k] = c
    cell_px = left.width // cw
    right = Image.fromarray(t).resize((cw * cell_px, ch * cell_px), Image.NEAREST)
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), (20, 20, 20))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(out_path)


IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def stage_list():
    """stages.json entries (order, names, descriptions) plus any unlisted image in this folder — same rule as build.py."""
    path = os.path.join(HERE, "stages.json")
    manifest = json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else []
    listed = {e["file"] for e in manifest}
    for fn in sorted(os.listdir(HERE)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in IMG_EXT and not fn.startswith("preview_") and not stem.endswith("_mask") and fn not in listed:
            manifest.append({"id": stem, "name": stem, "file": fn})
    return manifest

TARGETS = {"floor": (50, 65), "water": (12, 25), "warm": (3, 8), "wall": (10, 25)}


def verdict(res, extra):
    bad = []
    for k, (lo, hi) in TARGETS.items():
        if not (lo <= res[k] <= hi):
            bad.append(f"{k}={res[k]} outside {lo}-{hi}")
    if extra["largest_region_share_of_nonwall"] < 90:
        bad.append(f"largest region {extra['largest_region_share_of_nonwall']}% < 90%")
    return bad


def main(argv):
    if "--all" in argv:
        out = stage_list()
        for e in out:
            p = os.path.join(HERE, e["file"])
            res, extra, final = analyse(p)
            make_preview(p, final, os.path.join(HERE, f"preview_{e['id']}.png"))
            print(e["id"], res, extra, verdict(res, extra) or "OK")
            e["terrain"] = res
        with open(os.path.join(HERE, "stages.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return
    if len(argv) < 2:
        print(__doc__)
        return
    p = argv[1]
    res, extra, final = analyse(p)
    print(json.dumps({"terrain_percent": res, **extra, "problems": verdict(res, extra)}, ensure_ascii=False, indent=2))
    if "--preview" in argv:
        make_preview(p, final, argv[argv.index("--preview") + 1])


if __name__ == "__main__":
    main(sys.argv)
