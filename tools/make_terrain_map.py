#!/usr/bin/env python3
"""Draft a terrain map for a detailed stage picture (stages.json "hd": true).

usage:
  python3 tools/make_terrain_map.py stages/foo.webp            # → stages/foo_map.png and stages/preview_foo_map.png

Detailed art is full of shadows and dark outlines, which the game's colour reading would turn into walls, so a
detailed stage brings its own terrain map: the picture's shape in four flat colours, one per cell
(64 cells across, like the game). This script paints a first draft from the picture's colours:

  water  teal / blue (hue 150-205, some saturation)
  floor  paving, earth, gravel: low saturation, middle lightness
  朱     vermilion paint — kept only for big red runs touching water (bridges) and small red specks standing
         on the floor (lanterns); other red (house fronts, gates) is wall
  wall   everything else: roofs, walls, trees, rocks

then fills the pond's inside (posts, boats, lily pads become water), closes gaps in bridge decks and turns lone
dark specks on the paving back into floor. Open preview_foo_map.png (the map over the picture), fix the map in
any paint program with the same four colours, and run build.py.
"""
import sys
import pathlib

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

FLOOR, WATER, WARM, WALL = 0, 1, 2, 3
COLS = np.array([(141, 136, 123), (143, 201, 184), (208, 83, 58), (11, 15, 28)], np.uint8)


def hls(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    l = (mx + mn) / 2
    d = mx - mn
    s = np.where(d == 0, 0, d / (1 - np.abs(2 * l - 1) + 1e-9))
    rc, gc, bc = (mx - r) / (d + 1e-9), (mx - g) / (d + 1e-9), (mx - b) / (d + 1e-9)
    h = np.where(r == mx, bc - gc, np.where(g == mx, 2 + rc - bc, 4 + gc - rc))
    return (h / 6 % 1) * 360, l, s


def draft(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB")).astype(float) / 255
    ih, iw = a.shape[:2]
    cw = 64 if iw >= ih else max(2, round(128 * iw / ih / 2) * 2) // 2
    ch = max(2, round(128 * ih / iw / 2) * 2) // 2 if iw >= ih else 64
    h, l, s = hls(a)
    water = (h > 150) & (h < 205) & (s > 0.16) & (l > 0.28) & (l < 0.8)
    pave = (s < 0.3) & (l > 0.38) & (l < 0.78) & ((h < 50) | (h > 330) | (s < 0.12))
    red = ((h < 20) | (h > 345)) & (s > 0.33) & (l > 0.3) & (l < 0.6)
    dark = l < 0.36
    Y = np.broadcast_to((np.arange(ih) * ch // ih)[:, None], (ih, iw))
    X = np.broadcast_to((np.arange(iw) * cw // iw)[None, :], (ih, iw))
    cnt = np.zeros((ch, cw))
    np.add.at(cnt, (Y, X), 1)

    def frac(m):
        o = np.zeros((ch, cw))
        np.add.at(o, (Y, X), m)
        return o / cnt

    fw, fp, fr, fd = frac(water), frac(pave), frac(red), frac(dark)
    T = np.full((ch, cw), WALL)
    T[fp > 0.45] = FLOOR
    T[fr > 0.35] = WARM
    T[fd > 0.4] = WALL
    T[fw > 0.4] = WATER
    lab, n = ndi.label(T == WARM)
    bridge = np.zeros_like(T, bool)
    for i in range(1, n + 1):
        m = lab == i
        ring = ndi.binary_dilation(m) & ~m
        if m.sum() >= 20 and (T[ring] == WATER).any():
            bridge |= m
            continue
        if m.sum() <= 4 and (T[ring] == FLOOR).mean() >= 0.6:
            continue
        T[m] = WALL
    # bridge decks: close the gaps between their red rails
    deck = ndi.binary_closing(bridge, structure=np.ones((3, 3), bool), iterations=2) & ((T == WATER) | (T == FLOOR) | bridge)
    T[deck] = WARM
    T[ndi.binary_fill_holes((T == WATER) | (T == WARM)) & (T != WATER) & (T != WARM)] = WATER
    for _ in range(2):
        fl = ndi.convolve((T == FLOOR).astype(int), np.ones((3, 3), int), mode="constant")
        T[(T == WALL) & (fl >= 6)] = FLOOR
    # lanterns: small specks of wall standing on the paving
    lab, n = ndi.label(T == WALL)
    for i in range(1, n + 1):
        m = lab == i
        ring = ndi.binary_dilation(m) & ~m
        if m.sum() <= 8 and (T[ring] == FLOOR).mean() >= 0.7:
            T[m] = WARM
    return T


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = pathlib.Path(sys.argv[1])
    img = Image.open(src)
    T = draft(img)
    ch, cw = T.shape
    iw, ih = img.size
    full = COLS[T[(np.arange(ih) * ch // ih)[:, None], (np.arange(iw) * cw // iw)[None, :]]]
    out = src.with_name(src.stem + "_map.png")
    Image.fromarray(full).convert("P", palette=Image.ADAPTIVE, colors=4).save(out, optimize=True)
    check = img.convert("RGBA")
    ov = Image.fromarray(np.dstack([full, np.full(full.shape[:2], 140, np.uint8)]))
    check.alpha_composite(ov)
    dr = ImageDraw.Draw(check)
    for x in range(0, cw + 1, 8):
        dr.line([(x * iw / cw, 0), (x * iw / cw, ih)], fill=(255, 255, 255, 100))
    for y in range(0, ch + 1, 8):
        dr.line([(0, y * ih / ch), (iw, y * ih / ch)], fill=(255, 255, 255, 100))
    check.convert("RGB").save(src.with_name("preview_" + src.stem + "_map.png"))
    pct = {k: round(float(np.mean(T == i)) * 100, 1) for i, k in enumerate(["floor", "water", "warm", "wall"])}
    print(f"wrote {out} ({cw}x{ch} cells) {pct}")


if __name__ == "__main__":
    main()
