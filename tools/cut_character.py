#!/usr/bin/env python3
"""キャラクター一覧画像から1人を切り抜き、ゲーム用スプライトにする。

使い方:
    python tools/cut_character.py 一覧画像.jpg --box 118 20 196 130

処理の流れ:
  1. 指定範囲を切り出し、外周から背景色を塗りつぶして透明化
  2. いちばん大きい塊だけ残す（キャプション文字などを除去）
  3. 輪郭に残った明るいにじみを削る
  4. 24色に減色 → characters/aoi/portrait.png（タイトル画面用）
  5. 2x2の多数決で半分に縮小し、1px の明るい縁取りを付ける
     → characters/aoi/sprite.png（ゲーム内用）

※ 画像を characters/ に置くだけでも遊べる（背景の除去はゲームが自動で行う）。
  このスクリプトは、ドット絵として作り込んだ版を作るためのもの。
"""
import argparse
import pathlib
from collections import Counter, deque

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTLINE = (232, 226, 214, 255)


def cut(src, box, bg):
    im = Image.open(src).convert("RGB")
    x0, y0, x1, y1 = box
    c = np.array(im.crop(box)).astype(int)
    H, W, _ = c.shape
    bg = np.array(bg)

    def is_bg(p):
        return np.abs(p - bg).sum() < 70 and p.sum() > 560

    mask = np.zeros((H, W), bool)
    q = deque([(y, x) for x in range(W) for y in (0, H - 1)] +
              [(y, x) for y in range(H) for x in (0, W - 1)])
    while q:
        y, x = q.popleft()
        if not (0 <= y < H and 0 <= x < W) or mask[y, x] or not is_bg(c[y, x]):
            continue
        mask[y, x] = True
        q.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
    fg = ~mask

    lab = np.zeros((H, W), int)
    sizes, n = {}, 0
    for yy in range(H):
        for xx in range(W):
            if fg[yy, xx] and not lab[yy, xx]:
                n += 1
                s, q = 0, deque([(yy, xx)])
                while q:
                    y, x = q.popleft()
                    if not (0 <= y < H and 0 <= x < W) or not fg[y, x] or lab[y, x]:
                        continue
                    lab[y, x] = n
                    s += 1
                    q.extend([(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
                sizes[n] = s
    keep = lab == max(sizes, key=sizes.get)
    ys, xs = np.where(keep)
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = c
    rgba[..., 3] = keep * 255
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def clean_fringe(a):
    a = a.astype(int)
    al = a[..., 3] > 0
    for _ in range(2):
        edge = np.zeros_like(al)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            edge |= ~np.roll(np.roll(al, dy, 0), dx, 1)
        rgb = a[..., :3]
        mx, mn = rgb.max(-1), rgb.min(-1)
        al &= ~(edge & (mx > 200) & ((mx - mn) < 60))
    a[..., 3] = al * 255
    return a.astype(np.uint8)


def quantize(img, n):
    alpha = img.getchannel("A").point(lambda v: 255 if v > 127 else 0)
    q = img.convert("RGB").quantize(colors=n, method=Image.MEDIANCUT,
                                    dither=Image.Dither.NONE).convert("RGB")
    q.putalpha(alpha)
    return q


def half_majority(a):
    h, w = a.shape[0] // 2, a.shape[1] // 2
    out = np.zeros((h, w, 4), np.uint8)
    for y in range(h):
        for x in range(w):
            blk = [tuple(int(v) for v in px) for px in a[y * 2:y * 2 + 2, x * 2:x * 2 + 2].reshape(-1, 4) if px[3]]
            if len(blk) >= 2:
                out[y, x] = max(Counter(blk).items(), key=lambda kv: (kv[1], -sum(kv[0][:3])))[0]
    return out


def add_outline(a):
    big = np.zeros((a.shape[0] + 2, a.shape[1] + 2, 4), np.uint8)
    big[1:-1, 1:-1] = a
    body = big[..., 3] > 0
    ring = np.zeros_like(body)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ring |= np.roll(np.roll(body, dy, 0), dx, 1)
    big[ring & ~body] = OUTLINE
    return big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--box", nargs=4, type=int, default=[118, 20, 196, 130])
    ap.add_argument("--bg", nargs=3, type=int, default=[245, 237, 214])
    ap.add_argument("--out", default=str(ROOT / "characters" / "aoi"))
    args = ap.parse_args()

    raw = clean_fringe(cut(args.src, tuple(args.box), args.bg))
    portrait = quantize(Image.fromarray(raw), 24)
    out = pathlib.Path(args.out)
    portrait.save(out / "portrait.png")
    sprite = add_outline(half_majority(np.array(portrait)))
    Image.fromarray(sprite).save(out / "sprite.png")
    print("portrait", portrait.size, "sprite", sprite.shape[1::-1])


if __name__ == "__main__":
    main()
