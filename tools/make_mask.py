#!/usr/bin/env python3
"""ステージ画像から「ギミック指定の画像」（マスク）のたたき台を作る。

使い方:
    python3 tools/make_mask.py stages/竹林.png            # → stages/竹林_mask.png
    python3 tools/make_mask.py stages/竹林.png --landmarks 4 --springs 3

できたマスクは元の画像と同じ大きさで、背景は透明。次の純色で塗られている:
    黄 (255,255,0)   名所。ひとかたまりが1つの名所（同じ色の大きな区画を自動で選ぶ）
    赤 (255,0,0)     墨溜まり。ひとかたまりが1つ（暗く開けた場所を自動で選ぶ）
    緑 (0,255,0)     スタート位置
自分で描き足せる色:
    青 (0,0,255)     低い土地（潮が最初に満ちる）
    白 (255,255,255) 高い土地（潮が来ない）
ペイントソフトで直して（ぼかしやアンチエイリアスは切る）、python3 build.py を実行すれば反映される。
選ばれ方はゲームの自動判定（名所めぐり・墨溜まり）とほぼ同じ。
"""
import argparse
import pathlib
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "stages"))
from check import classify_cells, flood, WALL, CLOSED, WATER, FLOOR  # noqa: E402  (same terrain rules as the game)

YELLOW, RED, GREEN = (255, 255, 0, 255), (255, 0, 0, 255), (0, 255, 0, 255)
SHU, KIN, MIDORI, AI, MURASAKI, SHIRO, NOFAM = range(7)


def families(hsl, cell_rgb):
    """Colour family per cell (same rule as famOf in the game)."""
    h, _, l = hsl
    c = cell_rgb.max(-1) - cell_rgb.min(-1)
    fam = np.full(h.shape, NOFAM, np.int8)
    fam[(c < 0.12) & (l >= 0.72)] = SHIRO
    col = c >= 0.12
    fam[col & ((h < 15) | (h >= 330))] = SHU
    fam[col & (h >= 15) & (h < 50) & (l >= 0.45) & (c >= 0.2)] = KIN
    fam[col & (h >= 50) & (h < 70)] = KIN
    fam[col & (h >= 70) & (h < 170)] = MIDORI
    fam[col & (h >= 170) & (h < 260)] = AI
    fam[col & (h >= 260) & (h < 330)] = MURASAKI
    return fam


def components(mask):
    """4-connected components of a boolean grid → list of [(y, x), ...]."""
    ch, cw = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    for y in range(ch):
        for x in range(cw):
            if not mask[y, x] or seen[y, x]:
                continue
            comp, q = [], deque([(y, x)])
            seen[y, x] = True
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < ch and 0 <= nx < cw and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            out.append(comp)
    return out


def open_cell(play, y, x, r):
    ch, cw = play.shape
    return y - r >= 0 and x - r >= 0 and y + r < ch and x + r < cw and play[y - r:y + r + 1, x - r:x + r + 1].all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out")
    ap.add_argument("--landmarks", type=int, default=3)
    ap.add_argument("--springs", type=int, default=3)
    args = ap.parse_args()
    src = pathlib.Path(args.image)
    img = Image.open(src)
    kind, hsl = classify_cells(img)
    final = flood(kind)
    play = (final != WALL) & (final != CLOSED)
    ch, cw = play.shape
    # cell colours (same downscale as check.py)
    from check import art_grid
    a = np.asarray(art_grid(img), dtype=np.float64)[: ch * 2, : cw * 2] / 255.0
    cell_rgb = a.reshape(ch, 2, cw, 2, 3).mean(axis=(1, 3))
    fam = families(hsl, cell_rgb)
    light = hsl[2]
    # start: open floor nearest the middle
    my, mx = (ch - 1) / 2, (cw - 1) / 2
    start = min(((y, x) for y in range(ch) for x in range(cw) if final[y, x] == FLOOR and open_cell(play, y, x, 1)),
                key=lambda p: (p[0] - my) ** 2 + (p[1] - mx) ** 2, default=None)
    # 名所: big patches of one colour family, apart from each other, not holding the start
    cands = []
    for f in range(6):
        for comp in components(play & (fam == f)):
            if len(comp) < 24 or (start and start in comp):
                continue
            ys, xs = zip(*comp)
            cy, cx = sum(ys) / len(ys), sum(xs) / len(xs)
            comp.sort(key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
            comp = comp[:180]
            cands.append((min(len(comp), 200) * (0.5 if f == SHIRO else 1), cy, cx, comp))
    cands.sort(key=lambda k: -k[0])
    marks = []
    for _, cy, cx, comp in cands:
        if len(marks) >= args.landmarks:
            break
        if all((cy - oy) ** 2 + (cx - ox) ** 2 >= 196 for oy, ox, _ in marks):
            marks.append((cy, cx, comp))
    # 墨溜まり: the darkest open ground (water counts as higher), away from the start and each other
    order = sorted(((light[y, x] + (0.5 if final[y, x] == WATER else 0), y, x) for y in range(ch) for x in range(cw)
                    if play[y, x] and open_cell(play, y, x, 1) and (not start or (y - start[0]) ** 2 + (x - start[1]) ** 2 >= 324)))
    springs = []
    for _, y, x in order:
        if len(springs) >= args.springs:
            break
        if all((y - sy) ** 2 + (x - sx) ** 2 >= 196 for sy, sx in springs):
            springs.append((y, x))
    # paint at the picture's own size
    W, H = img.size
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)
    box = lambda y, x: (x * W / cw, y * H / ch, (x + 1) * W / cw - 1, (y + 1) * H / ch - 1)
    for _, _, comp in marks:
        for y, x in comp:
            d.rectangle(box(y, x), fill=YELLOW)
    for y, x in springs:
        x0, y0, x1, y1 = box(y, x)
        d.ellipse((x0 - (x1 - x0), y0 - (y1 - y0), x1 + (x1 - x0), y1 + (y1 - y0)), fill=RED)
    if start:
        d.ellipse(box(*start), fill=GREEN)
    dst = pathlib.Path(args.out) if args.out else src.with_name(f"{src.stem}_mask.png")
    out.save(dst)
    print(f"wrote {dst}  名所 {len(marks)} / 墨溜まり {len(springs)} / スタート {'あり' if start else 'なし'}")


if __name__ == "__main__":
    main()
