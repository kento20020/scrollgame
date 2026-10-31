#!/usr/bin/env python3
"""Generate characters/aoi/walk.png / attack.png + previews from characters/aoi/sprite.png.

All edits are done on the "body" layer (outline stripped); the 1px light outline
is regenerated for every frame (transparent px 4-adjacent to body -> outline).
"""
import math
import numpy as np
from PIL import Image

import pathlib
ASSETS = str(pathlib.Path(__file__).resolve().parent.parent / 'characters' / 'aoi')
SRC = f'{ASSETS}/sprite.png'
H, W = 54, 28
OL = np.array([232, 226, 214, 255], dtype=np.uint8)
TRANSP = np.zeros(4, dtype=np.uint8)

# source palette letters (see char map)
C = dict(
    K=(1, 0, 17), n=(39, 46, 83), G=(88, 92, 111), d=(15, 14, 39), m=(25, 29, 54),
    S=(243, 215, 196), g=(126, 113, 119), H=(141, 144, 155), s=(215, 193, 183),
    k=(33, 37, 68), N=(6, 6, 24), J=(183, 159, 153), p=(66, 62, 77), I=(38, 39, 74),
    i=(1, 3, 21),
)


def col(ch):
    return np.array(list(C[ch]) + [255], dtype=np.uint8)


def load():
    return np.array(Image.open(SRC).convert('RGBA'))


def strip_outline(a):
    b = a.copy()
    m = np.all(a == OL, axis=2)
    b[m] = TRANSP
    return b


def add_outline(body):
    out = body.copy()
    opaque = body[..., 3] > 0
    for y in range(H):
        for x in range(W):
            if opaque[y, x]:
                continue
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and opaque[yy, xx]:
                    out[y, x] = OL
                    break
    return out


# ---------------------------------------------------------------- transforms
def shift_rows(body, dx_row, segs=None):
    """Row-wise horizontal shear.

    dx_row[y]  : default shift of row y (whole row, rigid).
    segs[y]    : optional list of (xa, xb, dx) sub-segments painted AFTER the row
                 body, so they take priority (used for hair tips / hands).
                 Where a moved segment leaves a 1+px gap against an attached
                 neighbour, the gap is filled by stretching the segment's edge.
    """
    segs = segs or {}
    out = np.zeros_like(body)
    for y in range(H):
        row = body[y]
        d = dx_row[y]
        base_mask = np.ones(W, dtype=bool)
        sl = segs.get(y, [])
        for xa, xb, _ in sl:
            base_mask[xa:xb + 1] = False
        # base part
        for x in range(W):
            if base_mask[x] and row[x, 3] > 0:
                nx = x + d
                if 0 <= nx < W:
                    out[y, nx] = row[x]
        # stretch fill for gaps between base part and segments
        for xa, xb, dd in sl:
            # left neighbour attached?
            if xa - 1 >= 0 and row[xa - 1, 3] > 0 and row[xa, 3] > 0:
                left_end = xa - 1 + d
                seg_start = xa + dd
                for x in range(left_end + 1, seg_start):
                    if 0 <= x < W:
                        out[y, x] = row[xa]
            if xb + 1 < W and row[xb + 1, 3] > 0 and row[xb, 3] > 0:
                right_start = xb + 1 + d
                seg_end = xb + dd
                for x in range(seg_end + 1, right_start):
                    if 0 <= x < W:
                        out[y, x] = row[xb]
        for xa, xb, dd in sl:
            for x in range(xa, xb + 1):
                if row[x, 3] > 0:
                    nx = x + dd
                    if 0 <= nx < W:
                        out[y, nx] = row[x]
    return out


def lift_foot(body, side):
    """Raise one foot by 1px (drop the ankle row 49 for that foot, shoe moves up)."""
    out = body.copy()
    xs = range(0, 16) if side == 'L' else range(16, W)
    for x in xs:
        col_old = body[:, x].copy()
        new = col_old.copy()
        new[48] = col_old[48]
        new[49] = col_old[50]
        new[50] = col_old[51]
        new[51] = col_old[52]
        new[52] = TRANSP
        out[:, x] = new
    return out


def raise_hand_fan(body, k):
    """Move hand + fan (region R) up by k px, leaving the arm slightly foreshortened."""
    out = body.copy()
    R = np.zeros((H, W), dtype=bool)
    for y in range(29, 38):
        for x in range(0, 10):
            R[y, x] = body[y, x, 3] > 0
    # clear old R pixels
    for y, x in zip(*np.where(R)):
        out[y, x] = TRANSP
    # paste shifted
    for y, x in zip(*np.where(R)):
        ny = y - k
        if ny >= 0:
            out[ny, x] = body[y, x]
    return out


def paste(body, pixels):
    """pixels: list of (x, y, letter)."""
    out = body.copy()
    for x, y, ch in pixels:
        out[y, x] = col(ch)
    return out


def clear(body, pixels):
    out = body.copy()
    for x, y in pixels:
        out[y, x] = TRANSP
    return out


def profile(up, low, up_end=29, low_start=43):
    d = [0] * H
    for y in range(H):
        if y <= up_end:
            d[y] = up
        elif y >= low_start:
            d[y] = low
    return d


def tip_segs(up, tl, tr, rows_l=range(18, 23), rows_r=range(17, 23)):
    """hair tips: absolute total shift tl (left tips) / tr (right tips)."""
    segs = {}
    for y in rows_l:
        segs.setdefault(y, []).append((0, 10, tl))
    for y in rows_r:
        segs.setdefault(y, []).append((22, W - 1, tr))
    return segs


# ---------------------------------------------------------------- frame builders
def make_walk_frames(body):
    frames = [None] * 4
    frames[0] = None  # idle = original (handled by caller)

    # f1: step A. upper body +1, hem/feet -1, right foot raised, hair tips lag left
    b = lift_foot(body, 'R')
    b = shift_rows(b, profile(+1, -1), tip_segs(+1, 0, 0))
    frames[1] = b

    # f2: pass. neutral body, fan hand raised 2px, hair flares outward
    b = raise_hand_fan(body, 2)
    b = shift_rows(b, profile(0, 0), tip_segs(0, -1, +1, range(20, 23), range(19, 23)))
    frames[2] = b

    # f3: step B (mirror of f1). left foot raised
    b = lift_foot(body, 'L')
    b = shift_rows(b, profile(-1, +1), tip_segs(-1, 0, 0))
    frames[3] = b
    return frames


def fan_wedge(body, apex=(9, 31), r=7.4, phi0=-4, phi1=72):
    """Open fan wedge: returns body with old fan removed and wedge painted."""
    out = body.copy()
    # remove the old fan blade (everything left of the hand in rows 31-37)
    for y in range(31, 38):
        for x in range(0, 9):
            out[y, x] = TRANSP
    ax, ay = apex
    px = []
    for y in range(ay, ay + 10):
        for x in range(0, ax + 1):
            u = ax - x
            v = y - ay
            rr = math.hypot(u, v)
            if rr > r or rr < 1.0:
                continue
            phi = math.degrees(math.atan2(v, u))
            if phi < phi0 or phi > phi1:
                continue
            px.append((x, y, rr, phi))
    pattern = ['K', 'G', 's', 'G', 'g', 'G', 'K']   # per ~11deg bin: edge, blade, highlight ribs
    for x, y, rr, phi in px:
        b = min(int((phi - phi0) // 11), len(pattern) - 1)
        ch = pattern[b]
        if rr > r - 1.1:
            ch = 'K'                       # outer rim
        if rr < 2.2:
            ch = 'k'                       # hinge
        out[y, x] = col(ch)
    return out


def make_attack_frames(body):
    # frame 0: fan raised 2px, opened a bit wider (+3 fan px)
    b = raise_hand_fan(body, 2)
    b = shift_rows(b, [0] * H)
    extra = [(2, 31, 'p'), (1, 32, 'G'), (4, 34, 's')]
    b = paste(b, extra)
    a0 = b
    # frame 1: fan swept forward (wide wedge), slight lean
    b = fan_wedge(body)
    b = shift_rows(b, profile(+1, 0), tip_segs(+1, 0, 0))
    a1 = b
    return [a0, a1]


# ---------------------------------------------------------------- output helpers
def to_img(a):
    return Image.fromarray(a, 'RGBA')


def sheet(frames):
    out = np.zeros((H, W * len(frames), 4), dtype=np.uint8)
    for i, f in enumerate(frames):
        out[:, i * W:(i + 1) * W] = f
    return out


def preview(frames, scale=8, gray=(128, 128, 128), sep=(40, 40, 40)):
    n = len(frames)
    pw = W * scale
    ph = H * scale
    img = Image.new('RGB', (pw * n + (n - 1), ph), gray)
    for i, f in enumerate(frames):
        fi = to_img(f)
        bg = Image.new('RGBA', fi.size, gray + (255,))
        bg.alpha_composite(fi)
        big = bg.resize((pw, ph), Image.NEAREST).convert('RGB')
        img.paste(big, (i * (pw + 1), 0))
        if i < n - 1:
            for y in range(ph):
                img.putpixel((i * (pw + 1) + pw, y), sep)
    return img


def main():
    src = load()
    body = strip_outline(src)
    walk_b = make_walk_frames(body)
    walk = [src] + [add_outline(b) for b in walk_b[1:]]
    atk = [add_outline(b) for b in make_attack_frames(body)]

    to_img(sheet(walk)).save(f'{ASSETS}/walk.png')
    to_img(sheet(atk)).save(f'{ASSETS}/attack.png')
    preview(walk).save(f'{ASSETS}/preview_walk.png')
    preview(atk).save(f'{ASSETS}/preview_attack.png')

    order = [1, 2, 3, 2]
    gifs = [preview([walk[i]]) for i in order]
    # shared palette so colours do not shift between frames
    gifs[0].save(f'{ASSETS}/preview_walk.gif', save_all=True, append_images=gifs[1:],
                 duration=150, loop=0, disposal=2)
    return walk, atk


if __name__ == '__main__':
    main()
    print('done')
