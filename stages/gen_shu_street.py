#!/usr/bin/env python3
"""Generate stages/shu_street.png  (朱の楼閣街) -- top-down night street, procedural pixel art.

Drawn on the 128x96 art grid, saved at 256x192 (2x nearest).  Terrain cells are 2x2 art px.
Light comes from the north-west; shadows fall to the south-east.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pxlib import Canvas, bayer, cell_mask_to_px, dist_to_mask, value_noise  # noqa: E402

SEED = 11
rng = np.random.default_rng(SEED)
CW, CH = 64, 48
W, H = 128, 96

PAL = {
    # cool night stone (kept desaturated so it never reads as water / warm)
    "st0": (190, 188, 200), "st1": (176, 174, 190), "st2": (160, 158, 176), "stj": (116, 114, 136),
    "sts": (128, 126, 148), "stsj": (90, 88, 112),
    "glow1": (214, 206, 178), "glow2": (234, 222, 172),
    # slate roofs (must stay lightness < 0.17 on average)
    "rf0": (8, 10, 20), "rf1": (20, 26, 46), "rf2": (38, 50, 82), "rf3": (48, 62, 98),
    # canal water
    "w0": (18, 70, 92), "w1": (34, 100, 122), "w2": (58, 134, 150), "w3": (126, 190, 198), "w4": (14, 58, 78),
    # vermilion
    "r0": (110, 24, 28), "r1": (178, 40, 36), "r2": (216, 74, 46), "r3": (240, 138, 70), "gold": (224, 198, 98),
    # greens, pot, wood
    "g0": (28, 78, 52), "g1": (56, 122, 70), "g2": (112, 170, 98),
    "pot": (128, 98, 80), "wd0": (92, 76, 66), "wd1": (134, 112, 92),
    "moss": (108, 134, 112),
}
cv = Canvas(PAL, W, H)
C = cv.c
STONE_N = ["st0", "st1", "st2", "glow1", "glow2"]
ST_IDS = [C(n) for n in STONE_N]
JOINT = C("stj")

# --------------------------------------------------------------------------------------
# layout (cell coordinates)
# --------------------------------------------------------------------------------------
HALF = 3.6


def canal_c(x):
    return 25.0 + 4.0 * math.sin((x + 6) / 9.5)


wmask = np.zeros((CH, CW), bool)
for x in range(CW):
    c = canal_c(x + 0.5)
    for y in range(CH):
        if abs(y + 0.5 - c) <= HALF:
            wmask[y, x] = True
for x in range(44, 48):                      # tributary
    for y in range(6, 22):
        wmask[y, x] = True
for y in range(CH):
    for x in range(CW):
        if ((x + 0.5 - 46) / 3.4) ** 2 + ((y + 0.5 - 3.6) / 3.4) ** 2 <= 1.0:   # north basin
            wmask[y, x] = True
        if ((x + 0.5 - 40) / 5.8) ** 2 + ((y + 0.5 - 43.2) / 3.0) ** 2 <= 1.0:  # south pond
            wmask[y, x] = True

VSTREETS = [(11, 14), (30, 33), (51, 54)]
BRIDGES = [("v", 11, 14), ("v", 30, 33), ("v", 51, 54), ("h", 43, 48, 8, 11)]
bridge_rects = []
for br in BRIDGES:
    if br[0] == "v":
        _, x0, x1 = br
        ys = [y for y in range(CH) if wmask[y, x0:x1 + 1].any() and 12 <= y <= 36]
        bridge_rects.append(("v", x0, x1, min(ys) - 1, max(ys) + 1))
    else:
        bridge_rects.append(br)
bridge_cells = np.zeros((CH, CW), bool)
for o, x0, x1, y0, y1 in bridge_rects:
    bridge_cells[y0:y1 + 1, x0:x1 + 1] = True
water_cells = wmask & ~bridge_cells

# buildings: groups of rects (x, y, w, h); later rects drawn over earlier ones
BUILDINGS = [
    [(1, 0, 10, 3), (1, 3, 3, 4), (8, 3, 3, 4)],           # N1a  U-compound, court open to the south
    [(16, 0, 11, 3), (16, 3, 3, 4), (24, 3, 3, 4)],        # N1b
    [(36, 0, 6, 6)],                                       # N1c
    [(1, 13, 9, 5), (6, 18, 4, 2)],                        # N2a
    [(17, 13, 10, 4)],                                     # N2b
    [(36, 13, 7, 3)],                                      # N2c
    [(56, 13, 7, 6)],                                      # N2d
    [(1, 39, 9, 3), (1, 35, 3, 4), (7, 35, 3, 4)],         # S1a  court open to the north (canal quay)
    [(18, 36, 8, 5)],                                      # S1b
    [(37, 27, 8, 6)],                                      # S2
    [(56, 34, 7, 5)],                                      # S3
    [(1, 45, 9, 3)],                                       # S4
    [(17, 45, 11, 3)],                                     # S5
    [(55, 45, 9, 3)],                                      # S6
]
bmask = np.zeros((CH, CW), bool)
for grp in BUILDINGS:
    for (x, y, w, h) in grp:
        bmask[y:y + h, x:x + w] = True
PAV = (55, 0, 8, 8)
PAV_CORE = (57, 2, 4, 4)
bmask[2:6, 57:61] = True
GATE_ROOFS = [(28, 3, 8, 3)]
COURTS = [(4, 3, 4, 4), (19, 3, 5, 4), (4, 35, 3, 4)]
# sanity: 1 cell clearance between buildings and water, nothing on bridges
dil = wmask.copy()
dil[1:] |= wmask[:-1]; dil[:-1] |= wmask[1:]; dil[:, 1:] |= wmask[:, :-1]; dil[:, :-1] |= wmask[:, 1:]
assert not (bmask & dil).any(), np.argwhere(bmask & dil)[:5]
assert not (bmask & bridge_cells).any()

# --------------------------------------------------------------------------------------
# ground
# --------------------------------------------------------------------------------------

def slab_pattern(h, w, sw=(7, 11), sh=3, wear=1):
    out = np.zeros((h, w), np.int16)
    names = ["st0", "st1", "st2"]
    probs = [0.40, 0.42, 0.18]
    for y0 in range(0, h, sh):
        x = -int(rng.integers(0, 9))
        while x < w:
            L = int(rng.integers(sw[0], sw[1] + 1))
            col = C(names[rng.choice(3, p=probs)])
            for yy in range(y0, min(h, y0 + sh)):
                for xx in range(max(0, x), min(w, x + L)):
                    out[yy, xx] = C("stj") if (yy == y0 + sh - 1 or xx == x) else col
            for _ in range(int(rng.integers(0, wear + 1))):
                sx = int(rng.integers(x + 1, x + L))
                sy = int(rng.integers(y0, y0 + sh - 1))
                if 0 <= sx < w and sy < h:
                    out[sy, sx] = C(names[int(rng.integers(0, 3))])
            x += L
    return out


def tile_pattern(h, w, t=4):
    out = np.zeros((h, w), np.int16)
    for y in range(h):
        for x in range(w):
            tx, ty = x // t, y // t
            col = "st0" if (tx + ty) % 2 == 0 else "st1"
            if rng.random() < 0.05:
                col = "st2"
            if x % t == t - 1 or y % t == t - 1:
                col = "stj"
            out[y, x] = C(col)
    return out


horiz = slab_pattern(H, W)
vert = slab_pattern(W, H).T
tiles = tile_pattern(H, W)
ground = horiz.copy()
zone_v = np.zeros((CH, CW), bool)
for (a, b) in VSTREETS:
    zone_v[:, a:b + 1] = True
zone_v[8:12, :] = False
ground[cell_mask_to_px(zone_v)] = vert[cell_mask_to_px(zone_v)]
zone_t = np.zeros((CH, CW), bool)
zone_t[0:8, 28:36] = True
zone_t[39:48, 30:50] = True
zone_t[0:8, 55:63] = True
for (x, y, w, h) in COURTS:
    zone_t[y:y + h, x:x + w] = True
gt = cell_mask_to_px(zone_t)
ground[gt] = tiles[gt]
cv.px[:, :] = ground
# large-scale tonal variation (worn / fresh paving)
n3 = value_noise(rng, H, W, 14)
dark = n3 < 0.36
light = n3 > 0.70
cv.remap(dark, {"st1": "st2", "st0": "st1"})
cv.remap(light, {"st1": "st0", "st2": "st1"})

# inlaid centre stripe along the avenue (dashed mid-tone) and street centre lines
for x in range(0, W):
    if (x // 5) % 2 == 0:
        cv.set(x, 19, "st2")
for (a, b) in VSTREETS:
    cx = a + b + 1   # centre in px = (2a + 2(b+1)) / 2
    for y in range(0, H):
        if (y // 5) % 2 == 0 and not (16 <= y < 24):
            cv.set(cx, y, "st2")
# cracks / stains
for _ in range(50):
    x = int(rng.integers(0, W - 4))
    y = int(rng.integers(0, H - 4))
    if cv.px[y, x] in ST_IDS[:3]:
        for k in range(int(rng.integers(2, 4))):
            cv.set(x + k, y + (k // 2), "stj" if rng.random() < 0.6 else "st2")

# --------------------------------------------------------------------------------------
# water
# --------------------------------------------------------------------------------------
wpx = cell_mask_to_px(water_cells)
n1 = value_noise(rng, H, W, 8)
n2 = value_noise(rng, H, W, 3)
land_d = dist_to_mask(~wpx, 6)
for y in range(H):
    for x in range(W):
        if not wpx[y, x]:
            continue
        v = n1[y, x] * 0.65 + n2[y, x] * 0.35
        d = land_d[y, x]
        shallow = d <= 3
        if shallow:
            col = "w2" if v > 0.42 else "w1"
        else:
            col = "w1" if v > 0.45 else "w0"
            if v > 0.78:
                col = "w2"
        cv.px[y, x] = C(col)
# ripples: short dashes
for _ in range(420):
    x = int(rng.integers(0, W - 8))
    y = int(rng.integers(0, H))
    L = int(rng.integers(3, 7))
    if wpx[y, x:x + L].all() and land_d[y, x:x + L].min() >= 2:
        for k in range(L):
            cv.set(x + k, y, "w3" if (k in (L // 2, L // 2 - 1) and rng.random() < 0.7) else "w2")
for _ in range(80):
    x = int(rng.integers(0, W))
    y = int(rng.integers(0, H))
    if wpx[y, x] and land_d[y, x] >= 2:
        cv.set(x, y, "w3")
lp = ~wpx
for y in range(H):
    for x in range(W):
        if wpx[y, x] and land_d[y, x] <= 1:
            cv.set(x, y, "w4" if (y > 0 and lp[y - 1, x]) else "w0")
d2 = dist_to_mask(wpx, 3)
for y in range(H):
    for x in range(W):
        if not wpx[y, x] and cv.px[y, x] in ST_IDS[:3] + [JOINT]:
            if d2[y, x] == 1:
                cv.px[y, x] = JOINT
            elif d2[y, x] == 2:
                cv.px[y, x] = C("st0")
            elif d2[y, x] == 3 and (x + y) % 3 == 0:
                cv.px[y, x] = C("st2")

# --------------------------------------------------------------------------------------
# bridges
# --------------------------------------------------------------------------------------

def draw_bridge(o, x0, x1, y0, y1):
    px0, px1 = x0 * 2, (x1 + 1) * 2
    py0, py1 = y0 * 2, (y1 + 1) * 2
    if o == "v":
        for sx in (px0 - 2, px0 - 1, px1, px1 + 1):
            for y in range(py0, py1):
                if cv.inb(sx, y) and wpx[y, sx]:
                    cv.set(sx, y, "w4")
    else:
        for sy in (py0 - 2, py0 - 1, py1, py1 + 1):
            for x in range(px0, px1):
                if cv.inb(x, sy) and wpx[sy, x]:
                    cv.set(x, sy, "w4")
    L = (py1 - py0) if o == "v" else (px1 - px0)
    for a in range(L):
        # arch: lighter in the middle of the span, darker at the ramps
        t = abs(a - (L - 1) / 2) / (L / 2)
        base = "st0" if t < 0.55 else "st1"
        for bpos in range(2 * ((x1 - x0 + 1) if o == "v" else (y1 - y0 + 1))):
            x, y = (px0 + bpos, py0 + a) if o == "v" else (px0 + a, py0 + bpos)
            col = "stj" if a % 3 == 2 else base
            cv.set(x, y, col)
    # parapets
    n = 2 * ((x1 - x0 + 1) if o == "v" else (y1 - y0 + 1))
    for a in range(L):
        for side in (0, 1):
            for k, col in enumerate(("stj", "st0", "st2")):
                bp = k if side == 0 else n - 1 - k
                x, y = (px0 + bp, py0 + a) if o == "v" else (px0 + a, py0 + bp)
                cv.set(x, y, col)
    # corner posts (stone caps; lanterns are added later)
    for (cx_, cy_) in ((px0, py0), (px1 - 3, py0), (px0, py1 - 3), (px1 - 3, py1 - 3)):
        cv.rect(cx_, cy_, cx_ + 3, cy_ + 3, "st0")
        cv.rect(cx_, cy_ + 2, cx_ + 3, cy_ + 3, "st2")


for br in bridge_rects:
    draw_bridge(*br)

# --------------------------------------------------------------------------------------
# lantern list
# --------------------------------------------------------------------------------------
lanterns = []
strings = []
for x in range(2, 28, 2):
    lanterns.append((x, 7))
strings.append((2, 27, 7))
for x in range(37, 54, 2):
    lanterns.append((x, 7))
strings.append((37, 53, 7))
for x in range(2, 28, 2):
    lanterns.append((x, 12))
strings.append((2, 27, 12))
for x in range(36, 43, 2):
    lanterns.append((x, 12))
strings.append((36, 42, 12))
for x in range(56, 63, 2):
    lanterns.append((x, 12))
strings.append((56, 62, 12))
for x in range(3, 10, 3):
    lanterns.append((x, 43))
strings.append((3, 9, 43))
for x in range(19, 28, 3):
    lanterns.append((x, 43))
strings.append((19, 27, 43))
for o, x0, x1, y0, y1 in bridge_rects:
    for (cx_, cy_) in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        lanterns.append((cx_, cy_))
# canal quay posts
for x, side in [(3, 0), (8, 1), (18, 0), (24, 1), (36, 0), (41, 1), (49, 0), (58, 1), (61, 0), (22, 0)]:
    col = np.nonzero(water_cells[:, x])[0]
    if len(col):
        y = (col.min() - 1) if side == 0 else (col.max() + 1)
        lanterns.append((x, y))
lanterns += [(29, 7), (34, 7), (54, 8), (55, 8)]
seen = set()
good = []
for (x, y) in lanterns:
    if not (0 <= x < CW and 0 <= y < CH) or (x, y) in seen:
        continue
    if bmask[y, x] or (wmask[y, x] and not bridge_cells[y, x]):
        continue
    if any(a <= x <= b for a, b in [(55, 62)]) and y <= 7:
        continue
    seen.add((x, y))
    good.append((x, y))
lanterns = good + [(56, 1), (61, 1), (56, 6), (61, 6)]

# --------------------------------------------------------------------------------------
# glow on the stone around lanterns (dithered pale yellow, low saturation)
# --------------------------------------------------------------------------------------
glow_sources = [(2 * x + 1, 2 * y + 1, 6) for (x, y) in lanterns]
glow_sources += [(63, 8, 14), (80, 87, 12)]
for (gx, gy, R) in glow_sources:
    for y in range(int(gy - R), int(gy + R) + 1):
        for x in range(int(gx - R), int(gx + R) + 1):
            if not cv.inb(x, y):
                continue
            d = math.hypot(x - gx, y - gy)
            if d > R:
                continue
            p = (1 - d / R) * 0.95
            if cv.px[y, x] in ST_IDS[:3] and p > bayer(x, y):
                cv.px[y, x] = C("glow2" if d < R * 0.35 else "glow1")

# --------------------------------------------------------------------------------------
# shadows (NW light -> SE shadow)
# --------------------------------------------------------------------------------------
gate_cells = np.zeros((CH, CW), bool)
for (x, y, w, h) in GATE_ROOFS:
    gate_cells[y:y + h, x:x + w] = True
bpx = cell_mask_to_px(bmask | gate_cells)
sh = np.zeros_like(bpx)
sh[2:, 2:] = bpx[:-2, :-2]
sh[1:, 1:] |= bpx[:-1, :-1]
sh[2:, 1:] |= bpx[:-2, :-1]
shadow = sh & ~bpx


def shade_px(x, y):
    v = cv.px[y, x]
    if v in ST_IDS:
        cv.px[y, x] = C("sts")
    elif v == JOINT:
        cv.px[y, x] = C("stsj")
    elif v in (C("w1"), C("w2")):
        cv.px[y, x] = C("w0")
    elif v in (C("w0"), C("w3")):
        cv.px[y, x] = C("w4")


for y in range(H):
    for x in range(W):
        if shadow[y, x]:
            shade_px(x, y)

# --------------------------------------------------------------------------------------
# roofs
# --------------------------------------------------------------------------------------

def roof_pattern(w, h, pal):
    """Hip roof with horizontal ridge (w >= h). pal = (dark, base, light, accent).
    Tile ribs run down the slope; ridge = light row + dark row sharing one terrain cell row."""
    d0, d1, d2, d3 = [C(n) for n in pal]
    out = np.zeros((h, w), np.int16)
    nc = h // 2
    jr = 2 * ((nc - 1) // 2)             # ridge light row (even), dark row = jr + 1
    m = max(1, h // 2 - 1)
    for j in range(h):
        for i in range(w):
            dl, dr, dt, db = i, w - 1 - i, j, h - 1 - j
            dm = min(dl, dr, dt, db)
            if dm == 0:
                corner = (min(dl, dr) == 0 and min(dt, db) == 0)
                lit = (dt == 0 or dl == 0)
                out[j, i] = d2 if (corner or (lit and (i + j) % 2 == 0)) else d1
                continue
            long_d, end_d = min(dt, db), min(dl, dr)
            if m <= i <= w - 1 - m and j == jr:
                out[j, i] = d2
                continue
            if m <= i <= w - 1 - m and j == jr + 1:
                out[j, i] = d0
                continue
            if long_d == end_d:
                out[j, i] = d1                      # hip line
                continue
            if long_d < end_d:                      # long facet: ribs down the slope
                par, k = i % 2, j
                lit = dt < db                       # NW light: north facet lit, south facet in shade
            else:                                   # end facet
                par, k = j % 2, i
                lit = dl < dr
            if par == 1:
                out[j, i] = d0
            elif lit:
                out[j, i] = d1 if (k % 3 == 0 or (i * 7 + j * 13) % 11 == 0) else d2
            else:
                out[j, i] = d0 if (i * 7 + j * 13) % 9 == 0 else d1
    return out


def draw_roof(x0, y0, x1, y1, pal=("rf0", "rf1", "rf2", "rf3")):
    w, h = x1 - x0, y1 - y0
    pat = roof_pattern(w, h, pal) if w >= h else roof_pattern(h, w, pal).T
    cv.px[y0:y1, x0:x1] = pat


for grp in BUILDINGS:
    for k, (x, y, w, h) in enumerate(grp):
        # tiny drop shadow of annexes on the main roof
        draw_roof(x * 2, y * 2, (x + w) * 2, (y + h) * 2)
px, py, pw, ph = PAV_CORE
draw_roof(px * 2, py * 2, (px + pw) * 2, (py + ph) * 2)
cv.set(px * 2 + pw - 1, py * 2 + ph - 1, "r3")
cv.set(px * 2 + pw, py * 2 + ph, "r3")
cv.set(px * 2 + pw - 1, py * 2 + ph, "r1")
cv.set(px * 2 + pw, py * 2 + ph - 1, "r1")

# --------------------------------------------------------------------------------------
# vermilion things
# --------------------------------------------------------------------------------------

def red_block(cx0, cy0, cw, ch, grain=True):
    x0, y0, x1, y1 = cx0 * 2, cy0 * 2, (cx0 + cw) * 2, (cy0 + ch) * 2
    cv.rect(x0, y0, x1, y1, "r1")
    cv.hline(x0, x1, y0, "r2")
    cv.vline(x0, y0, y1, "r2")
    cv.hline(x0, x1, y1 - 1, "r0")
    cv.vline(x1 - 1, y0, y1, "r0")
    cv.set(x0, y0, "r3")
    if grain:
        for xx in range(x0 + 2, x1 - 1, 3):
            for yy in range(y0 + 1, y1 - 1):
                if rng.random() < 0.5:
                    cv.set(xx, yy, "r2")


def lantern(cx, cy):
    x, y = cx * 2, cy * 2
    cv.set(x, y, "r3")
    cv.set(x + 1, y, "r2")
    cv.set(x, y + 1, "r1")
    cv.set(x + 1, y + 1, "r0")


# red carpet along the axis (gate -> avenue -> bridge approach)
for cy_ in range(6, 16):
    for cx_ in (31, 32):
        if not (wmask[cy_, cx_] or bmask[cy_, cx_]):
            X, Y = cx_ * 2, cy_ * 2
            cv.rect(X, Y, X + 2, Y + 2, "r1")
            cv.set(X, Y + (cy_ % 2), "r2")
            cv.set(X + 1, Y + 1 - (cy_ % 2), "r0")
    # gold edging
    cv.set(62, cy_ * 2, "gold") if cy_ % 2 == 0 else None
    cv.set(65, cy_ * 2 + 1, "gold") if cy_ % 2 == 1 else None

# gate: glazed red roofs over the axis street (+ pillars at the front corners)
for (x, y, w, h) in GATE_ROOFS:
    draw_roof(x * 2, y * 2, (x + w) * 2, (y + h) * 2, pal=("r0", "r1", "r2", "r3"))
for (x, y) in ((28, 6), (34, 6)):
    red_block(x, y, 2, 1, grain=False)
    cv.set(x * 2 + 1, y * 2 + 1, "gold")
# gold plaque and ridge ornaments
for xx in range(62, 66):
    cv.set(xx, 8, "gold")
cv.set(61, 8, "gold"); cv.set(66, 8, "gold")
cv.set(63, 7, "gold"); cv.set(64, 7, "gold")
cv.set(57, 8, "gold"); cv.set(70, 8, "gold")
# pavilion railing
x0, y0, pw_, ph_ = PAV
for cx_ in range(x0, x0 + pw_):
    for cy_ in range(y0, y0 + ph_):
        edge = cx_ in (x0, x0 + pw_ - 1) or cy_ in (y0, y0 + ph_ - 1)
        gap = cy_ == y0 + ph_ - 1 and cx_ in (x0 + 3, x0 + 4)
        if not edge or gap:
            continue
        corner = cx_ in (x0, x0 + pw_ - 1) and cy_ in (y0, y0 + ph_ - 1)
        X, Y = cx_ * 2, cy_ * 2
        if corner:
            red_block(cx_, cy_, 1, 1, grain=False)
            cv.set(X + 1, Y + 1, "gold")
        else:
            cv.rect(X, Y, X + 2, Y + 2, "r1")
            if cy_ in (y0, y0 + ph_ - 1):
                cv.hline(X, X + 2, Y, "r2")
                cv.hline(X, X + 2, Y + 1, "r0")
            else:
                cv.vline(X, Y, Y + 2, "r2")
                cv.vline(X + 1, Y, Y + 2, "r0")
            if (cx_ + cy_) % 2 == 0:
                cv.set(X, Y, "r3")

# striped shop awnings (red / cream, scalloped hem) in front of shops
AWNINGS = [(20, 34, 4), (58, 32, 4), (58, 19, 4), (19, 17, 4), (2, 18, 3), (38, 24, 4)]
awning_cells = np.zeros((CH, CW), bool)
for (ax, ay, aw) in AWNINGS:
    if wmask[ay:ay + 2, ax:ax + aw].any() or bmask[ay:ay + 2, ax:ax + aw].any():
        print("awning skipped", ax, ay)
        continue
    awning_cells[ay:ay + 2, ax:ax + aw] = True
    for i in range(aw * 2):
        for j in range(4):
            red = (i % 2 == 0)
            if j < 3:
                col = "r1" if red else "glow1"
                if j == 0 and red:
                    col = "r2"
            else:
                col = ("r0" if red else "wd0") if (i // 2) % 2 == 0 else ("r0" if red else "stj")
            cv.set(ax * 2 + i, ay * 2 + j, col)
    for i in range(aw * 2):
        for j in (4, 5):
            if cv.px[ay * 2 + j, ax * 2 + i] in ST_IDS:
                cv.px[ay * 2 + j, ax * 2 + i] = C("sts")
            elif cv.px[ay * 2 + j, ax * 2 + i] == JOINT:
                cv.px[ay * 2 + j, ax * 2 + i] = C("stsj")

# lantern strings (string line, then lanterns)
for (xa, xb, cy) in strings:
    for xpx in range(xa * 2, xb * 2 + 2):
        yy = cy * 2 - 1
        if cv.px[yy, xpx] in ST_IDS + [JOINT, C("sts"), C("stsj")]:
            cv.set(xpx, yy + (1 if (xpx // 2 - xa) % 2 == 1 else 0) - 0, "wd0")
for (x, y) in lanterns:
    lantern(x, y)

# --------------------------------------------------------------------------------------
# trees, pots, boats, lotus
# --------------------------------------------------------------------------------------

def tree(cx, cy, r):
    for y in range(int(cy - r) + 1, int(cy + r) + 5):
        for x in range(int(cx - r) + 1, int(cx + r) + 5):
            if cv.inb(x, y) and (x - cx - 2) ** 2 + (y - cy - 2) ** 2 <= (r + 0.5) ** 2:
                shade_px(x, y)
    for y in range(int(cy - r) - 1, int(cy + r) + 2):
        for x in range(int(cx - r) - 1, int(cx + r) + 2):
            dx, dy = x - cx, y - cy
            rr = r * (1 + 0.14 * math.sin(math.atan2(dy, dx) * 5 + cx))
            d = math.hypot(dx, dy)
            if d <= rr and cv.inb(x, y):
                lit = (-dx - dy) / (rr + 1e-6)
                v = rng.random()
                if d > rr - 1.2:
                    col = "g0"
                elif lit > 0.35 and v < 0.6:
                    col = "g2"
                elif lit > -0.1 and v < 0.7:
                    col = "g1"
                elif v < 0.35:
                    col = "g1"
                else:
                    col = "g0"
                cv.set(x, y, col)


def pot(cx, cy):
    x, y = cx * 2, cy * 2
    cv.set(x, y, "g1"); cv.set(x + 1, y, "g2")
    cv.set(x, y + 1, "pot"); cv.set(x + 1, y + 1, "wd0")


TREES = [(11, 9, 3), (43, 9, 3), (11, 75, 3)]            # courtyard trees (fixed)
free_tree = ~(bmask | dil | bridge_cells)
block = np.zeros((CH, CW), bool)                         # keep streets / lantern rows / landmarks clear
for (a_, b_) in VSTREETS:
    block[:, a_ - 1:b_ + 2] = True
block[7:13, :] = False
block[6:13, :] = True
block[0:16, 26:37] = True
block[0:9, 54:64] = True
block[41:48, 0:12] = True
for (x, y) in seen:
    block[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
awn_cells = np.zeros((CH, CW), bool)
for (ax, ay, aw) in AWNINGS:
    awn_cells[max(0, ay - 1):ay + 4, max(0, ax - 1):ax + aw + 1] = True
block |= awn_cells
placed_trees = []
for (tx, ty, tr) in TREES:
    placed_trees.append((tx, ty, tr))
cand = [(int(rng.integers(6, W - 6)), int(rng.integers(6, H - 6)), int(rng.integers(3, 6))) for _ in range(900)]
for (tx, ty, tr) in cand:
    if len(placed_trees) >= 3 + 11:
        break
    cx0, cx1 = int((tx - tr) // 2), int((tx + tr) // 2)
    cy0, cy1 = int((ty - tr) // 2), int((ty + tr) // 2)
    if cx0 < 1 or cy0 < 1 or cx1 >= CW - 1 or cy1 >= CH - 1:
        continue
    if not free_tree[cy0:cy1 + 1, cx0:cx1 + 1].all() or block[cy0:cy1 + 1, cx0:cx1 + 1].any():
        continue
    if any(math.hypot(tx - a_, ty - b_) < 15 for (a_, b_, _) in placed_trees):
        continue
    placed_trees.append((tx, ty, tr))
for (tx, ty, tr) in placed_trees:
    tree(tx, ty, tr)

occupied = np.zeros((CH, CW), bool)
for (tx, ty, tr) in placed_trees:
    occupied[int((ty - tr) // 2):int((ty + tr) // 2) + 1, int((tx - tr) // 2):int((tx + tr) // 2) + 1] = True
free = ~(bmask | wmask | bridge_cells | occupied)
pots = []
for y in range(1, CH - 1):
    for x in range(1, CW - 1):
        if not free[y, x] or (x, y) in seen:
            continue
        if any(a - 1 <= x <= b + 1 for a, b in VSTREETS):
            continue
        if 27 <= x <= 36 and y <= 15:
            continue
        if (y, x) and 55 <= x <= 62 and y <= 8:
            continue
        near_b = bmask[y - 1, x] or bmask[y + 1, x]
        if near_b and (x * 3 + y * 7) % 6 == 0 and y not in (7, 12, 43):
            if not any(abs(x - a) + abs(y - b) < 5 for a, b in pots):
                pots.append((x, y))
for (x, y) in pots:
    pot(x, y)

# ---- street furniture: stone lamps, pond posts, benches, barrels, puddles ----
used = set(pots) | set(lanterns)
def cell_free(x, y):
    return (0 <= x < CW and 0 <= y < CH and not bmask[y, x] and not wmask[y, x] and not bridge_cells[y, x]
            and not occupied[y, x] and not awning_cells[y, x] and (x, y) not in used)


def stone_lamp(cx, cy):
    x, y = cx * 2, cy * 2
    cv.set(x, y, "glow2"); cv.set(x + 1, y, "glow1")
    cv.set(x, y + 1, "st0"); cv.set(x + 1, y + 1, "stj")
    used.add((cx, cy))


def post(cx, cy):
    x, y = cx * 2, cy * 2
    cv.rect(x, y, x + 2, y + 2, "st0")
    cv.set(x + 1, y + 1, "stj"); cv.set(x, y + 1, "st2")
    used.add((cx, cy))


def bench(cx, cy):
    x, y = cx * 2, cy * 2
    cv.rect(x, y, x + 4, y + 2, "wd1")
    cv.hline(x, x + 4, y + 1, "wd0")
    cv.set(x, y, "wd0"); cv.set(x + 3, y, "wd0")
    used.add((cx, cy)); used.add((cx + 1, cy))


def barrel(cx, cy):
    x, y = cx * 2, cy * 2
    cv.set(x, y, "wd1"); cv.set(x + 1, y, "wd0")
    cv.set(x, y + 1, "wd0"); cv.set(x + 1, y + 1, "wd1")
    used.add((cx, cy))


for (x, y) in [(29, 13), (34, 13), (15, 37), (28, 38), (50, 35), (29, 34), (12, 21), (53, 10)]:
    if cell_free(x, y):
        stone_lamp(x, y)
for (x, y) in [(33, 40), (46, 40), (33, 46), (46, 46)]:
    if cell_free(x, y):
        post(x, y)
for (x, y) in [(37, 39), (42, 39), (36, 46), (41, 46)]:
    if cell_free(x, y) and cell_free(x + 1, y):
        bench(x, y)
for (x, y) in [(17, 37), (26, 38), (55, 36), (36, 31), (45, 30), (10, 17), (16, 15), (27, 15), (35, 15), (43, 14), (55, 15)]:
    if cell_free(x, y):
        barrel(x, y)
def stall(cx, cy, wc=3, alt=False):
    """Market stall: striped canopy (2 cells deep) + wooden counter in front."""
    cells = [(x, y) for x in range(cx, cx + wc) for y in range(cy, cy + 3)]
    if not all(cell_free(x, y) for (x, y) in cells):
        return False
    for i in range(wc * 2):
        for j in range(4):
            red = (i % 2 == 0)
            col = ("r1" if red else "glow1") if j < 3 else ("r0" if red else "stj")
            if alt and j < 3:
                col = ("r2" if red else "st0")
            cv.set(cx * 2 + i, cy * 2 + j, col)
        cv.set(cx * 2 + i, cy * 2 + 4, "wd1")
        cv.set(cx * 2 + i, cy * 2 + 5, "wd0")
        for k in (6,):
            if cv.px[cy * 2 + k, cx * 2 + i] in ST_IDS:
                cv.px[cy * 2 + k, cx * 2 + i] = C("sts")
    for (x, y) in cells:
        used.add((x, y))
    return True


for (x, y, w_, alt) in [(27, 30, 3, False), (34, 37, 3, True), (47, 38, 3, False), (58, 41, 3, True), (36, 20, 3, False)]:
    stall(x, y, w_, alt)

# decorative inlay ring around the south pond and a medallion at the avenue crossing
for y in range(H):
    for x in range(W):
        for (rx, ry) in ((15.8, 9.6), (20.0, 13.4)):
            v = ((x + 0.5 - 80) / rx) ** 2 + ((y + 0.5 - 86.4) / ry) ** 2
            if 0.93 < v < 1.07 and cv.px[y, x] in ST_IDS[:3] + [JOINT]:
                cv.px[y, x] = C("st2") if (x + y) % 3 else C("stj")
# damp patches
for _ in range(14):
    x = int(rng.integers(4, W - 8)); y = int(rng.integers(4, H - 4))
    for dx in range(int(rng.integers(3, 6))):
        for dy in range(2):
            if cv.inb(x + dx, y + dy) and cv.px[y + dy, x + dx] in ST_IDS[:3]:
                cv.px[y + dy, x + dx] = C("sts")
# moss along the quays and at the foot of buildings
shade_zone = shadow
for y in range(H):
    for x in range(W):
        v = cv.px[y, x]
        if v in ST_IDS[:3] + [C("sts")]:
            if (d2[y, x] in (2, 3) and rng.random() < 0.13) or (shade_zone[y, x] and rng.random() < 0.10):
                cv.px[y, x] = C("moss")


def boat(cx, cy, length=6):
    """Small wooden boat (art-px 12 x 5) with a paper lantern; prow to the right."""
    x0, y0 = cx * 2, cy * 2 + 1
    L = length * 2
    for i in range(L):
        t = (i - (L - 1) / 2) / (L / 2)
        hh = 2.6 * math.sqrt(max(0.0, 1 - t ** 4)) if t < 0.8 else 2.6 * (1 - t) / 0.2 * 0.8 + 0.4
        for j in range(5):
            if abs(j - 2) <= hh:
                edge = abs(abs(j - 2) - hh) < 1.0 or j in (0, 4)
                col = "wd0" if edge else ("wd1" if i % 3 else "wd0")
                cv.set(x0 + i, y0 + j, col)
    cv.set(x0 + L // 2 - 1, y0 + 2, "glow2")
    cv.set(x0 + L // 2, y0 + 2, "glow1")
    for k in range(3):
        cv.set(x0 + L + k, y0 + 2, "w3" if k == 1 else "w2")
        cv.set(x0 - 1 - k, y0 + 1 + (k % 2), "w3" if k == 0 else "w2")


boat_cells = np.zeros((CH, CW), bool)
for (bx, by) in [(19, 26), (47, 22), (4, 27)]:
    if wmask[by:by + 3, bx:bx + 6].all():
        boat(bx, by)
        boat_cells[by:by + 3, bx:bx + 6] = True
lotus = np.zeros((CH, CW), bool)
for (lx, ly) in [(37, 43), (40, 44), (42, 42), (44, 44), (45, 4), (47, 5), (44, 3), (46, 2)]:
    if wmask[ly, lx]:
        cv.rect(lx * 2, ly * 2, lx * 2 + 2, ly * 2 + 2, "g1")
        cv.set(lx * 2, ly * 2, "g2")
        cv.set(lx * 2 + 1, ly * 2 + 1, "g0")
        lotus[ly, lx] = True

# --------------------------------------------------------------------------------------
# post-fixes and output
# --------------------------------------------------------------------------------------
wall_cells = bmask.copy()
n_w = cv.fix_walls(wall_cells, ["rf0"])
n_wa = cv.fix_water(water_cells & ~boat_cells & ~lotus, "w1")
print("fixed wall cells:", n_w, " water cells:", n_wa, " unique colours:", cv.unique_colors())
out = os.path.join(HERE, "shu_street.png")
cv.save(out)
print("saved", out, "trees:", len(placed_trees), "pots:", len(pots), "lanterns:", len(lanterns))
