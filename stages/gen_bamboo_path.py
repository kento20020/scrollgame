#!/usr/bin/env python3
"""Generate stages/bamboo_path.png  (竹林の小径) -- top-down bamboo grove, procedural pixel art.

Drawn on the 128x96 art grid, saved at 256x192 (2x nearest).  Terrain cells are 2x2 art px.
Light from the north-west; shadows fall to the south-east.
"""
import math
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pxlib import Canvas, bayer, cell_mask_to_px, dist_to_mask, value_noise  # noqa: E402

SEED = 23
rng = np.random.default_rng(SEED)
CW, CH = 64, 48
W, H = 128, 96

PAL = {
    # forest floor
    "m0": (50, 70, 46), "m1": (66, 90, 56), "m2": (88, 112, 66),
    # path earth
    "e0": (116, 108, 82), "e1": (140, 132, 98), "e2": (164, 156, 118),
    # stone
    "t0": (150, 152, 142), "t1": (180, 180, 168), "t2": (112, 116, 110), "wh": (234, 232, 218),
    # bamboo canopy / mossy roof (walls)
    "b0": (6, 22, 18), "b1": (12, 40, 28), "b2": (22, 66, 40), "b3": (40, 96, 54),
    # stream
    "w0": (30, 92, 118), "w1": (46, 122, 148), "w2": (82, 164, 180), "w3": (160, 212, 218), "w4": (22, 72, 96),
    # vermilion / maple
    "r0": (118, 28, 30), "r1": (190, 48, 38), "r2": (226, 92, 50), "r3": (236, 136, 56),
    # wood, leaf litter
    "wd0": (84, 66, 52), "wd1": (124, 100, 76), "lf": (150, 162, 72),
    "k1": (40, 28, 24), "k2": (74, 52, 40),
}
cv = Canvas(PAL, W, H)
C = cv.c
GROUND_IDS = [C(n) for n in ("m0", "m1", "m2", "e0", "e1", "e2", "lf")]


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------
def catmull(points, n=24):
    P = [points[0]] + list(points) + [points[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = (np.array(P[k], float) for k in (i - 1, i, i + 1, i + 2))
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(np.array(points[-1], float))
    return np.array(out)


def dist_field(gx, gy, Q):
    """gx, gy: arrays (same shape); Q: (M,2). Returns min distance and argmin index."""
    d = np.hypot(gx[..., None] - Q[:, 0], gy[..., None] - Q[:, 1])
    return d.min(-1), d.argmin(-1)


CY_, CX_ = np.mgrid[0:CH, 0:CW]
cell_x, cell_y = CX_ + 0.5, CY_ + 0.5
PY_, PX_ = np.mgrid[0:H, 0:W]

# --------------------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------------------
STREAM = catmull([(-5, 12), (6, 10), (16, 12), (24, 17), (30, 24), (33, 31), (40, 37), (52, 39), (69, 36)])
tang = np.gradient(STREAM, axis=0)
hw = 3.2 + 0.55 * np.sin(np.arange(len(STREAM)) * 0.17) + 0.4 * np.sin(np.arange(len(STREAM)) * 0.071 + 1)
d_cells = np.hypot(cell_x[..., None] - STREAM[:, 0], cell_y[..., None] - STREAM[:, 1]) - hw
wmask = (d_cells.min(-1) <= 0)
# pool on the west side of the stream
wmask |= ((cell_x - 27.0) / 5.2) ** 2 + ((cell_y - 21.0) / 4.0) ** 2 <= 1.0

PATH_A = catmull([(13, 49), (15, 42), (21, 36), (27, 32), (33, 29), (40, 25), (46, 20), (49, 15), (50, 8)])
PATH_B = catmull([(-2, 25), (7, 26), (13, 29), (18, 34), (21, 36)])
PATH_C = catmull([(40, 25), (45, 31), (47, 36), (48, 43), (48, 49)])
PATH_D = catmull([(48, 15), (43, 11), (37, 8), (31, 5)])
PATH_E = catmull([(48, 44), (54, 43), (60, 44), (66, 43)])
PATHS = [(PATH_A, 1.9), (PATH_B, 1.6), (PATH_C, 1.7), (PATH_D, 1.4), (PATH_E, 1.5)]


def path_dist_cells():
    best = np.full((CH, CW), 99.0)
    for P, w in PATHS:
        d, _ = dist_field(cell_x, cell_y, P)
        best = np.minimum(best, d - w)
    return best


pdist = path_dist_cells()
path_corr = pdist <= 0.9                      # keep bamboo away from the trails

SHRINE_ZONE = np.zeros((CH, CW), bool)
SHRINE_ZONE[0:16, 40:60] = True
clear_spots = [(21, 36, 3.6), (40, 25, 3.2), (48, 44, 3.8), (33, 29, 3.6), (8, 39, 4.0), (37, 8, 3.0), (52, 44, 4.2),
               (18, 22, 3.0), (56, 25, 3.5), (6, 15, 3.0)]
clearing = np.zeros((CH, CW), bool)
for (cx_, cy_, r_) in clear_spots:
    clearing |= (cell_x - cx_) ** 2 + (cell_y - cy_) ** 2 <= r_ ** 2

# stepping stones (2x2 cells each) where trail A crosses the stream
stone_cells = np.zeros((CH, CW), bool)
stones_list = []
for p in PATH_A:
    cx0, cy0 = int(p[0]) // 2 * 2, int(p[1]) // 2 * 2
    if not (0 <= cx0 < CW - 1 and 0 <= cy0 < CH - 1):
        continue
    if wmask[cy0:cy0 + 2, cx0:cx0 + 2].sum() >= 3 and not stone_cells[max(0, cy0 - 2):cy0 + 4, max(0, cx0 - 2):cx0 + 4].any():
        stones_list.append((cx0, cy0))
        stone_cells[cy0:cy0 + 2, cx0:cx0 + 2] = True
wmask_stream = wmask.copy()
wmask &= ~stone_cells

# wooden bridge across the lower stream on path C
BR_X0, BR_X1 = 46, 48
ys = [y for y in range(CH) if wmask[y, BR_X0:BR_X1 + 1].any()]
BR_Y0, BR_Y1 = min(ys) - 1, max(ys) + 1
bridge_cells = np.zeros((CH, CW), bool)
bridge_cells[BR_Y0:BR_Y1 + 1, BR_X0:BR_X1 + 1] = True
wmask &= ~bridge_cells

# distance from water (cells)
wd = np.full((CH, CW), 99.0)
wy, wx = np.nonzero(wmask | stone_cells | bridge_cells)
if len(wy):
    wd = np.hypot(cell_x[..., None] - (wx + 0.5), cell_y[..., None] - (wy + 0.5)).min(-1)
water_margin = wd <= 1.6

MAPLES = [(76, 40, 9, 8), (22, 72, 8, 7)]       # art px: cx, cy, rx, ry
maple_zone = np.zeros((CH, CW), bool)
for (mx, my, rx, ry) in MAPLES:
    maple_zone |= ((cell_x - mx / 2) / (rx / 2 + 1.4)) ** 2 + ((cell_y - my / 2) / (ry / 2 + 1.4)) ** 2 <= 1.0
LEAF_PILES = [(44, 36, 6, 4), (30, 76, 6, 4), (96, 54, 6, 4), (14, 56, 5, 3), (74, 20, 5, 3)]
for (lx, ly, rx, ry) in LEAF_PILES:
    maple_zone |= ((cell_x - lx / 2) / (rx / 2 + 1.0)) ** 2 + ((cell_y - ly / 2) / (ry / 2 + 1.0)) ** 2 <= 1.0
forbid = maple_zone | path_corr | water_margin | SHRINE_ZONE | clearing | wmask | stone_cells | bridge_cells


def has_pockets(wall):
    # use a guaranteed open cell: a path-A cell
    py_, px_ = np.unravel_index(np.argmin(np.where(~wall, pdist, 99)), wall.shape)
    seen = np.zeros_like(wall)
    q = deque([(py_, px_)])
    seen[py_, px_] = True
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < CH and 0 <= nx < CW and not seen[ny, nx] and not wall[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return bool((~wall & ~seen).any())


wall = np.zeros((CH, CW), bool)


def blob(cx, cy, r):
    ang = np.arctan2(cell_y - cy, cell_x - cx)
    ph = rng.random(3) * 6.28
    rr = r * (1 + 0.22 * np.sin(3 * ang + ph[0]) + 0.12 * np.sin(5 * ang + ph[1]))
    return np.hypot(cell_x - cx, cell_y - cy) <= rr


TARGET = 0.215
# border framing first
border_centres = []
for x in range(-1, CW + 2, 5):
    border_centres.append((x + rng.integers(-1, 2), rng.integers(-1, 3), rng.integers(3, 5)))
    border_centres.append((x + rng.integers(-1, 2), CH - rng.integers(-1, 3), rng.integers(3, 5)))
for y in range(3, CH - 2, 5):
    border_centres.append((rng.integers(-1, 3), y + rng.integers(-1, 2), rng.integers(3, 5)))
    border_centres.append((CW - rng.integers(-1, 3), y + rng.integers(-1, 2), rng.integers(3, 5)))
rng.shuffle(border_centres)
for (cx, cy, r) in border_centres:
    if wall.mean() >= TARGET * 0.62:
        break
    b = blob(cx, cy, r)
    if (b & forbid).any():
        continue
    new = wall | b
    if has_pockets(new):
        continue
    wall = new
# interior clumps
tries = 0
while wall.mean() < TARGET and tries < 6000:
    tries += 1
    cx, cy, r = rng.uniform(2, CW - 2), rng.uniform(2, CH - 2), rng.uniform(2.0, 4.0)
    b = blob(cx, cy, r)
    if (b & forbid).any():
        continue
    new = wall | b
    if has_pockets(new):
        continue
    wall = new
bamboo_cells = wall.copy()

# shrine
SHRINE = (48, 3, 5, 4)
shrine_cells = np.zeros((CH, CW), bool)
shrine_cells[SHRINE[1]:SHRINE[1] + SHRINE[3], SHRINE[0]:SHRINE[0] + SHRINE[2]] = True
wall_cells = bamboo_cells | shrine_cells
print("bamboo %.1f%%  water %.1f%%" % (100 * bamboo_cells.mean(), 100 * wmask.mean()))

# --------------------------------------------------------------------------------------
# ground
# --------------------------------------------------------------------------------------
n1 = value_noise(rng, H, W, 10)
n2 = value_noise(rng, H, W, 4)
v = 0.6 * n1 + 0.4 * n2
cv.px[:, :] = C("m1")
cv.px[v < 0.40] = C("m0")
cv.px[v > 0.60] = C("m2")
# grass tufts and leaf litter
for _ in range(520):
    x, y = int(rng.integers(0, W)), int(rng.integers(0, H - 1))
    cv.set(x, y, "m2"); cv.set(x, y + 1, "m0")
for _ in range(70):
    x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
    cv.set(x, y, "lf")
for _ in range(30):
    x, y = int(rng.integers(0, W - 1)), int(rng.integers(0, H))
    cv.set(x, y, "lf"); cv.set(x + 1, y, "lf")

# paths: earth with ragged edges
pxd = np.full((H, W), 99.0)
for P, w in PATHS:
    d, _ = dist_field(PX_ + 0.5, PY_ + 0.5, P * 2)
    pxd = np.minimum(pxd, d - w * 2)
pn = value_noise(rng, H, W, 5) * 2.4 - 1.2
pn2 = value_noise(rng, H, W, 2)
e_edge = pxd + pn
for y in range(H):
    for x in range(W):
        e = e_edge[y, x]
        if e < 0.0:
            col = "e1"
            if e < -3.0 and pn2[y, x] > 0.45:
                col = "e2"
            elif pn2[y, x] < 0.28:
                col = "e0"
            if e > -1.3:
                col = "e0" if bayer(x, y) < 0.5 else "e1"
            cv.px[y, x] = C(col)
        elif e < 1.3 and bayer(x, y) < 0.35:
            cv.px[y, x] = C("e0")
# wheel-worn / pebbles / roots on the trails
for _ in range(260):
    x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
    if e_edge[y, x] < -1.0:
        r_ = rng.random()
        cv.set(x, y, "t0" if r_ < 0.4 else ("wh" if r_ < 0.55 else ("e0" if r_ < 0.9 else "t2")))
        if r_ > 0.9:
            cv.set(x + 1, y, "e0")

# --------------------------------------------------------------------------------------
# shrine precinct ground: raked gravel inside the fence, paved approach
# --------------------------------------------------------------------------------------
FENCE = (46, 2, 9, 8)     # x, y, w, h  (cells), entrance gap at the south side
gx0, gy0, gx1, gy1 = (FENCE[0] + 1) * 2, (FENCE[1] + 1) * 2, (FENCE[0] + FENCE[2] - 1) * 2, (FENCE[1] + FENCE[3] - 1) * 2
for y in range(gy0, gy1):
    for x in range(gx0, gx1):
        col = "t1"
        if (y % 3) == 0:
            col = "t0" if (x + y) % 4 else "wh"
        elif (x * 7 + y * 3) % 13 == 0:
            col = "wh"
        cv.px[y, x] = C(col)
# two small garden rocks
for (rx_, ry_) in ((95, 16), (105, 16)):
    for dy in range(-1, 1):
        for dx in range(-1, 2):
            cv.set(rx_ + dx, ry_ + dy, "t2" if (dx + dy) > 0 else ("t0" if dx + dy < 0 else "t1"))
# paved approach (torii -> shrine steps)
for y in range(20, 34):
    for x in range(97, 105):
        k = (y - 20) // 3
        off = (k % 2) * 3
        edge = ((x - 97 - off) % 6 == 0) or ((y - 20) % 3 == 2)
        col = "t2" if edge else ("t1" if (x + y) % 5 else "t0")
        cv.px[y, x] = C(col)
for y in range(34, 44):                      # dirt path blending into stones
    for x in range(98, 104):
        if (x + y) % 3 == 0:
            cv.px[y, x] = C("t0")

# --------------------------------------------------------------------------------------
# water
# --------------------------------------------------------------------------------------
wpx = cell_mask_to_px(wmask)
n3 = value_noise(rng, H, W, 8)
n4 = value_noise(rng, H, W, 3)
land_d = dist_to_mask(~wpx, 6)
_, nearest = dist_field(PX_ + 0.5, PY_ + 0.5, STREAM * 2)
for y in range(H):
    for x in range(W):
        if not wpx[y, x]:
            continue
        vv = n3[y, x] * 0.65 + n4[y, x] * 0.35
        d = land_d[y, x]
        if d <= 3:
            col = "w2" if vv > 0.5 else "w1"
        else:
            col = "w1" if vv > 0.42 else "w0"
        cv.px[y, x] = C(col)
# flow streaks along the current
for _ in range(700):
    x, y = int(rng.integers(0, W - 7)), int(rng.integers(0, H - 7))
    if not wpx[y, x] or land_d[y, x] < 2:
        continue
    t = tang[nearest[y, x]]
    ang = math.atan2(t[1], t[0])
    dx, dy = round(math.cos(ang)), round(math.sin(ang))
    if abs(t[0]) > 2 * abs(t[1]):
        dx, dy = 1, 0
    elif abs(t[1]) > 2 * abs(t[0]):
        dx, dy = 0, 1
    L = int(rng.integers(3, 7))
    pts = [(x + dx * k, y + dy * k) for k in range(L)]
    if all(0 <= px < W and 0 <= py < H and wpx[py, px] and land_d[py, px] >= 2 for px, py in pts):
        for k, (px, py) in enumerate(pts):
            cv.set(px, py, "w3" if (L // 2 - 1 <= k <= L // 2 and rng.random() < 0.7) else "w2")
lp = ~wpx
for y in range(H):
    for x in range(W):
        if wpx[y, x] and land_d[y, x] <= 1:
            nl = (y > 0 and lp[y - 1, x]) or (x > 0 and lp[y, x - 1])
            cv.set(x, y, "w4" if nl else "w0")
# banks: dark wet edge, pebbles and reeds
d2 = dist_to_mask(wpx, 4)
for y in range(H):
    for x in range(W):
        if wpx[y, x]:
            continue
        v_ = cv.px[y, x]
        if v_ in GROUND_IDS:
            if d2[y, x] == 1:
                cv.px[y, x] = C("m0") if bayer(x, y) < 0.7 else C("e0")
            elif d2[y, x] == 2 and bayer(x, y) < 0.3:
                cv.px[y, x] = C("m0")
# bank rocks
for _ in range(170):
    x, y = int(rng.integers(1, W - 4)), int(rng.integers(1, H - 3))
    if not wpx[y, x] and 1 <= d2[y, x] <= 2:
        w_ = int(rng.integers(2, 4))
        if all(0 <= x + i < W and not wpx[y, x + i] for i in range(w_)) and not wpx[y + 1, x]:
            cv.set(x, y, "t1")
            for i in range(1, w_):
                cv.set(x + i, y, "t0")
            for i in range(w_):
                cv.set(x + i, y + 1, "t2")
# reeds
for _ in range(110):
    x, y = int(rng.integers(1, W - 2)), int(rng.integers(2, H - 3))
    if not wpx[y, x] and d2[y, x] in (2, 3):
        cv.set(x, y, "m2"); cv.set(x, y + 1, "m1"); cv.set(x, y - 1, "lf" if rng.random() < 0.4 else "m2")

# --------------------------------------------------------------------------------------
# stepping stones and the wooden bridge
# --------------------------------------------------------------------------------------
for (cx0, cy0) in stones_list:
    x0, y0 = cx0 * 2, cy0 * 2
    for j in range(4):
        for i in range(4):
            if (i in (0, 3)) and (j in (0, 3)):
                continue
            col = "t1"
            if i == 0 or j == 0:
                col = "t0" if (i + j) % 2 else "t1"
            if i == 3 or j == 3:
                col = "t2"
            cv.set(x0 + i, y0 + j, col)
    cv.set(x0 + 1, y0 + 1, "wh")
    # foam ring on the water around the stone
    for (fx, fy) in ((x0 - 1, y0 + 1), (x0 + 4, y0 + 2), (x0 + 1, y0 - 1), (x0 + 2, y0 + 4)):
        if cv.inb(fx, fy) and wpx[fy, fx]:
            cv.set(fx, fy, "w3")
    for (fx, fy) in ((x0 + 1, y0 + 4), (x0 + 3, y0 + 4), (x0 + 4, y0 + 3), (x0 + 4, y0 + 1)):
        if cv.inb(fx, fy) and wpx[fy, fx]:
            cv.set(fx, fy, "w4")

bx0, bx1, by0, by1 = BR_X0 * 2, (BR_X1 + 1) * 2, BR_Y0 * 2, (BR_Y1 + 1) * 2
for x in (bx0 - 2, bx0 - 1, bx1, bx1 + 1):          # bridge shadow on the water
    for y in range(by0, by1):
        if cv.inb(x, y) and wpx[y, x]:
            cv.set(x, y, "w4")
for y in range(by0, by1):
    for x in range(bx0, bx1):
        k = (y - by0) % 3
        col = "wd0" if k == 2 else ("wd1" if ((y - by0) // 3 + (x - bx0) // 3) % 2 == 0 else "wd0")
        if k != 2 and (x * 5 + y * 3) % 7 == 0:
            col = "e0"
        cv.set(x, y, col)
for y in range(by0, by1):                              # rails
    for (xx, c1, c2) in ((bx0, "wd0", "wd1"), (bx1 - 1, "wd0", "wd0")):
        cv.set(xx, y, c1)
    cv.set(bx0 + 1, y, "wd1")
    cv.set(bx1 - 2, y, "wd0")
for (px_, py_) in ((bx0, by0), (bx1 - 2, by0), (bx0, by1 - 2), (bx1 - 2, by1 - 2)):
    cv.rect(px_, py_, px_ + 2, py_ + 2, "wd1")
    cv.set(px_ + 1, py_ + 1, "wd0")

# --------------------------------------------------------------------------------------
# bamboo canopy (walls)
# --------------------------------------------------------------------------------------
bpx = cell_mask_to_px(bamboo_cells)
# shadow of the canopy on the ground (SE)
sh = np.zeros_like(bpx)
for dx, dy in ((2, 2), (1, 1), (3, 2), (2, 3), (1, 2), (2, 1)):
    sh[dy:, dx:] |= bpx[:H - dy, :W - dx]
shadow = sh & ~bpx
smap = {"m1": "m0", "m2": "m1", "e1": "e0", "e2": "e1", "e0": "e0", "lf": "m0", "t0": "t2", "t1": "t0", "wh": "t1",
        "w1": "w0", "w2": "w1", "w0": "w4", "w3": "w2"}
cv.remap(shadow, smap)

# base canopy texture: large soft clumps of light / dark, lit rim on the NW edge, dark rim on the SE edge
bn = value_noise(rng, H, W, 5)
bn2 = value_noise(rng, H, W, 2)
for y in range(H):
    for x in range(W):
        if bpx[y, x]:
            cv.px[y, x] = C("b1" if (bn[y, x] * 0.7 + bn2[y, x] * 0.3) > 0.58 else "b0")
# culm crowns: radiating narrow leaves (star-shaped, ~1 crown per 7 cells)
cells_idx = np.argwhere(bamboo_cells)
n_culms = int(len(cells_idx) * 0.22)
order = rng.permutation(len(cells_idx))[:n_culms]
for k in order:
    cy_, cx_ = cells_idx[k]
    x = cx_ * 2 + int(rng.integers(0, 2))
    y = cy_ * 2 + int(rng.integers(0, 2))
    n_bl = int(rng.integers(7, 10))
    a0 = rng.random() * 6.28
    for b in range(n_bl):
        ang = a0 + b * 6.28 / n_bl + rng.normal(0, 0.12)
        L = int(rng.integers(5, 8))
        lit = (math.cos(ang) + math.sin(ang)) < 0.3
        for s_ in range(1, L + 1):
            px_ = x + int(round(math.cos(ang) * s_))
            py_ = y + int(round(math.sin(ang) * s_))
            if 0 <= px_ < W and 0 <= py_ < H and bpx[py_, px_]:
                if lit:
                    col = "b3" if s_ == L - 1 and rng.random() < 0.5 else "b2"
                else:
                    col = "b1" if s_ < L else "b0"
                cv.set(px_, py_, col)
    cv.set(x, y, "b3"); cv.set(x + 1, y, "b2") if bpx[min(y, H - 1), min(x + 1, W - 1)] else None
# rim light on the north/west edge, dark rim on the south/east edge
for y in range(1, H - 1):
    for x in range(1, W - 1):
        if bpx[y, x]:
            if (not bpx[y - 1, x] or not bpx[y, x - 1]) and rng.random() < 0.6:
                cv.set(x, y, "b2")
            elif (not bpx[y + 1, x] or not bpx[y, x + 1]):
                cv.set(x, y, "b0")
# stray leaves poking out of the canopy edge (1 px only; fixed up later if a floor cell would go dark)
edge_px = bpx & ~(np.roll(bpx, 1, 0) & np.roll(bpx, -1, 0) & np.roll(bpx, 1, 1) & np.roll(bpx, -1, 1))
for y, x in np.argwhere(edge_px):
    for (ox, oy) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + ox, y + oy
        if 0 <= nx < W and 0 <= ny < H and not bpx[ny, nx] and not wpx[ny, nx] and rng.random() < 0.08:
            cv.set(nx, ny, "b1" if (ox + oy) > 0 else "b2")

# --------------------------------------------------------------------------------------
# shrine, fence, torii, lanterns
# --------------------------------------------------------------------------------------

def roof_pattern(w, h, pal):
    d0, d1, d2, d3 = [C(n) for n in pal]
    out = np.zeros((h, w), np.int16)
    nc = h // 2
    jr = 2 * ((nc - 1) // 2)
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
                out[j, i] = d1
                continue
            if long_d < end_d:
                par, k = i % 2, j
                lit = dt < db
            else:
                par, k = j % 2, i
                lit = dl < dr
            if par == 1:
                out[j, i] = d0
            elif lit:
                out[j, i] = d1 if (k % 3 == 0 or (i * 7 + j * 13) % 11 == 0) else d2
            else:
                out[j, i] = d0 if (i * 7 + j * 13) % 9 == 0 else d1
    return out


sx, sy, sw_, sh_ = SHRINE
# shadow of the shrine on the gravel
for y in range(sy * 2 + 2, (sy + sh_) * 2 + 3):
    for x in range(sx * 2 + 2, (sx + sw_) * 2 + 3):
        if not (sx * 2 <= x < (sx + sw_) * 2 and sy * 2 <= y < (sy + sh_) * 2) and cv.inb(x, y):
            nm = cv.names[cv.px[y, x]]
            m_ = {"t1": "t0", "t0": "t2", "wh": "t1"}
            if nm in m_:
                cv.px[y, x] = C(m_[nm])
cv.px[sy * 2:(sy + sh_) * 2, sx * 2:(sx + sw_) * 2] = roof_pattern(sw_ * 2, sh_ * 2, ("b0", "k1", "k2", "k2"))
# red veranda (engawa) with railings and stone steps in front of the hall
vy = (sy + sh_) * 2
vx0, vx1 = sx * 2 + 2, (sx + sw_) * 2 - 2
cv.rect(vx0, vy, vx1, vy + 2, "r1")
cv.hline(vx0, vx1, vy, "r2")
cv.hline(vx0, vx1, vy + 1, "r0")
for xx in range(vx0, vx1, 2):
    cv.set(xx, vy, "r3")
for j in range(2):
    cv.hline(sx * 2 + 2, (sx + sw_) * 2 - 2, vy + 2 + j, "t1" if j == 0 else "t2")
cv.set(sx * 2 + 2, vy + 2, "wh")


def red_rail_h(cx0, cx1, cy):
    for cx in range(cx0, cx1):
        X, Y = cx * 2, cy * 2
        cv.hline(X, X + 2, Y, "r2"); cv.hline(X, X + 2, Y + 1, "r1")
        if cx % 2 == 0:
            cv.set(X, Y + 1, "r0")


def red_rail_v(cx, cy0, cy1):
    for cy in range(cy0, cy1):
        X, Y = cx * 2, cy * 2
        cv.vline(X, Y, Y + 2, "r2"); cv.vline(X + 1, Y, Y + 2, "r1")
        if cy % 2 == 0:
            cv.set(X + 1, Y, "r0")


def red_post(cx, cy):
    X, Y = cx * 2, cy * 2
    cv.rect(X, Y, X + 2, Y + 2, "r1")
    cv.set(X, Y, "r3"); cv.set(X + 1, Y + 1, "r0")


fx, fy, fw, fh = FENCE
gap0, gap1 = fx + fw // 2 - 1, fx + fw // 2 + 2          # 3-cell entrance
red_rail_h(fx, fx + fw, fy)
red_rail_h(fx, gap0, fy + fh - 1)
red_rail_h(gap1, fx + fw, fy + fh - 1)
red_rail_v(fx, fy, fy + fh)
red_rail_v(fx + fw - 1, fy, fy + fh)
for (cx, cy) in [(fx, fy), (fx + fw - 1, fy), (fx, fy + fh - 1), (fx + fw - 1, fy + fh - 1),
                 (gap0, fy + fh - 1), (gap1 - 1, fy + fh - 1), (fx + fw // 2, fy), (fx, fy + fh // 2), (fx + fw - 1, fy + fh // 2),
                 (fx + 3, fy), (fx + fw - 4, fy)]:
    red_post(cx, cy)

# torii (seen from above): kasagi + nuki + pillar feet
TX0, TX1, TY = 47, 54, 12         # cells x0..x1 (exclusive), kasagi row (cells)
kx0, kx1 = TX0 * 2 - 1, TX1 * 2 + 1
ky = TY * 2
shade = {"m1": "m0", "m2": "m1", "e1": "e0", "e2": "e1", "e0": "e0", "t0": "t2", "t1": "t0", "lf": "m0", "wh": "t1"}
for y in range(ky + 3, ky + 12):
    for x in range(kx0 + 2, kx1 + 2):
        if cv.inb(x, y) and cv.names[cv.px[y, x]] in shade:
            cv.px[y, x] = C(shade[cv.names[cv.px[y, x]]])
cv.rect(kx0 + 1, ky - 1, kx1 - 1, ky, "b0")            # shimaki (dark cap)
cv.rect(kx0, ky, kx1, ky + 1, "r3")
cv.rect(kx0, ky + 1, kx1, ky + 3, "r1")
cv.rect(kx0 + 1, ky + 3, kx1 - 1, ky + 4, "r0")
cv.set(kx0, ky - 1, "r2"); cv.set(kx1 - 1, ky - 1, "r2")     # upturned tips
nx0, nx1 = kx0 + 2, kx1 - 2
cv.rect(nx0, ky + 5, nx1, ky + 7, "r1")
cv.hline(nx0, nx1, ky + 5, "r2")
cv.hline(nx0, nx1, ky + 7, "r0")
for px_ in (kx0 + 3, kx1 - 5):                          # pillars + stone feet
    cv.rect(px_, ky + 5, px_ + 2, ky + 9, "r1")
    cv.vline(px_ + 1, ky + 5, ky + 9, "r0")
    cv.rect(px_ - 1, ky + 9, px_ + 3, ky + 11, "t1")
    cv.hline(px_ - 1, px_ + 3, ky + 10, "t2")
cv.rect((kx0 + kx1) // 2 - 1, ky + 5, (kx0 + kx1) // 2 + 1, ky + 7, "wh")   # plaque

# stone lanterns beside the approach
def stone_lantern(cx, cy):
    X, Y = cx * 2, cy * 2
    cv.rect(X, Y, X + 2, Y + 2, "t1")
    cv.set(X + 1, Y + 1, "t2")
    cv.set(X, Y, "wh")
    # shadow
    for (ax, ay) in ((X + 2, Y + 1), (X + 2, Y + 2), (X + 1, Y + 2), (X, Y + 2)):
        if cv.inb(ax, ay) and cv.names[cv.px[ay, ax]] in ("m1", "m2", "t1", "t0", "e1", "e2", "wh"):
            cv.px[ay, ax] = C("m0" if cv.names[cv.px[ay, ax]].startswith("m") else "t2")


for (cx, cy) in [(45, 14), (55, 14), (47, 7), (53, 7)]:
    stone_lantern(cx, cy)

# --------------------------------------------------------------------------------------
# red maple trees, fallen leaves, white stones, shoots
# --------------------------------------------------------------------------------------
warm_cells = np.zeros((CH, CW), bool)


def maple(cx, cy, rx, ry):
    """Round maple crown (art px). Clumped foliage: bright NW, dark SE, dark leaf gaps."""
    for y in range(int(cy - ry), int(cy + ry) + 5):
        for x in range(int(cx - rx), int(cx + rx) + 5):
            if cv.inb(x, y) and ((x - cx - 2) / (rx + 0.5)) ** 2 + ((y - cy - 3) / (ry + 0.5)) ** 2 <= 1:
                nm = cv.names[cv.px[y, x]]
                if nm in smap and not wpx[y, x]:
                    cv.px[y, x] = C(smap[nm])
    clump = value_noise(rng, H, W, 3)
    clump2 = value_noise(rng, H, W, 6)
    for y in range(int(cy - ry) - 2, int(cy + ry) + 3):
        for x in range(int(cx - rx) - 2, int(cx + rx) + 3):
            dx, dy = (x - cx) / rx, (y - cy) / ry
            ang = math.atan2(dy, dx)
            rr = 1 + 0.07 * math.sin(ang * 8 + cx) + 0.05 * math.sin(ang * 3 + cy)
            d = math.hypot(dx, dy)
            if d <= rr and cv.inb(x, y):
                lit = (-dx - dy) / 1.4                       # -1 (SE) .. 1 (NW)
                v_ = clump[y, x] * 0.6 + clump2[y, x] * 0.4 + lit * 0.35
                if d > rr - 0.17 and lit < 0.1:
                    col = "r0"
                elif v_ > 0.92:
                    col = "r3"
                elif v_ > 0.62:
                    col = "r2"
                elif v_ > 0.34:
                    col = "r1"
                else:
                    col = "r0"
                if rng.random() < 0.14:
                    col = {"r0": "r1", "r1": "r2", "r2": "r3", "r3": "r2"}[col] if rng.random() < 0.6 else {"r0": "r0", "r1": "r0", "r2": "r1", "r3": "r3"}[col]
                cv.set(x, y, col)


for (mx, my, rx, ry) in MAPLES:
    maple(mx, my, rx, ry)


def leaf_pile(cx, cy, rx, ry, dens=0.82):
    for y in range(int(cy - ry) - 1, int(cy + ry) + 2):
        for x in range(int(cx - rx) - 1, int(cx + rx) + 2):
            d = math.hypot((x - cx) / rx, (y - cy) / ry)
            if d <= 1.0 + 0.2 * math.sin((x + y) * 0.9) and cv.inb(x, y) and not wpx[y, x] and not bpx[y, x]:
                if rng.random() < dens * (1.1 - 0.5 * d):
                    r_ = rng.random()
                    cv.set(x, y, "r1" if r_ < 0.4 else ("r2" if r_ < 0.75 else ("r3" if r_ < 0.9 else "r0")))


for (lx, ly, rx, ry) in LEAF_PILES:
    leaf_pile(lx, ly, rx, ry)
# scattered single red leaves (never enough to change a cell by themselves)
for _ in range(45):
    x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
    if cv.px[y, x] in GROUND_IDS and not bpx[y, x]:
        cv.set(x, y, "r2" if rng.random() < 0.6 else "r1")

# white stones (small clusters beside the trails)
def white_stones(x, y):
    for (dx, dy, w_) in ((0, 0, 3), (2, 2, 2), (-2, 1, 2)):
        for i in range(w_):
            if cv.inb(x + dx + i, y + dy) and cv.px[y + dy, x + dx + i] in GROUND_IDS:
                cv.set(x + dx + i, y + dy, "wh" if i < w_ - 1 else "t1")
                if cv.inb(x + dx + i, y + dy + 1) and cv.px[y + dy + 1, x + dx + i] in GROUND_IDS:
                    cv.set(x + dx + i, y + dy + 1, "t2")


for _ in range(34):
    x, y = int(rng.integers(2, W - 6)), int(rng.integers(2, H - 4))
    if -1.5 < e_edge[y, x] < 3.0 and not wpx[y, x] and not bpx[y, x] and not (82 <= x <= 112 and y <= 24):
        white_stones(x, y)
# bamboo shoots on the moss
for _ in range(40):
    x, y = int(rng.integers(2, W - 3)), int(rng.integers(2, H - 4))
    if cv.px[y, x] in (C("m0"), C("m1"), C("m2")) and not bpx[y, x] and e_edge[y, x] > 3 and not wpx[y, x] and not (80 <= x <= 114 and y <= 26):
        cv.set(x, y, "lf"); cv.set(x + 1, y, "m2"); cv.set(x, y + 1, "m0"); cv.set(x + 1, y + 1, "m0")

# fallen bamboo culms and ferns on the forest floor
def fallen_culm(x, y, L, dx, dy):
    pts = [(x + dx * k, y + dy * k) for k in range(L)]
    if not all(cv.inb(px_, py_) and cv.px[py_, px_] in GROUND_IDS and not wpx[py_, px_] and not bpx[py_, px_] for px_, py_ in pts):
        return False
    for k, (px_, py_) in enumerate(pts):
        cv.set(px_, py_, "lf" if k % 4 != 3 else "wd1")
        cv.set(px_, py_ + 1, "m0")
    return True


cnt = 0
for _ in range(400):
    if cnt >= 8:
        break
    x, y = int(rng.integers(3, W - 14)), int(rng.integers(3, H - 12))
    dx, dy = [(1, 0), (1, 1), (1, -1)][int(rng.integers(0, 3))]
    if e_edge[y, x] > 2 and not (80 <= x <= 114 and y <= 30):
        if fallen_culm(x, y, int(rng.integers(8, 13)), dx, dy):
            cnt += 1
for _ in range(60):
    x, y = int(rng.integers(3, W - 4)), int(rng.integers(3, H - 4))
    if cv.px[y, x] in (C("m0"), C("m1"), C("m2")) and e_edge[y, x] > 2 and not wpx[y, x] and not bpx[y, x] and not (80 <= x <= 114 and y <= 30):
        for (ox, oy) in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-2, 1), (2, 1), (-1, -2), (1, -2)):
            if cv.inb(x + ox, y + oy) and cv.px[y + oy, x + ox] in (C("m0"), C("m1"), C("m2")):
                cv.set(x + ox, y + oy, "m2" if (ox + oy) % 2 == 0 else "lf")

# --------------------------------------------------------------------------------------
# class fixes + output
# --------------------------------------------------------------------------------------
n_w = cv.fix_walls(wall_cells, ["b0"])
n_wa = cv.fix_water(wmask, "w1")
# floor cells that accidentally became too dark -> lighten their darkest pixels
n_fl = 0
for _ in range(8):
    h_, s_, l_ = cv.classify()
    bad = (l_ < 0.175) & ~wall_cells
    if not bad.any():
        break
    for cy_, cx_ in zip(*np.nonzero(bad)):
        blk = cv.px[cy_ * 2:cy_ * 2 + 2, cx_ * 2:cx_ * 2 + 2]
        lum = cv.rgb[blk].astype(int).sum(-1)
        j = np.unravel_index(np.argmin(lum), lum.shape)
        blk[j] = C("m1")
        n_fl += 1
print("fixed wall:", n_w, "water:", n_wa, "dark-floor:", n_fl, "colours:", cv.unique_colors())
out = os.path.join(HERE, "bamboo_path.png")
cv.save(out)
print("saved", out)
