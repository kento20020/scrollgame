#!/usr/bin/env python3
"""Draw 藍's actions in high resolution and save them as a Pixelorama project.

    python3 tools/make_aoi_pxo.py        → characters/aoi/aoi.pxo (+ preview_actions.png / .gif)

Drawn from characters/aoi/portrait.png — the heroine at twice the detail of the old 28×54 sprite
(52×104: her face, the bamboo pattern, the fan) — so in the game every source pixel is half a world pixel.

The poses are big, in the manner of a fan-fighting stance: a low ready stance with the fan raised, striding
steps, a deep lunge that sweeps the open fan through a swirl of colour petals (the five colours of the game's
色の力), a knock-back, and a win that crouches behind the fan, springs into a turning leap and lands in pose.
Since the long dress hides her legs in the portrait, the picture is used as a puppet: the head, torso and
the front panel of the skirt come from the portrait, while legs, the fan arm and the fan are drawn for
every pose.

The canvas is 136×130 (44 px of room in front, 40 behind, 26 above), feet at (74, 130). The .pxo is the
source the game is built from (build.py reads it directly); after this first draft the frames are meant to
be touched up by hand in Pixelorama — and this script then left alone, since it overwrites the file. It
follows Pixelorama's own save format (src/Autoload/OpenSave.gd, src/Classes/Project.gd): a ZIP holding
data.json, a mimetype and one raw RGBA8 buffer per cel at image_data/frames/<frame>/layer_<layer>, both
counted from 1.
"""
import io
import json
import math
import pathlib
import random
import zipfile

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'characters' / 'aoi'
SRC = OUT / 'portrait.png'
PL, PR, PT = 44, 40, 26           # room in front (left), behind (right) and above
W, H = 52 + PL + PR, 104 + PT     # 136 × 130
FEET = (74, 130)
FPS = 20
OL = np.array([232, 226, 214, 255], np.uint8)   # the light outline the sprite always had
T0 = np.zeros(4, np.uint8)

P = {  # palette taken from the portrait (dark → light)
    'A': (1, 0, 17), 'D': (15, 14, 39), 'E': (25, 29, 54), 'H': (39, 46, 83), 'I': (66, 62, 77), 'J': (88, 92, 111),
    'K': (126, 113, 119), 'L': (141, 144, 155), 'M': (183, 159, 153), 'N': (215, 193, 183), 'O': (243, 215, 196),
}
FX = {'white': (255, 251, 238), 'mist': (178, 214, 222), 'gold': (232, 196, 96), 'dust': (196, 190, 178)}
PETALS = [(214, 88, 62), (232, 196, 96), (111, 174, 94), (91, 134, 201), (155, 114, 196)]  # 朱 金 緑 藍 紫


def c(k, a=255):
    v = P.get(k) or FX[k]
    return np.array([v[0], v[1], v[2], a], np.uint8)


def cx(x):  # portrait x → canvas x
    return x + PL


def cy(y):  # portrait y → canvas y
    return y + PT


# ---------------------------------------------------------------- pieces of the portrait
SRC_A = np.array(Image.open(SRC).convert('RGBA'))


def piece_full():
    return SRC_A.copy()


def piece_body(face_kind=None):
    """Head, torso and the left arm (rows 0–64) plus the skirt's front panel (rows 65–90, x 25–36);
    her right arm and fan are lifted off, so are the skirt's sides and the feet (drawn per pose)."""
    a = SRC_A.copy()
    for y in range(38, 45):
        a[y, 19:23] = T0
    for y in range(45, 65):
        a[y, : (23 if y <= 56 else 20 if y <= 60 else 19)] = T0
    a[65:] = T0
    panel = SRC_A[65:91, 25:37]
    a[65:91, 25:37] = panel
    if face_kind:
        a = face(a, face_kind)
    return a


def face(a, kind):
    """'pain' >_<, 'joy' ^ ^ with a smile."""
    b = a.copy()
    O, A, M = c('O'), c('A'), c('M')
    for y in range(16, 20):
        for x in list(range(23, 27)) + list(range(31, 36)):
            b[y, x] = O
    if kind == 'pain':
        for x, y in ((23, 16), (24, 17), (25, 18), (24, 19), (23, 20), (35, 16), (34, 17), (33, 18), (34, 19), (35, 20)):
            b[y, x] = A
        for x in range(27, 31):
            b[23, x] = M
        b[24, 28] = A; b[24, 29] = A
    if kind == 'joy':
        for x, y in ((23, 19), (24, 18), (25, 17), (26, 18), (27, 19), (31, 19), (32, 18), (33, 17), (34, 18), (35, 19)):
            b[y, x] = A
        for x in range(27, 31):
            b[23, x] = O
        for x, y in ((26, 22), (27, 23), (28, 23), (29, 23), (30, 22)):
            b[y, x] = M if x in (26, 30) else A
    return b


def hair(a, dl, dr):
    """Swing the hair ends: left ends (x ≤ 18) and right ends (x ≥ 41) of rows 26–46 by dl / dr."""
    b = a.copy()
    for y in range(26, 47):
        k = (y - 26) / 20
        for d, xs, inner in ((dl, range(0, 19), lambda nx: nx > 18), (dr, range(41, 52), lambda nx: nx < 41)):
            s = round(d * k)
            if not s:
                continue
            row = a[y].copy()
            seg = [x for x in xs if row[x, 3]]
            for x in seg:
                b[y, x] = T0
            for x in seg:
                nx = x + s
                if 0 <= nx < 52 and not (inner(nx) and a[y, nx, 3]):
                    b[y, nx] = row[x]
    return b


def lean(a, top, pivot):
    """Shear about row `pivot`: the top row moves `top` px (negative = forward), rows below the pivot stay."""
    b = np.zeros_like(a)
    for y in range(a.shape[0]):
        d = round(top * max(0, pivot - y) / pivot)
        for x in np.nonzero(a[y, :, 3])[0]:
            if 0 <= x + d < a.shape[1]:
                b[y, x + d] = a[y, x]
    return b


# ---------------------------------------------------------------- drawing on the canvas
def put(a, x, y, v, over=True):
    x, y = int(round(x)), int(round(y))
    if 0 <= x < W and 0 <= y < H and (over or a[y, x, 3] == 0):
        a[y, x] = v


def paste(a, piece, dx, dy):
    ys, xs = np.nonzero(piece[..., 3])
    for y, x in zip(ys, xs):
        put(a, cx(x) + dx, cy(y) + dy, piece[y, x])


def limb(a, joints, radii, rim=True, light=(-1, -1)):
    """A tapered limb through `joints` (canvas coords) with radii per joint; lit from the upper left."""
    segs = list(zip(joints, joints[1:], radii, radii[1:]))
    x0 = int(min(p[0] for p in joints) - 6); x1 = int(max(p[0] for p in joints) + 6)
    y0 = int(min(p[1] for p in joints) - 6); y1 = int(max(p[1] for p in joints) + 6)
    edge = []
    for py in range(max(0, y0), min(H, y1)):
        for px in range(max(0, x0), min(W, x1)):
            best = None
            for (ax, ay), (bx, by), ra, rb in segs:
                vx, vy = bx - ax, by - ay
                L2 = vx * vx + vy * vy or 1
                t = max(0, min(1, ((px - ax) * vx + (py - ay) * vy) / L2))
                qx, qy = ax + vx * t, ay + vy * t
                d = math.hypot(px - qx, py - qy)
                r = ra + (rb - ra) * t
                if best is None or d - r < best[0]:
                    L = math.sqrt(L2)
                    side = ((px - qx) * light[0] + (py - qy) * light[1]) / max(1e-6, d) if d > 0 else 1
                    best = (d - r, side, d / max(r, 0.5))
            gap, side, rel = best
            if gap <= 0:
                k = 'O' if side > 0.25 else 'N' if side > -0.45 or rel < 0.45 else 'M'
                if rel > 0.82 and side < -0.2:
                    k = 'K'
                a[py, px] = c(k)
            elif gap <= 0.9 and rim:
                edge.append((px, py))
    for px, py in edge:
        put(a, px, py, c('A'), over=False)


def shoe(a, ankle, toe):
    """A low heeled shoe at the ankle, pointing `toe` (-1 left, +1 right)."""
    axp, ayp = ankle
    for dy in range(0, 5):
        for dx in range(-2, 8):
            x = axp + dx * toe
            y = ayp + dy
            length = 8 - max(0, 3 - dy) if dy < 3 else 8
            if dx > length or (dy == 0 and dx > 4):
                continue
            k = 'A' if dy == 4 or dx == length or dx == -2 else 'J' if dy == 1 and 0 <= dx <= 3 else 'E'
            put(a, x, y, c(k))
    put(a, axp - 2 * toe, ayp + 5, c('A')); put(a, axp - 1 * toe, ayp + 5, c('A'))   # heel


def leg(a, hip, knee, ankle, toe):
    limb(a, [hip, knee, ankle], [4.4, 3.3, 2.5])
    shoe(a, (round(ankle[0]), round(ankle[1]) + 1), toe)


def arm(a, shoulder, elbow, hand):
    limb(a, [shoulder, elbow, hand], [2.5, 2.1, 1.8])
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx * dx + dy * dy <= 5:
                put(a, hand[0] + dx, hand[1] + dy, c('O') if dx * dx + dy * dy <= 2 else c('N'))


def fan_open(a, apex, r, a0, a1):
    ax, ay = apex
    span = (a1 - a0) % 360 or 360
    for py in range(int(ay - r - 1), int(ay + r + 2)):
        for px in range(int(ax - r - 1), int(ax + r + 2)):
            u, v = px - ax, py - ay
            d = math.hypot(u, v)
            t = (math.degrees(math.atan2(v, u)) - a0) % 360
            if d > r or t > span or d < 1:
                continue
            f = t / span * 10
            k = 'A' if d > r - 1.3 or t < 2.5 or t > span - 2.5 else 'J' if f % 1 < 0.12 else ('N' if int(f) % 2 == 0 else 'O')
            if d < r * 0.38:
                k = 'J' if f % 1 < 0.3 else 'D'
            if d < 2.2:
                k = 'A'
            put(a, px, py, c(k))


def fan_shut(a, hand, ang, n=16):
    hx, hy = hand
    ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    for i in range(n):
        for w in (-1, 0, 1):
            put(a, hx + ux * i - uy * w, hy + uy * i + ux * w, c('A') if w or i == n - 1 else c('N') if i % 3 else c('J'))


def outline(a):
    out = a.copy()
    op = a[..., 3] > 0
    nb = np.zeros_like(op)
    nb[1:] |= op[:-1]; nb[:-1] |= op[1:]; nb[:, 1:] |= op[:, :-1]; nb[:, :-1] |= op[:, 1:]
    out[nb & ~op] = OL
    return out


def arcpts(ox, oy, r, frm, to, n):
    return [(ox + r * math.cos(math.radians(frm + (to - frm) * i / (n - 1))), oy + r * math.sin(math.radians(frm + (to - frm) * i / (n - 1))), i / (n - 1)) for i in range(n)]


def swirl(a, ox, oy, r, frm, to, n=34, spread=5, seed=1, streak=True):
    """A swirl of colour petals along an arc from `frm` to `to` degrees (the head is at `to`)."""
    rnd = random.Random(seed)
    if streak:
        for x, y, t in arcpts(ox, oy, r, frm, to, 120):
            for w in range(-1, 3 if t > 0.5 else 1):
                put(a, x, y + w, c('mist', int(90 + 150 * t)), over=False)
    for x, y, t in arcpts(ox, oy, r, frm, to, n):
        if rnd.random() < 0.25 * (1 - t):
            continue
        x += rnd.uniform(-spread, spread) * (1.2 - t); y += rnd.uniform(-spread, spread) * (1.2 - t)
        col = PETALS[int(rnd.random() * 5)]
        alpha = int(110 + 145 * t)
        ang = math.radians(frm + (to - frm) * t + 90 + rnd.uniform(-40, 40))
        ux, uy = math.cos(ang), math.sin(ang)
        for i in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5):              # a petal: 6 long, up to 3 wide
            for j in ((0,) if abs(i) > 2 else (-1, 0, 1) if abs(i) < 1 else (-0.5, 0.5)):
                put(a, x + ux * i - uy * j, y + uy * i + ux * j, np.array([*col, alpha], np.uint8), over=False)


def smear(a, ox, oy, r0, r1, frm, to):
    """A white swing smear, thickest at the head (`to`)."""
    for x, y, t in arcpts(ox, oy, 1, frm, to, 160):
        pass
    span = to - frm
    for py in range(H):
        for px in range(W):
            if a[py, px, 3]:
                continue
            d = math.hypot(px - ox, py - oy)
            ang = math.degrees(math.atan2(py - oy, px - ox))
            t = None
            for k in (-720, -360, 0, 360, 720):
                tt = (ang + k - frm) / span if span else 0
                if 0 <= tt <= 1:
                    t = tt
                    break
            if t is None:
                continue
            if r1 - (r1 - r0) * t <= d <= r1:
                put(a, px, py, c('white') if t > 0.7 else c('mist', int(90 + 150 * t)), over=False)


def star(a, x0, y0, n, k='white'):
    for i in range(-n, n + 1):
        put(a, x0 + i, y0, c(k), over=False); put(a, x0, y0 + i, c(k), over=False)
    for i in range(-(n // 2), n // 2 + 1):
        put(a, x0 + i, y0 + i, c(k), over=False); put(a, x0 + i, y0 - i, c(k), over=False)
    put(a, x0, y0, c('white'))


def dust(a, x0, y0, flip=1):
    for dx, dy, r in ((0, 0, 3.2), (6 * flip, -2, 2.6), (11 * flip, 0, 2.0)):
        for y in range(-4, 5):
            for x in range(-4, 5):
                if x * x + y * y <= r * r:
                    put(a, x0 + dx + x, y0 + dy + y, c('dust', 210 if x * x + y * y < (r - 1) ** 2 else 120), over=False)


# ---------------------------------------------------------------- one pose
def figure(*, body_dx=0, body_dy=0, top=0, face_kind=None, dl=0, dr=0, legs=None, arm_pose=None, fan=None, full=False):
    """Compose a frame. Coordinates are in portrait space (x 0–51, ground at y 103) and converted here.
    body_dx/dy move the torso block (hips included); `top` leans it (negative = forward).
    legs = ((knee, ankle, toe) front, (knee, ankle, toe) back); arm_pose = (elbow, hand), each point given
    relative to her shoulder; fan = ('open', r, a0, a1) or ('shut', angle)."""
    a = np.zeros((H, W, 4), np.uint8)
    piece = piece_full() if full else piece_body(face_kind)
    if full and face_kind:
        piece = face(piece, face_kind)
    if dl or dr:
        piece = hair(piece, dl, dr)
    pivot = 64
    if top:
        piece = lean(piece, top, 90 if full else pivot)
    hip_f = (cx(25) + body_dx, cy(64) + body_dy)
    hip_b = (cx(36) + body_dx, cy(64) + body_dy)
    if legs:
        (kf, af, tf), (kb, ab, tb) = legs
        leg(a, hip_b, (cx(kb[0]), cy(kb[1])), (cx(ab[0]), cy(ab[1])), tb)       # the far leg, behind
    paste(a, piece, body_dx, body_dy)
    if legs:
        leg(a, hip_f, (cx(kf[0]), cy(kf[1])), (cx(af[0]), cy(af[1])), tf)       # the near leg, in front
    if arm_pose:
        sh = (cx(21) + body_dx + round(top * (pivot - 39) / pivot), cy(39) + body_dy)
        (ex, ey), (hx, hy) = arm_pose
        el, hd = (sh[0] + ex, sh[1] + ey), (sh[0] + hx, sh[1] + hy)
        arm(a, sh, el, hd)
        if fan:
            if fan[0] == 'open':
                fan_open(a, hd, *fan[1:])
            else:
                fan_shut(a, hd, *fan[1:])
    return outline(a), (sh if arm_pose else None)


def mirror(a):
    """Turn her round (for the spin), keeping the feet anchor where it was."""
    b = np.zeros_like(a)
    for x in range(W):
        nx = 2 * FEET[0] - x
        if 0 <= nx < W:
            b[:, nx] = a[:, x]
    return b


# ---------------------------------------------------------------- the actions
def frames():
    act = {}
    G = 101  # ankle height on the ground

    # 構え — a low ready stance, the fan raised open beside her face; breathing, the fan fluttering
    def ready(dy, flutter, dl=0, dr=0):
        f, _ = figure(body_dy=dy, top=-2, dl=dl, dr=dr,
                      legs=(((17, 84 + dy // 2), (12, G), -1), ((42, 84 + dy // 2), (47, G), 1)),
                      arm_pose=((-8, 8), (-10, -6)), fan=('open', 15, 200 + flutter, 322 + flutter))
        return f
    act['idle'] = [(ready(5, 0), 5), (ready(7, 4, 1, 2), 5), (ready(5, 0), 5), (ready(7, -4, -1, -1), 5)]

    # 歩き — striding: legs swinging out from under the panel, a bounce, leaning in, hair streaming behind
    def stride(dy, front, back, hand):
        f, _ = figure(body_dy=dy, top=-4, dr=4, dl=2, legs=(front, back),
                      arm_pose=((-6, 9), hand), fan=('shut', 120, 15))
        return f
    walk = [(ready(5, 0), 2)]
    walk.append((stride(3, ((15, 82), (6, G), -1), ((42, 84), (50, G - 2), 1), (-12, 14)), 2.2))   # contact
    walk.append((stride(-1, ((24, 82), (24, G - 4), -1), ((34, 82), (34, G), 1), (-8, 17)), 2.2))   # passing
    walk.append((stride(3, ((40, 84), (48, G - 2), -1), ((20, 82), (10, G), -1), (-4, 18)), 2.2))   # contact, other foot
    walk.append((stride(-1, ((30, 82), (30, G), -1), ((28, 82), (32, G - 4), 1), (-8, 17)), 2.2))   # passing
    act['walk'] = walk

    # 扇 — weight back with the fan drawn high (shown just before the gust leaves), a deep lunge sweeping
    # the open fan forward through a swirl of colour petals, the follow-through, rising back to the stance
    a0, _ = figure(body_dx=4, body_dy=6, top=7, dl=-3, dr=-3,
                   legs=(((12, 86), (2, G), -1), ((42, 82), (44, G), 1)),
                   arm_pose=((-2, -12), (6, -26)), fan=('shut', 280, 17))
    a1, sh = figure(body_dx=-8, body_dy=14, top=-8, dl=4, dr=6,
                    legs=(((4, 84), (-2, G), -1), ((44, 88), (58, G), 1)),
                    arm_pose=((-12, 2), (-27, 1)), fan=('open', 19, 124, 248))
    swirl(a1, sh[0] + 6, sh[1] - 2, 38, -80, -175, seed=3)
    smear(a1, sh[0] + 2, sh[1] + 1, 30, 37, -130, -185)
    a2, sh = figure(body_dx=-8, body_dy=14, top=-6, dl=5, dr=7,
                    legs=(((4, 84), (-2, G), -1), ((44, 88), (58, G), 1)),
                    arm_pose=((-9, 10), (-18, 24)), fan=('open', 17, 92, 200))
    swirl(a2, sh[0] + 2, sh[1] + 2, 36, -150, -240, seed=5)
    smear(a2, sh[0], sh[1] + 4, 28, 34, -180, -235)
    a3, _ = figure(body_dx=-3, body_dy=9, top=-3, dl=2, dr=3,
                   legs=(((10, 85), (6, G), -1), ((43, 85), (52, G), 1)),
                   arm_pose=((-8, 8), (-11, -2)), fan=('open', 14, 205, 320))
    act['attack'] = [(a0, 2.4), (a1, 1.4), (a2, 1.8), (a3, 1.8), (ready(5, 0), 1.6)]

    # 被弾 — thrown back off her feet, eyes squeezed shut, hair flung forward, a burst where it landed
    h0, sh = figure(body_dx=6, body_dy=6, top=13, face_kind='pain', dl=-6, dr=-5,
                    legs=(((14, 80), (6, 92), -1), ((40, 84), (40, G), 1)),
                    arm_pose=((-4, 10), (-12, 14)), fan=('shut', 170, 15))
    star(h0, cx(4), cy(34), 7); star(h0, cx(-6), cy(24), 2, 'gold'); star(h0, cx(6), cy(18), 2)
    h1, _ = figure(body_dx=4, body_dy=7, top=8, face_kind='pain', dl=-3, dr=-3,
                   legs=(((16, 84), (10, G), -1), ((41, 84), (44, G), 1)),
                   arm_pose=((-6, 10), (-12, 16)), fan=('shut', 150, 15))
    star(h1, cx(2), cy(32), 3)
    act['hurt'] = [(h0, 1.6), (h1, 2.4), (ready(5, 0), 2.4)]

    # 勝ち — hiding low behind the open fan, springing up, a turn in the air inside a ring of petals,
    # the top of the leap with the fan flung open, landing, and the pose held with the fan raised
    v0, _ = figure(body_dy=20, top=-2, legs=(((10, 98), (26, G), 1), ((46, 92), (40, G), 1)),
                   arm_pose=((-8, -4), (-4, -16)), fan=('open', 16, 200, 345))
    v1, _ = figure(body_dy=-10, top=1, dl=2, dr=2, legs=(((26, 88), (24, 102 - 10), -1), ((34, 88), (36, 102 - 10), 1)),
                   arm_pose=((-6, -12), (-8, -26)), fan=('shut', 270, 17))
    sp, sh = figure(body_dy=-20, dl=3, dr=3, legs=(((18, 70), (30, 80), 1), ((40, 72), (34, 84), 1)),
                    arm_pose=((-6, -12), (-8, -26)), fan=('shut', 265, 17))
    v2 = mirror(sp)
    swirl(v2, FEET[0], cy(46) - 20, 46, 20, 340, n=46, seed=7)
    v3, sh = figure(body_dy=-24, face_kind='joy', legs=(((10, 74), (2, 82), -1), ((42, 72), (52, 66), 1)),
                    arm_pose=((-6, -12), (-9, -26)), fan=('open', 19, 190, 335))
    swirl(v3, FEET[0], cy(40) - 24, 50, 120, 400, n=40, seed=9, streak=False)
    for x, y, n, k in ((cx(54), cy(-18), 4, 'gold'), (cx(64), cy(0), 3, 'white'), (cx(-14), cy(-10), 3, 'gold'),
                       (cx(44), cy(-24), 2, 'white'), (cx(-20), cy(12), 2, 'white'), (cx(70), cy(-12), 2, 'gold')):
        star(v3, x, y, n, k)
    v4, _ = figure(body_dy=9, face_kind='joy', legs=(((14, 86), (8, G), -1), ((44, 86), (50, G), 1)),
                   arm_pose=((-6, -12), (-9, -26)), fan=('open', 18, 190, 335))
    dust(v4, cx(6), H - 3, -1); dust(v4, cx(52), H - 3, 1)
    v5, _ = figure(body_dy=4, top=-1, face_kind='joy', legs=(((17, 84), (12, G), -1), ((42, 84), (47, G), 1)),
                   arm_pose=((-6, -12), (-9, -26)), fan=('open', 19, 190, 335))
    star(v5, cx(54), cy(-10), 2, 'gold'); star(v5, cx(-12), cy(-2), 2)
    act['win'] = [(v0, 2.4), (v1, 1.6), (v2, 2.2), (v3, 4.4), (v4, 2.0), (v5, 24.0)]
    return act


# ---------------------------------------------------------------- Pixelorama project
def write_pxo(path, act, layer_name='藍'):
    order = ['idle', 'walk', 'attack', 'hurt', 'win']
    tag_colors = {'idle': 'c9a64aff', 'walk': '8fc9b8ff', 'attack': '5b86c9ff', 'hurt': 'd0533aff', 'win': 'e58fa6ff'}
    frames_out, tags, n = [], [], 0
    for name in order:
        fr = act[name]
        tags.append({'name': name, 'color': tag_colors[name], 'from': n + 1, 'to': n + len(fr)})
        frames_out += fr
        n += len(fr)
    data = {
        'pixelorama_version': 'v1.2.3-stable', 'pxo_version': 7,
        'size_x': W, 'size_y': H, 'color_mode': 5,  # Image.FORMAT_RGBA8
        'tile_mode_x_basis_x': W, 'tile_mode_x_basis_y': 0, 'tile_mode_y_basis_x': 0, 'tile_mode_y_basis_y': H,
        'layers': [{
            'name': layer_name, 'visible': True, 'locked': False, 'blend_mode': 0, 'clipping_mask': False,
            'opacity': 1.0, 'parent': -1, 'effects': [], 'type': 0, 'new_cels_linked': False, 'metadata': {},
        }],
        'tags': tags, 'guides': [], 'symmetry_points': [W / 2, H / 2],
        'frames': [{'cels': [{'opacity': 1.0, 'z_index': 0, 'image_size': f'Vector2i({W}, {H})', 'metadata': {}}],
                    'duration': dur, 'metadata': {}} for _, dur in frames_out],
        'current_frame': 0, 'current_layer': 0, 'brushes': [], 'palettes': [], 'reference_images': [],
        'tilesets': [], 'vanishing_points': [], 'fps': float(FPS), 'include_blended_images': False,
        'user_data': '', 'metadata': {},
    }
    pv = io.BytesIO()
    Image.fromarray(frames_out[0][0], 'RGBA').resize((W * 2, H * 2), Image.NEAREST).save(pv, 'PNG')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('data.json', json.dumps(data, ensure_ascii=False))
        z.writestr('mimetype', 'application/x-pixelorama')
        z.writestr('preview.png', pv.getvalue())
        for i, (img, _) in enumerate(frames_out):
            z.writestr(f'image_data/frames/{i + 1}/layer_1', img.astype(np.uint8).tobytes())
    return tags


def previews(act):
    S, gap = 2, 6
    rows = ['idle', 'walk', 'attack', 'hurt', 'win']
    cols = max(len(act[r]) for r in rows)
    sheet = Image.new('RGBA', (cols * (W * S + gap) + gap, len(rows) * (H * S + gap) + gap), (29, 40, 71, 255))
    for j, r in enumerate(rows):
        for i, (img, _) in enumerate(act[r]):
            big = Image.fromarray(img, 'RGBA').resize((W * S, H * S), Image.NEAREST)
            sheet.alpha_composite(big, (gap + i * (W * S + gap), gap + j * (H * S + gap)))
    sheet.save(OUT / 'preview_actions.png')
    gif, durs = [], []
    for r in rows:
        for _ in range(2 if r in ('idle', 'walk') else 1):
            for img, d in act[r]:
                bg = Image.new('RGBA', (W * 2, H * 2), (29, 40, 71, 255))
                bg.alpha_composite(Image.fromarray(img, 'RGBA').resize((W * 2, H * 2), Image.NEAREST))
                gif.append(bg.convert('P', palette=Image.ADAPTIVE, colors=128)); durs.append(int(min(d, 16) * 1000 / FPS))
    gif[0].save(OUT / 'preview_actions.gif', save_all=True, append_images=gif[1:], duration=durs, loop=0, disposal=2)


if __name__ == '__main__':
    act = frames()
    tags = write_pxo(OUT / 'aoi.pxo', act)
    previews(act)
    print('wrote', OUT / 'aoi.pxo', f'{W}×{H}', [(t['name'], t['from'], t['to']) for t in tags])
