#!/usr/bin/env python3
"""Draw 藍's actions (idle / walk / attack / hurt / win) and save them as a Pixelorama project.

    python3 tools/make_aoi_pxo.py        → characters/aoi/aoi.pxo (+ preview_actions.png / .gif)

The .pxo is the source the game is built from (build.py reads it directly), so after this first
draft the frames are meant to be touched up by hand in Pixelorama (File ▸ Open aoi.pxo, edit, save).
Every action is an animation tag; frame durations are kept in the file.

The frames start from characters/aoi/sprite.png (28×54) and reuse the pixel transforms of
make_walk_frames.py; the canvas is 32×58 so a hop, a raised fan and small effects have room
(2 px each side, 4 px on top). The file follows Pixelorama's own save format (src/Autoload/OpenSave.gd
and src/Classes/Project.gd in the Pixelorama source): a ZIP holding data.json, a mimetype and one raw
RGBA8 buffer per cel at image_data/frames/<frame>/layer_<layer>, both counted from 1.
"""
import io
import json
import math
import pathlib
import sys
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_walk_frames as mw  # noqa: E402  (shared pixel transforms and palette)

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'characters' / 'aoi'
SW, SH = mw.W, mw.H            # source sprite 28×54
PADX, PADT = 2, 4
W, H = SW + PADX * 2, SH + PADT  # canvas 32×58, feet at (18, 58)
FPS = 10
OL = mw.OL
TRANSP = np.zeros(4, dtype=np.uint8)
FX = {  # effect colours (no outline)
    'paper': (233, 227, 211, 255), 'mist': (176, 205, 214, 200), 'gold': (226, 190, 92, 255),
    'spark': (255, 250, 236, 255), 'shu': (208, 83, 58, 255),
}


def col(ch):
    return mw.col(ch)


def place(b, dx=0, dy=0):
    """Put a 28×54 body onto the 32×58 canvas (feet on the bottom row unless dy moves it)."""
    out = np.zeros((H, W, 4), dtype=np.uint8)
    for y in range(SH):
        for x in range(SW):
            if b[y, x, 3] == 0:
                continue
            nx, ny = x + PADX + dx, y + PADT + dy
            if 0 <= nx < W and 0 <= ny < H:
                out[ny, nx] = b[y, x]
    return out


def outline(a):
    """1 px light outline around every opaque pixel (4-neighbourhood), as on the original sprite."""
    out = a.copy()
    op = a[..., 3] > 0
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            if op[y, x]:
                continue
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < a.shape[0] and 0 <= xx < a.shape[1] and op[yy, xx]:
                    out[y, x] = OL
                    break
    return out


def fx(a, pts, name):
    out = a.copy()
    for x, y in pts:
        if 0 <= x < W and 0 <= y < H and out[y, x, 3] == 0:
            out[y, x] = FX[name]
    return out


def over(a, b):
    """b painted over a (where b is opaque)."""
    out = a.copy()
    m = b[..., 3] > 0
    out[m] = b[m]
    return out


def sink(b, k=1, upto=29):
    """Rows 0..upto drop by k (shoulders settle / a crouch); the rows below stay."""
    out = b.copy()
    for y in range(upto, -1, -1):
        out[y + k] = b[y]
    for y in range(k):
        out[y] = TRANSP
    return out


def lean(b, top_dx, pivot=40):
    """Shear the body: the top row moves top_dx, easing to 0 at the pivot row and below."""
    d = [round(top_dx * max(0, pivot - y) / pivot) for y in range(SH)]
    return mw.shift_rows(b, d)


def wedge(b, apex, r, a0, a1, rim='K', ribs=('K', 'G', 's', 'G', 'g', 'G', 'K')):
    """An open fan: angles in degrees, 0 = right, 90 = down (image space)."""
    out = b.copy()
    ax, ay = apex
    span = (a1 - a0) % 360 or 360
    for y in range(max(0, int(ay - r - 1)), min(SH, int(ay + r + 2))):
        for x in range(max(0, int(ax - r - 1)), min(SW, int(ax + r + 2))):
            u, v = x - ax, y - ay
            rr = math.hypot(u, v)
            if rr > r or rr < 1.0:
                continue
            t = (math.degrees(math.atan2(v, u)) - a0) % 360
            if t > span:
                continue
            ch = ribs[min(int(t / span * len(ribs)), len(ribs) - 1)]
            if rr > r - 1.1:
                ch = rim
            if rr < 2.2:
                ch = 'k'
            out[y, x] = col(ch)
    return out


def drop_fan_arm(b):
    """Remove her right forearm, hand and the closed fan (left side of the picture)."""
    out = b.copy()
    for y in range(24, 38):
        lim = 10 if y <= 31 else 9
        for x in range(0, lim + 1):
            out[y, x] = TRANSP
    return out


def raised_arm(b, hand=(7, 15)):
    """Her right arm lifted from the shoulder up to `hand`, holding the fan open above it."""
    out = drop_fan_arm(b)
    sx, sy = 10, 25
    hx, hy = hand
    n = max(abs(hx - sx), abs(hy - sy))
    for i in range(n + 1):  # a 2 px arm: lit edge S, shade s, a darker inner line J near the elbow
        x = round(sx + (hx - sx) * i / n)
        y = round(sy + (hy - sy) * i / n)
        out[y, x] = col('S')
        if x + 1 < SW:
            out[y, x + 1] = col('s' if i < n * 0.6 else 'J')
    out[hy, hx] = col('S'); out[hy - 1, hx] = col('s'); out[hy, hx - 1] = col('J')  # hand
    return wedge(out, (hx - 1, hy - 1), 6.0, 186, 292)


# ---------------------------------------------------------------- the actions
def frames():
    src = mw.load()
    body = mw.strip_outline(src)
    act = {}

    # 待機: breathing (shoulders settle 1 px), hair tips swaying, the fan hand easing
    i1 = sink(mw.shift_rows(body, [0] * SH, mw.tip_segs(0, 0, +1)), 1)
    i2 = mw.raise_hand_fan(body, 1)
    i3 = sink(mw.shift_rows(body, [0] * SH, mw.tip_segs(0, -1, 0)), 1)
    act['idle'] = [(outline(place(b)), 3) for b in (body, i1, i2, i3)]

    # 歩き: the four steps drawn earlier by make_walk_frames.py (frame 0 = standing)
    wk = mw.make_walk_frames(body)
    act['walk'] = [(outline(place(b)), 1.5) for b in (body, wk[1], wk[2], wk[3])]

    # 扇: wind-up → sweep (with a gust) → follow-through → recover
    a0 = lean(mw.raise_hand_fan(body, 4), +1)
    a1 = mw.shift_rows(mw.fan_wedge(body), mw.profile(-1, 0), mw.tip_segs(-1, 0, +1))
    a2 = mw.shift_rows(wedge(over(drop_fan_arm(body), _arm_down(body)), (8, 33), 7.2, 95, 175), mw.profile(-1, 0), mw.tip_segs(-1, 0, +1))
    a3 = mw.make_attack_frames(body)[0]
    gust = [(1, 32), (0, 34), (0, 35), (0, 36), (1, 38), (1, 39), (2, 41)]
    act['attack'] = [
        (outline(place(a0)), 0.6),
        (fx(outline(place(a1)), gust, 'mist'), 0.8),
        (fx(outline(place(a2)), [(p[0] + 1, p[1] + 3) for p in gust[2:]], 'mist'), 0.8),
        (outline(place(a3)), 1.0),
    ]

    # 被弾: knocked back (lean away from the blow), a spark where it landed, then recovering
    h0 = lean(body, +3, pivot=46)
    h1 = lean(body, +1, pivot=46)
    spark = [(2, 22), (1, 23), (3, 23), (2, 24), (2, 21), (0, 23), (4, 23), (2, 25)]
    act['hurt'] = [(fx(outline(place(h0, dx=1)), spark, 'spark'), 1.5), (outline(place(h1)), 2.0)]

    # 勝ち: a crouch, a hop with the fan raised open, the top of the hop with sparkles, landing
    up = raised_arm(body)
    plus = lambda cx, cy: [(cx, cy), (cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
    w0 = outline(place(sink(body, 1, upto=45)))
    w1 = outline(place(up, dy=-2))
    w2 = fx(fx(outline(place(up, dy=-3)), plus(28, 4) + plus(25, 1) + [(30, 9)], 'gold'), plus(4, 2) + [(1, 7)], 'spark')
    w3 = fx(outline(place(sink(up, 1, upto=45))), [(28, 5), (2, 4)], 'gold')
    act['win'] = [(w0, 1.5), (w1, 1.5), (w2, 2.5), (w3, 4.0)]
    return act


def _arm_down(body):
    """The forearm and hand lowered after the sweep (kept from the original, moved 1 px down)."""
    out = np.zeros_like(body)
    for y in range(24, 33):
        for x in range(5, 11):
            if body[y, x, 3] and y + 1 < SH:
                out[y + 1, x] = body[y, x]
    return out


# ---------------------------------------------------------------- Pixelorama project
def write_pxo(path, act, layer_name='藍'):
    order = ['idle', 'walk', 'attack', 'hurt', 'win']
    tag_colors = {'idle': 'c9a64aff', 'walk': '8fc9b8ff', 'attack': '5b86c9ff', 'hurt': 'd0533aff', 'win': 'e58fa6ff'}
    frames_out, tags, n = [], [], 0
    for name in order:
        fr = act[name]
        tags.append({'name': name, 'color': tag_colors[name], 'from': n + 1, 'to': n + len(fr)})
        for img, dur in fr:
            frames_out.append((img, dur))
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
    preview = Image.fromarray(frames_out[0][0], 'RGBA').resize((W * 4, H * 4), Image.NEAREST)
    pv = io.BytesIO(); preview.save(pv, 'PNG')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('data.json', json.dumps(data, ensure_ascii=False))
        z.writestr('mimetype', 'application/x-pixelorama')
        z.writestr('preview.png', pv.getvalue())
        for i, (img, _) in enumerate(frames_out):
            z.writestr(f'image_data/frames/{i + 1}/layer_1', img.astype(np.uint8).tobytes())
    return tags


def previews(act):
    """A contact sheet of every action and a looping GIF of each (for the README / checking)."""
    S, gap = 6, 4
    rows = ['idle', 'walk', 'attack', 'hurt', 'win']
    cols = max(len(act[r]) for r in rows)
    sheet = Image.new('RGBA', (cols * (W * S + gap) + gap, len(rows) * (H * S + gap) + gap), (29, 40, 71, 255))
    for j, r in enumerate(rows):
        for i, (img, _) in enumerate(act[r]):
            big = Image.fromarray(img, 'RGBA').resize((W * S, H * S), Image.NEAREST)
            sheet.alpha_composite(big, (gap + i * (W * S + gap), gap + j * (H * S + gap)))
    sheet.save(OUT / 'preview_actions.png')
    gif = []
    durs = []
    for r in rows:
        for _ in range(2 if r in ('idle', 'walk') else 1):
            for img, d in act[r]:
                bg = Image.new('RGBA', (W * 5, H * 5), (29, 40, 71, 255))
                bg.alpha_composite(Image.fromarray(img, 'RGBA').resize((W * 5, H * 5), Image.NEAREST))
                gif.append(bg.convert('P', palette=Image.ADAPTIVE, colors=64)); durs.append(int(d * 1000 / FPS))
    gif[0].save(OUT / 'preview_actions.gif', save_all=True, append_images=gif[1:], duration=durs, loop=0, disposal=2)


if __name__ == '__main__':
    act = frames()
    tags = write_pxo(OUT / 'aoi.pxo', act)
    previews(act)
    print('wrote', OUT / 'aoi.pxo', [(t['name'], t['from'], t['to']) for t in tags])
