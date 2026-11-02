#!/usr/bin/env python3
"""Cut the poses out of a character sheet (several poses on one plain background) and make a character.

    # 1. list the poses it finds (writes <out>/poses.png with a number on each)
    python3 tools/sheet_to_pxo.py sheet.png --out characters/mai

    # 2. say which poses make which action, then write the Pixelorama project and character.json
    python3 tools/sheet_to_pxo.py sheet.png --out characters/mai --name 舞 \\
        --map idle=0 attack=1,2 hurt=3 win=3,4,0 --lift 4=60 --windup 1 --height 72

The background is the colour found along the picture's border; everything that differs from it is kept.
Parts of a pose that are not touching (an open fan, a swirl of petals) are joined to the nearest pose.
Every pose is placed on one shared canvas with its feet on the same spot (the bottom of the pose, centred
on its lowest pixels); `--lift i=px` raises pose i (a jump). `--scale` shrinks the poses (default: so the
standing height is about 300 px, plenty for the game's high-resolution drawing).
The result is <out>/<id>.pxo (one animation tag per action, open it in Pixelorama to touch up) and
<out>/character.json pointing at it; build.py then includes the character.
"""
import argparse
import io
import json
import pathlib
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def find_poses(img, tol=34, join=5, min_frac=0.05):
    a = np.asarray(img.convert('RGBA')).astype(np.int32)
    h, w = a.shape[:2]
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])[:, :3]
    bg = np.median(border, axis=0)
    dist = np.abs(a[..., :3] - bg).sum(axis=2)
    fg = (dist > tol) & (a[..., 3] > 16)
    # soft alpha near the background colour keeps anti-aliased edges
    alpha = np.clip((dist - tol * 0.5) / (tol * 1.5), 0, 1) * (a[..., 3] / 255)
    m = Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(join | 1))
    grow = np.asarray(m) > 0
    lab = np.zeros((h, w), np.int32)
    comps = []
    for y0 in range(h):
        for x0 in range(w):
            if not grow[y0, x0] or lab[y0, x0]:
                continue
            lab[y0, x0] = len(comps) + 1
            stack, pts = [(y0, x0)], []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and grow[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = len(comps) + 1
                        stack.append((yy, xx))
            ys, xs = zip(*pts)
            comps.append({'id': len(comps) + 1, 'y0': min(ys), 'y1': max(ys), 'x0': min(xs), 'x1': max(xs), 'n': len(pts)})
    big = max(cp['n'] for cp in comps)
    poses = [cp for cp in comps if cp['n'] >= big * min_frac]
    # reading order: rows (by vertical overlap), then left to right
    poses.sort(key=lambda cp: (cp['y0'] + cp['y1']) / 2)
    rows = []
    for cp in poses:
        if rows and cp['y0'] < (rows[-1][0]['y0'] + rows[-1][0]['y1']) / 2:
            rows[-1].append(cp)
        else:
            rows.append([cp])
    ordered = [cp for r in rows for cp in sorted(r, key=lambda q: q['x0'])]
    out = []
    for cp in ordered:
        sel = (lab == cp['id']) & fg
        ys, xs = np.nonzero(sel)
        crop = np.zeros((cp['y1'] - cp['y0'] + 1, cp['x1'] - cp['x0'] + 1, 4), np.uint8)
        sub = (lab[cp['y0']:cp['y1'] + 1, cp['x0']:cp['x1'] + 1] == cp['id'])
        src = a[cp['y0']:cp['y1'] + 1, cp['x0']:cp['x1'] + 1]
        crop[..., :3] = src[..., :3]
        crop[..., 3] = (alpha[cp['y0']:cp['y1'] + 1, cp['x0']:cp['x1'] + 1] * 255 * sub).astype(np.uint8)
        low = ys >= ys.max() - max(4, (ys.max() - ys.min()) // 25)       # the lowest pixels: the feet
        out.append({'img': crop, 'box': (cp['x0'], cp['y0'], cp['x1'], cp['y1']),
                    'feet': (float(xs[low].mean()) - cp['x0'], ys.max() - cp['y0'])})
    return out


def contact_sheet(poses, path):
    S = 0.35
    tiles = [Image.fromarray(p['img']).resize((max(1, int(p['img'].shape[1] * S)), max(1, int(p['img'].shape[0] * S)))) for p in poses]
    W = sum(t.width + 16 for t in tiles) + 16
    H = max(t.height for t in tiles) + 40
    sheet = Image.new('RGBA', (W, H), (29, 40, 71, 255))
    d = ImageDraw.Draw(sheet)
    x = 16
    for i, t in enumerate(tiles):
        sheet.alpha_composite(t, (x, 30))
        d.text((x, 8), str(i), fill=(232, 196, 96, 255))
        x += t.width + 16
    sheet.save(path)


def write_pxo(path, frames, tags, fps=10):
    h, w = frames[0][0].shape[:2]
    data = {
        'pixelorama_version': 'v1.2.3-stable', 'pxo_version': 7, 'size_x': w, 'size_y': h, 'color_mode': 5,
        'tile_mode_x_basis_x': w, 'tile_mode_x_basis_y': 0, 'tile_mode_y_basis_x': 0, 'tile_mode_y_basis_y': h,
        'layers': [{'name': 'pose', 'visible': True, 'locked': False, 'blend_mode': 0, 'clipping_mask': False, 'opacity': 1.0,
                    'parent': -1, 'effects': [], 'type': 0, 'new_cels_linked': False, 'metadata': {}}],
        'tags': tags, 'guides': [], 'symmetry_points': [w / 2, h / 2],
        'frames': [{'cels': [{'opacity': 1.0, 'z_index': 0, 'image_size': f'Vector2i({w}, {h})', 'metadata': {}}], 'duration': d, 'metadata': {}}
                   for _, d in frames],
        'current_frame': 0, 'current_layer': 0, 'brushes': [], 'palettes': [], 'reference_images': [], 'tilesets': [],
        'vanishing_points': [], 'fps': float(fps), 'include_blended_images': False, 'user_data': '', 'metadata': {},
    }
    pv = io.BytesIO(); Image.fromarray(frames[0][0]).save(pv, 'PNG')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('data.json', json.dumps(data, ensure_ascii=False))
        z.writestr('mimetype', 'application/x-pixelorama')
        z.writestr('preview.png', pv.getvalue())
        for i, (img, _) in enumerate(frames):
            z.writestr(f'image_data/frames/{i + 1}/layer_1', img.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sheet')
    ap.add_argument('--out', required=True, help='character folder, e.g. characters/mai')
    ap.add_argument('--name', default='')
    ap.add_argument('--map', nargs='*', default=[], help='action=pose,pose… (idle walk attack hurt win)')
    ap.add_argument('--lift', nargs='*', default=[], help='pose=px  raise a pose (a jump)')
    ap.add_argument('--scale', type=float, default=0)
    ap.add_argument('--windup', type=int, default=0)
    ap.add_argument('--height', type=float, default=72, help='height in the game (world px, 36–80)')
    ap.add_argument('--faces', default='right')
    ap.add_argument('--dt', type=float, default=0.18, help='seconds per pose')
    ap.add_argument('--join', type=int, default=5, help='px: parts of a pose closer than this are joined')
    ap.add_argument('--tol', type=int, default=34, help='how far from the background colour counts as the picture')
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    poses = find_poses(Image.open(args.sheet), tol=args.tol, join=args.join)
    contact_sheet(poses, out / 'poses.png')
    print(f'{len(poses)} poses → {out / "poses.png"}')
    for i, p in enumerate(poses):
        print(f'  {i}: box {p["box"]}  feet {tuple(round(v) for v in p["feet"])}')
    if not args.map:
        return
    acts = {}
    for m in args.map:
        k, v = m.split('=')
        acts[k] = [int(i) for i in v.split(',')]
    lift = {int(k): int(v) for k, v in (m.split('=') for m in args.lift)}
    used = sorted({i for v in acts.values() for i in v})
    stand = max(poses[i]['img'].shape[0] for i in acts.get('idle', used[:1]))
    scale = args.scale or min(1.0, 300 / stand)
    sized = {}
    for i in used:
        im = Image.fromarray(poses[i]['img'])
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
        fx, fy = poses[i]['feet']
        sized[i] = (np.asarray(im), fx * scale, fy * scale + lift.get(i, 0) * scale)
    left = max(fx for _, fx, _ in sized.values()); right = max(im.shape[1] - fx for im, fx, _ in sized.values())
    up = max(fy for _, _, fy in sized.values())
    W, H = int(np.ceil(left + right)) + 4, int(np.ceil(up)) + 2
    ax, ay = int(np.ceil(left)) + 2, H
    canvases = {}
    for i, (im, fx, fy) in sized.items():
        cv = Image.new('RGBA', (W, H))
        cv.alpha_composite(Image.fromarray(im), (int(round(ax - fx)), int(round(ay - fy))))
        canvases[i] = np.asarray(cv).copy()
    frames, tags = [], []
    colors = {'idle': 'c9a64aff', 'walk': '8fc9b8ff', 'attack': '5b86c9ff', 'hurt': 'd0533aff', 'win': 'e58fa6ff'}
    fps = 10
    for k in ('idle', 'walk', 'attack', 'hurt', 'win'):
        if k not in acts:
            continue
        tags.append({'name': k, 'color': colors[k], 'from': len(frames) + 1, 'to': len(frames) + len(acts[k])})
        for j, i in enumerate(acts[k]):
            hold = 30 if k == 'win' and j == len(acts[k]) - 1 else 1
            frames.append((canvases[i], round(args.dt * fps * hold, 2)))
    cid = out.name
    write_pxo(out / f'{cid}.pxo', frames, tags, fps)
    Image.fromarray(canvases[acts.get('idle', used)[0]]).save(out / 'sprite.png')
    meta = {'name': args.name or cid, 'desc': '', 'order': 50, 'sprite': 'sprite.png', 'pxo': f'{cid}.pxo',
            'feet': [ax, ay], 'body': round(max(poses[i]['img'].shape[0] for i in acts.get('idle', used[:1])) * scale),
            'height': args.height, 'faces': args.faces, 'pixelated': False}
    if 'walk' not in acts:
        meta['walk'] = {'tag': 'idle'}       # a single standing pose walks by the game's own motion
    if args.windup:
        meta['attack'] = {'tag': 'attack', 'windup': args.windup}
    (out / 'character.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {out / (cid + ".pxo")} ({W}×{H}, {len(frames)} frames) and character.json')


if __name__ == '__main__':
    main()
