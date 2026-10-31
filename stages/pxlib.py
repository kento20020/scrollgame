"""Tiny palette-indexed pixel canvas used by the stage generators.

The art is drawn on the 128x96 "art grid" the game uses and saved at 2x nearest
(256x192), so the game's BOX downscale gives back exactly the pixels drawn here.
"""
import numpy as np
from PIL import Image

import check as chk


class Canvas:
    def __init__(self, palette, W=128, H=96):
        self.W, self.H = W, H
        self.names = list(palette.keys())
        self.rgb = np.array([palette[n] for n in self.names], dtype=np.uint8)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.px = np.zeros((H, W), dtype=np.int16)

    # ---- basic drawing -------------------------------------------------
    def c(self, name):
        return self.idx[name] if isinstance(name, str) else int(name)

    def inb(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H

    def set(self, x, y, col):
        if 0 <= x < self.W and 0 <= y < self.H:
            self.px[y, x] = self.c(col)

    def get(self, x, y):
        return self.names[self.px[y, x]] if self.inb(x, y) else None

    def rect(self, x0, y0, x1, y1, col):
        """Fill [x0,x1) x [y0,y1)."""
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.W, x1), min(self.H, y1)
        if x1 > x0 and y1 > y0:
            self.px[y0:y1, x0:x1] = self.c(col)

    def hline(self, x0, x1, y, col):
        self.rect(x0, y, x1, y + 1, col)

    def vline(self, x, y0, y1, col):
        self.rect(x, y0, x + 1, y1, col)

    def line(self, x0, y0, x1, y1, col):
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(n + 1):
            self.set(round(x0 + (x1 - x0) * i / n), round(y0 + (y1 - y0) * i / n), col)

    def disc(self, cx, cy, r, col, mask=None):
        for y in range(int(cy - r - 1), int(cy + r + 2)):
            for x in range(int(cx - r - 1), int(cx + r + 2)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r and self.inb(x, y):
                    if mask is None or mask[y, x]:
                        self.px[y, x] = self.c(col)

    def ellipse_mask(self, cx, cy, rx, ry):
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        return ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1.0

    # ---- helpers on index arrays --------------------------------------
    def remap(self, mask, mapping):
        """Where mask: replace colours by mapping {name: name}."""
        m = {self.c(a): self.c(b) for a, b in mapping.items()}
        sub = self.px[mask]
        out = np.array([m.get(int(v), int(v)) for v in sub], dtype=np.int16) if sub.size else sub
        self.px[mask] = out

    def isin(self, names):
        ids = [self.c(n) for n in names]
        return np.isin(self.px, ids)

    # ---- output --------------------------------------------------------
    def to_rgb(self):
        return self.rgb[self.px]

    def save(self, path, scale=2):
        im = Image.fromarray(self.to_rgb())
        im = im.resize((self.W * scale, self.H * scale), Image.NEAREST)
        im.save(path)
        return im

    def unique_colors(self):
        return len(np.unique(self.px))

    # ---- classification on the canvas itself --------------------------
    def classify(self):
        rgb = self.to_rgb().astype(np.float64) / 255.0
        ch, cw = self.H // 2, self.W // 2
        cell = rgb[: ch * 2, : cw * 2].reshape(ch, 2, cw, 2, 3).mean(axis=(1, 3))
        h, s, l = chk.rgb_to_hsl(cell)
        return h, s, l

    def fix_walls(self, wall_cells, dark_names, max_iter=12):
        """For cells intended as wall whose average lightness is too high,
        replace their brightest pixel with the darkest allowed colour."""
        darkest = min(dark_names, key=lambda n: self.rgb[self.c(n)].astype(int).sum())
        n_fixed = 0
        for _ in range(max_iter):
            _, _, l = self.classify()
            bad = wall_cells & (l >= 0.165)
            if not bad.any():
                break
            for cy, cx in zip(*np.nonzero(bad)):
                blk = self.px[cy * 2:cy * 2 + 2, cx * 2:cx * 2 + 2]
                lum = self.rgb[blk].astype(int).sum(-1)
                j = np.unravel_index(np.argmax(lum), lum.shape)
                blk[j] = self.c(darkest)
                n_fixed += 1
        return n_fixed

    def fix_water(self, water_cells, water_fill, max_iter=6):
        n_fixed = 0
        for _ in range(max_iter):
            h, s, l = self.classify()
            ok = (h >= 170) & (h <= 255) & (s >= 0.22) & (l >= 0.17)
            bad = water_cells & ~ok
            if not bad.any():
                break
            for cy, cx in zip(*np.nonzero(bad)):
                blk = self.px[cy * 2:cy * 2 + 2, cx * 2:cx * 2 + 2]
                blk[:, :] = self.c(water_fill)
                n_fixed += 1
        return n_fixed


# ---- noise / patterns -------------------------------------------------
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0


def bayer(x, y):
    return BAYER4[y % 4, x % 4]


def value_noise(rng, H, W, cell=6):
    """Smooth-ish value noise in 0..1 (bilinear)."""
    gh, gw = H // cell + 3, W // cell + 3
    g = rng.random((gh, gw))
    ys = np.arange(H) / cell
    xs = np.arange(W) / cell
    y0 = ys.astype(int)
    x0 = xs.astype(int)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def cell_mask_to_px(cm):
    return np.kron(cm.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)


def dist_to_mask(mask, maxd=6):
    """Chebyshev-ish distance (4-neighbour steps) from each pixel to nearest True, capped."""
    H, W = mask.shape
    d = np.full((H, W), maxd + 1, dtype=np.int16)
    d[mask] = 0
    cur = mask.copy()
    for k in range(1, maxd + 1):
        nxt = cur.copy()
        nxt[1:] |= cur[:-1]
        nxt[:-1] |= cur[1:]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        new = nxt & ~cur
        d[new] = k
        cur = nxt
    return d
