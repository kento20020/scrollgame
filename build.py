#!/usr/bin/env python3
"""Inline the image assets as base64 data URIs and write index.html."""
import base64
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "game.src.html"
OUT = ROOT / "index.html"

ASSETS = {
    "{{AOI_SPRITE}}": ("assets/aoi_sprite.png", "image/png"),
    "{{AOI_PORTRAIT}}": ("assets/aoi_portrait.png", "image/png"),
    "{{WARDROBE}}": ("assets/sample_wardrobe.jpg", "image/jpeg"),
    "{{AOI_WALK}}": ("assets/aoi_walk.png", "image/png"),
    "{{AOI_ATTACK}}": ("assets/aoi_attack.png", "image/png"),
    "{{SHU_STREET}}": ("stages/shu_street.png", "image/png"),
    "{{BAMBOO_PATH}}": ("stages/bamboo_path.png", "image/png"),
}

# Text files inserted verbatim (inline <script> bodies).
TEXTS = {
    "{{SFX_JS}}": "parts/sfx.js",
}


def data_uri(path: pathlib.Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    html = SRC.read_text(encoding="utf-8")
    for key, rel in TEXTS.items():
        if key not in html:
            raise SystemExit(f"placeholder {key} missing from source")
        text = (ROOT / rel).read_text(encoding="utf-8")
        if "</script" in text.lower():
            raise SystemExit(f"{rel} contains a closing script tag")
        html = html.replace(key, text)
    for key, (rel, mime) in ASSETS.items():
        if key not in html:
            raise SystemExit(f"placeholder {key} missing from source")
        html = html.replace(key, data_uri(ROOT / rel, mime))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
