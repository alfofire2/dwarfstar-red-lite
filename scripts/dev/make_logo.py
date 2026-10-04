#!/usr/bin/env python3
"""Draws docs/img/logo.svg: a red dwarf whose corona has one ray per Qwen3-Next layer.

48 rays clockwise from the top, layer 0 first. The 12 full-attention layers (3, 7, ..., 47) are long and red, the
36 Gated DeltaNet layers short and slate. The disc has limb darkening, as a real star does. One file for GitHub's
light and dark themes: every color reads on both backgrounds.
"""
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "img" / "logo.svg"
C, R_STAR = 120.0, 50.0
rays = []
for layer in range(48):
    a = math.radians(layer * 7.5 - 90 + 3.75)
    attention = layer % 4 == 3
    r0, r1 = (62.0, 96.0) if attention else (66.0, 82.0)
    x0, y0 = C + r0 * math.cos(a), C + r0 * math.sin(a)
    x1, y1 = C + r1 * math.cos(a), C + r1 * math.sin(a)
    cls = "a" if attention else "d"
    rays.append(f'<line class="{cls}" x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}"/>')

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240" width="240" height="240" role="img"
     aria-label="DwarfStar Red Lite: a red dwarf star with 48 rays, one per model layer">
  <defs>
    <radialGradient id="limb" cx="0.46" cy="0.44" r="0.58">
      <stop offset="0" stop-color="#FFB070"/>
      <stop offset="0.55" stop-color="#F2542D"/>
      <stop offset="1" stop-color="#9E1F1A"/>
    </radialGradient>
  </defs>
  <style>
    line {{ stroke-linecap: round; }}
    .d {{ stroke: #8C88A3; stroke-width: 3.2; }}
    .a {{ stroke: #F2542D; stroke-width: 4.4; }}
  </style>
  {chr(10).join("  " + r for r in rays).strip()}
  <circle cx="{C}" cy="{C}" r="{R_STAR}" fill="url(#limb)"/>
</svg>
'''
OUT.write_text(svg)
print(OUT)
