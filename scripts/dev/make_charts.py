#!/usr/bin/env python3
"""Draw the SVG charts of docs/img/ from benchmarks/charts.json (stdlib only).

Run after adding a measurement to benchmarks/charts.json:  python3 scripts/dev/make_charts.py
"""
import json
import sys
import textwrap
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#d64545", "#3b7dd8"]
MUTED = "#8c959f"
STYLE = (
    "<style>.bg{fill:#ffffff}text{font-family:-apple-system,Helvetica,Arial,sans-serif;fill:#24292f}"
    ".t{font-size:15px;font-weight:600}.n{font-size:11px;fill:#57606a}.l{font-size:12px}.v{font-size:12px;font-weight:600}"
    ".g{stroke:#d0d7de}"
    "@media (prefers-color-scheme:dark){.bg{fill:#0d1117}text{fill:#e6edf3}.n{fill:#8d96a0}.g{stroke:#30363d}}</style>"
)


def render(chart):
    series, bars = chart["series"], chart["bars"]
    ref = chart.get("reference")
    top = max([v for b in bars for v in b["values"]] + ([ref["value"]] if ref else []))
    label_w, plot_w, row_h = 230, 470, 18 * len(series) + 10
    width = label_w + plot_w + 110
    y0 = 58 + (18 if len(series) > 1 else 0)
    note = textwrap.wrap(f'{chart["unit"]}. {chart["note"]}', 125)
    height = y0 + row_h * len(bars) + 34 + 15 * len(note)
    scale = plot_w / top
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           STYLE, f'<rect class="bg" width="{width}" height="{height}"/>', f'<text class="t" x="10" y="24">{escape(chart["title"])}</text>']
    if len(series) > 1:
        for i, name in enumerate(series):
            x = label_w + 110 * i
            out.append(f'<rect x="{x}" y="38" width="12" height="12" fill="{COLORS[i]}"/>'
                       f'<text class="l" x="{x + 17}" y="48">{escape(name)}</text>')
    for r, bar in enumerate(bars):
        y = y0 + r * row_h
        out.append(f'<text class="l" x="{label_w - 8}" y="{y + 9 * len(series) + 4}" text-anchor="end">{escape(bar["label"])}</text>')
        for i, v in enumerate(bar["values"]):
            by, w = y + 18 * i, max(v * scale, 1)
            text = bar["text"] if i == 0 and "text" in bar else f"{v:g}"
            color = MUTED if bar.get("muted") else COLORS[i]
            out.append(f'<rect x="{label_w}" y="{by}" width="{w:.1f}" height="15" rx="2" fill="{color}"/>'
                       f'<text class="v" x="{label_w + w + 5:.1f}" y="{by + 12}">{escape(text)}</text>')
    bottom = y0 + row_h * len(bars)
    if ref:
        x = label_w + ref["value"] * scale
        out.append(f'<line class="g" x1="{x:.1f}" y1="{y0 - 6}" x2="{x:.1f}" y2="{bottom}" stroke-width="2" stroke-dasharray="5 4"/>'
                   f'<text class="n" x="{x:.1f}" y="{bottom + 14}" text-anchor="middle">{escape(ref["label"])}: {ref["value"]:g}</text>')
    for i, line in enumerate(note):
        out.append(f'<text class="n" x="10" y="{height - 10 - 15 * (len(note) - 1 - i)}">{escape(line)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(root=ROOT, out_dir=None):
    data = json.loads((root / "benchmarks" / "charts.json").read_text())
    out_dir = out_dir or root / "docs" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    for chart in data["charts"]:
        for src in chart["sources"]:
            if not (root / src).exists():
                sys.exit(f"{chart['id']}: source {src} does not exist")
        for bar in chart["bars"]:
            if len(bar["values"]) != len(chart["series"]):
                sys.exit(f"{chart['id']}: bar {bar['label']!r} needs one value per series")
        (out_dir / f"{chart['id']}.svg").write_text(render(chart))
        print(f"docs/img/{chart['id']}.svg")


if __name__ == "__main__":
    main()
