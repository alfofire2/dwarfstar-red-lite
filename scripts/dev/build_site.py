#!/usr/bin/env python3
"""Build the project site: site/ plus the repository's Markdown as /docs/ pages and a search index (Pages workflow).

    pip install markdown-it-py==3.0.0
    python3 scripts/dev/build_site.py _site

The Markdown files stay the only source: README.md becomes docs/guide.html, CHANGELOG.md docs/changelog.html,
docs/README.md docs/index.html and docs/X.md docs/X.html. Links between them are rewritten to the pages, links to
other repository files go to GitHub, and every h2/h3 section is one entry of search.json (site/search.js).
"""
from __future__ import annotations

import datetime
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
SITE = "https://redlite.alfonsodaniello.it/"
REPO = "https://github.com/alfofire2/dwarfstar-red-lite"

FIRST = ["README.md", "docs/FINDINGS.md", "docs/WHAT_DID_NOT_WORK.md", "docs/ROADMAP.md", "CHANGELOG.md",
         "docs/README.md"]
TITLES = {"README.md": "Guide", "CHANGELOG.md": "Changelog", "docs/README.md": "Documentation index"}


def page_of(src: str) -> str:
    return {"README.md": "guide.html", "CHANGELOG.md": "changelog.html",
            "docs/README.md": "index.html"}.get(src, PurePosixPath(src).stem + ".html")


def slug(text: str, seen: dict[str, int]) -> str:
    """GitHub's heading anchors, so README links like #install keep working."""
    s = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
    n = seen.get(s, 0)
    seen[s] = n + 1
    return s if n == 0 else f"{s}-{n}"


def plain(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def label(title: str) -> str:
    return re.sub(r"^(DwarfStar )?Red Lite\s*", "", title).strip(" —-:") or title


def group(src: str) -> str:
    if src in FIRST:
        return "Start here"
    return "Milestones" if re.match(r"docs/REDLITE_DEV\d", src) else "Reference"


def dev_number(src: str) -> tuple[int, str]:
    m = re.match(r"docs/REDLITE_DEV(\d+)", src)
    return (-int(m.group(1)) if m else 0, src)


def updated(src: str) -> str:
    try:
        return subprocess.run(["git", "log", "-1", "--format=%cs", "--", src], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def rewrite(body: str, src: str, pages: dict[str, str]) -> str:
    base = PurePosixPath(src).parent

    def fix(m: re.Match) -> str:
        attr, url = m.group(1), m.group(2)
        if re.match(r"^([a-z]+:|#|/)", url):
            return m.group(0)
        path, _, frag = url.partition("#")
        target = PurePosixPath(*[p for p in (base / path).parts])
        parts: list[str] = []
        for p in target.parts:             # normalize ../ without touching the file system
            if p == "..":
                if parts:
                    parts.pop()
            elif p != ".":
                parts.append(p)
        norm = "/".join(parts)
        frag = "#" + frag if frag else ""
        if norm in pages:
            new = pages[norm] + frag
        elif norm.startswith("docs/img/"):
            new = "../img/" + norm[len("docs/img/"):]
        elif (ROOT / norm).exists():
            new = f"{REPO}/{'tree' if (ROOT / norm).is_dir() else 'blob'}/main/{norm}{frag}"
        else:
            return m.group(0)
        return f'{attr}="{html.escape(new, quote=True)}"'

    return re.sub(r'\b(href|src)="([^"]*)"', fix, body)


def render(md, src: str, pages: dict[str, str]) -> tuple[str, str, list[dict], list[tuple[str, str]]]:
    body = md.render((ROOT / src).read_text(encoding="utf-8"))
    body = rewrite(body, src, pages)
    body = body.replace("<table>", '<div class="table-scroll"><table>').replace("</table>", "</table></div>")
    seen: dict[str, int] = {}
    title = TITLES.get(src, "")
    heads: list[tuple[int, str, str, int]] = []

    def add_id(m: re.Match) -> str:
        nonlocal title
        level, inner = int(m.group(1)), m.group(3)
        text = plain(inner)
        if level == 1 and not title:
            title = text
        anchor = slug(text, seen)
        heads.append((level, anchor, text, m.start()))
        link = f'<a class="anchor" href="#{anchor}" aria-label="Link to this section">#</a>' if level > 1 else ""
        return f'<h{level} id="{anchor}"{m.group(2)}>{inner}{link}</h{level}>'

    body = re.sub(r"<h([1-4])([^>]*)>(.*?)</h\1>", add_id, body, flags=re.S)
    title = title or PurePosixPath(src).stem
    # search: one entry per h2/h3 section (the text before the first h2 belongs to the page itself)
    url = "docs/" + pages[src]
    cuts = [(m.start(), m.group(1)) for m in re.finditer(r'<h[23] id="([^"]+)"', body)]
    sections, last, anchor = [], 0, ""
    for pos, nxt in cuts + [(len(body), None)]:
        chunk = re.sub(r'<a class="anchor"[^>]*>#</a>', "", body[last:pos])
        text = plain(re.sub(r"^<h[1-4][^>]*>.*?</h[1-4]>", "", chunk, count=1, flags=re.S))
        h = next((t for _, a, t, _ in heads if a == anchor), title) if anchor else title
        if text:
            sections.append({"t": label(title), "h": h, "u": url + (f"#{anchor}" if anchor else ""), "x": text[:6000]})
        last, anchor = pos, nxt or ""
    toc = [(a, t) for lvl, a, t, _ in heads if lvl == 2]
    return title, body, sections, toc


def main() -> int:
    from markdown_it import MarkdownIt

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "_site").resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "docs").mkdir(parents=True)
    (out / "img").mkdir()
    for f in (ROOT / "site").iterdir():
        if f.is_file() and f.name != "doc.html":
            shutil.copy2(f, out / f.name)
    for f in (ROOT / "docs" / "img").iterdir():
        shutil.copy2(f, out / "img" / f.name)

    sources = ["README.md", "CHANGELOG.md"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").glob("*.md"))
    pages = {s: page_of(s) for s in sources}
    md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    built = {s: render(md, s, pages) for s in sources}

    order = {"Start here": 0, "Reference": 1, "Milestones": 2}
    def key(s: str):
        return (order[group(s)], FIRST.index(s) if s in FIRST else 0, dev_number(s), label(built[s][0]).lower())
    nav_by_group: dict[str, list[str]] = {}
    for s in sorted(sources, key=key):
        nav_by_group.setdefault(group(s), []).append(s)

    template = (ROOT / "site" / "doc.html").read_text(encoding="utf-8")
    index, urls = [], [SITE]
    for s in sources:
        title, body, sections, toc = built[s]
        current = ' aria-current="page"'
        side = "".join(
            f'<details class="grp" {"open" if g != "Milestones" or s in items else ""}><summary>{g}</summary><ul>'
            + "".join(f'<li><a href="{pages[i]}"{current if i == s else ""}>'
                      f'{html.escape(label(built[i][0]))}</a></li>' for i in items)
            + "</ul></details>" for g, items in nav_by_group.items())
        toc_html = "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in toc[:30])
        first_p = re.search(r"<p>(.*?)</p>", body, re.S)
        desc = plain(first_p.group(1))[:155] if first_p else title
        date = updated(s)
        page = (template.replace("{{title}}", html.escape(title)).replace("{{description}}", html.escape(desc, quote=True))
                .replace("{{canonical}}", f"{SITE}docs/{pages[s]}").replace("{{group}}", group(s).upper())
                .replace("{{side}}", side).replace("{{toc}}", toc_html)
                .replace("{{source}}", f"{REPO}/blob/main/{s}").replace("{{source_name}}", s)
                .replace("{{updated}}", f" · updated {date}" if date else "")
                .replace("{{built}}", datetime.date.today().isoformat()).replace("{{body}}", body))
        (out / "docs" / pages[s]).write_text(page, encoding="utf-8")
        index.extend(sections)
        urls.append(f"{SITE}docs/{pages[s]}")
    (out / "search.json").write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <url><loc>{u}</loc></url>\n" for u in urls) + "</urlset>\n", encoding="utf-8")
    print(f"{len(sources)} pages, {len(index)} search sections -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
