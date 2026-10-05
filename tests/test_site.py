"""The project site build (scripts/dev/build_site.py): every internal link and image of every page resolves.

Skipped when markdown-it-py is not installed (only the Pages workflow needs it).
"""
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("markdown_it"), "markdown-it-py not installed")
class SiteBuildTests(unittest.TestCase):
    def test_internal_links_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "site"
            subprocess.run([sys.executable, str(ROOT / "scripts" / "dev" / "build_site.py"), str(out)],
                           check=True, capture_output=True)
            broken = []
            for page in out.rglob("*.html"):
                text = page.read_text(encoding="utf-8")
                ids = set(re.findall(r'\bid="([^"]+)"', text))
                for url in re.findall(r'\b(?:href|src)="([^"]+)"', text):
                    if re.match(r"^([a-z]+:|//)", url):
                        continue
                    path, _, frag = url.partition("#")
                    if not path:
                        if frag and frag not in ids:
                            broken.append(f"{page.relative_to(out)}: #{frag}")
                        continue
                    target = (page.parent / unquote(path)).resolve()
                    if target.is_dir():
                        target = target / "index.html"
                    if not target.exists():
                        broken.append(f"{page.relative_to(out)}: {url}")
            self.assertEqual(broken, [])
            index = json.loads((out / "search.json").read_text(encoding="utf-8"))
            pages = {e["u"].split("#")[0] for e in index}
            self.assertIn("docs/guide.html", pages)
            self.assertIn("docs/FINDINGS.html", pages)


if __name__ == "__main__":
    unittest.main()
