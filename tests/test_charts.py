import importlib.util
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_charts", ROOT / "scripts" / "dev" / "make_charts.py")
make_charts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_charts)


class ChartTests(unittest.TestCase):
    def test_every_chart_renders_and_is_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_charts.main(ROOT, Path(tmp))
            for svg in Path(tmp).glob("*.svg"):
                ET.fromstring(svg.read_text())
                committed = ROOT / "docs" / "img" / svg.name
                self.assertEqual(svg.read_text(), committed.read_text(),
                                 f"{svg.name} is stale: run python3 scripts/dev/make_charts.py")


if __name__ == "__main__":
    unittest.main()
