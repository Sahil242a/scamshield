"""CLI:  python -m scamshield "message text" [--sender VM-HDFCBK] [--json]"""
import argparse
import json

from .detector import ScamDetector, format_verdict

ap = argparse.ArgumentParser(prog="scamshield")
ap.add_argument("text")
ap.add_argument("--sender", default=None)
ap.add_argument("--json", action="store_true")
a = ap.parse_args()
v = ScamDetector().analyze(a.text, a.sender)
print(json.dumps(v.to_dict(), indent=2, ensure_ascii=False) if a.json else format_verdict(v))
