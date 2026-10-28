"""Verify immutable release inputs without writing or loading executable pickle."""
import argparse,hashlib,json,re,sys
from pathlib import Path
from .data import DEFAULT_ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=DEFAULT_ROOT);args=ap.parse_args();root=args.root
    manifest=json.loads((root/"MANIFEST.sha256.json").read_text());failures=[]
    for name,expected in manifest["files"].items():
        p=root/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:failures.append(name)
    print(json.dumps({"verified_files":len(manifest["files"]),"passed":not failures,"mismatches":failures},indent=2))
    if failures:raise SystemExit(1)
if __name__=="__main__":main()
