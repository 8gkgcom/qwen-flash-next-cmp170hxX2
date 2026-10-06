"""Check the 3871 original source paths; generated build outputs are not distributed."""
from pathlib import Path
import hashlib
import json
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 verify-source.py SOURCE_DIRECTORY")
    source = Path(sys.argv[1]).resolve()
    manifest = json.loads(Path(__file__).with_name("source-sha256.json").read_text(encoding="utf-8"))
    failed = []
    for relative, expected in manifest.items():
        file = source / relative
        if not file.is_file():
            failed.append({"path": relative, "error": "missing"})
        elif hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            failed.append({"path": relative, "error": "SHA256 mismatch"})
    print(json.dumps({"files": len(manifest), "failed": failed, "passed": not failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
