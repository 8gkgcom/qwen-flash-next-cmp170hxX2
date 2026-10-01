#!/usr/bin/env python3
"""Validate a pinned vLLM base, then optionally overlay captured source and RAM gather."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, required=True, help='Pinned base dist-packages directory')
    p.add_argument('--apply', action='store_true', help='Write files; default only validates and previews')
    p.add_argument('--compiler', default='g++')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.target.resolve()
    overlay = root / 'runtime/overlay'
    if not (target / 'vllm').is_dir() or not list((target / 'vllm').glob('_C*.so')):
        raise SystemExit('Missing base vLLM native extensions. Extract the pinned OCI base first.')
    expected = json.loads((root / 'runtime/base-sha256.json').read_text())
    errors = []
    for rel, old_hash in expected.items():
        existing, desired = target / rel, overlay / rel
        # Reapplying an identical published overlay is also permitted.
        allowed = {old_hash}
        if desired.is_file():
            allowed.add(digest(desired))
        observed = digest(existing) if existing.is_file() else None
        if observed not in allowed:
            errors.append(rel)
    if errors:
        raise SystemExit('Base does not match the pinned source or this overlay:\n' + '\n'.join(errors))
    files = {f.relative_to(overlay): f for f in overlay.rglob('*') if f.is_file()}
    changed = {rel: src for rel, src in files.items()
               if not (target / rel).is_file() or digest(target / rel) != digest(src)}
    print(json.dumps({'target': str(target), 'source_files': len(files), 'changed_files': len(changed),
                      'compile': 'ple_ram_gather.so', 'write_enabled': args.apply}, indent=2))
    if not args.apply:
        return
    # Compile first; a compiler failure leaves installed source untouched.
    with tempfile.TemporaryDirectory(prefix='ple-gather-build-') as temporary:
        library = Path(temporary) / 'ple_ram_gather.so'
        subprocess.run([args.compiler, '-O3', '-std=c++17', '-fPIC', '-shared',
                        str(overlay / 'ple_ram_gather.cpp'), '-o', str(library)], check=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = target.parent / ('sm80-source-backup-' + stamp)
        backup.mkdir()
        for rel in [*changed, Path('ple_ram_gather.so')]:
            old = target / rel
            if old.is_file():
                dest = backup / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, dest)
        (backup / 'CHANGED_FILES.json').write_text(json.dumps([str(r) for r in changed], indent=2))
        for rel, src in changed.items():
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        shutil.copy2(library, target / 'ple_ram_gather.so')
    print('Overlay installed. Backup:', backup)
    print('No model or service was started. Validate the Python/CUDA ABI before launching.')


if __name__ == '__main__':
    main()
