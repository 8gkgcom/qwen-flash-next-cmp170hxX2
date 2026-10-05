#!/usr/bin/env python3
"""Preview/apply version-checked PP2 patches, or restore the exact backup.

Stop the model before applying or restoring. This script never starts services.
It deliberately refuses unknown source versions instead of fuzz-applying a patch.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
MODULES = ['m3_hc_runtime.py', 'm3_hc_kernels.py', 'm7_joint_runtime.py',
           'm7_joint_kernels.py', 'pp_draft_scatter.py']

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

def under(root, rel):
    p = root / rel
    if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unsafe target path: '+rel)
    return p

def git_apply(target, *args):
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(target.parent))
    subprocess.run(['git', '-c', 'core.autocrlf=false', '-c', 'core.eol=lf',
                    'apply', *map(str, args)], cwd=target, env=env, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True, help='Existing private runtime site directory')
    parser.add_argument('--backup', type=Path, help='New backup directory, outside target')
    parser.add_argument('--restore', type=Path, help='Backup made by this script')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    target = args.target.resolve(strict=True)
    if args.restore:
        backup = args.restore.resolve(strict=True)
        state = json.loads((backup/'backup.json').read_text(encoding='utf-8'))
        if str(target) != state['target']:
            raise SystemExit('Backup belongs to a different target')
        for rel, info in state['files'].items():
            p = under(target, rel)
            if digest(p) != info['after']:
                raise SystemExit('Target changed after install: '+rel)
            if info['before'] is not None and digest(under(backup/'files', rel)) != info['before']:
                raise SystemExit('Backup checksum mismatch: '+rel)
        print(json.dumps({'operation':'restore','files':len(state['files']),'apply':args.apply}))
        if args.apply:
            for rel, info in state['files'].items():
                p = under(target, rel)
                if info['before'] is None:
                    p.unlink()
                else:
                    shutil.copy2(under(backup/'files',rel), p)
        return

    manifest = json.loads((HERE/'base-sha256.json').read_text(encoding='utf-8'))
    changed, patches, copies = {}, [], []
    for rel, entry in manifest.items():
        old = digest(under(target, rel))
        if old == entry['after']:
            continue
        if old != entry['before']:
            raise SystemExit('Unsupported source version: '+rel)
        patch = HERE / entry['patch']
        git_apply(target, '--check', patch)
        patches.append(patch)
        changed[rel] = {'before':old,'after':entry['after']}
    for rel in MODULES:
        old, new = digest(under(target,rel)), digest(HERE/rel)
        if old == new:
            continue
        if old is not None:
            raise SystemExit('Different existing module; review manually: '+rel)
        copies.append(rel)
        changed[rel] = {'before':old,'after':new}
    print(json.dumps({'operation':'install','files':len(changed),'apply':args.apply}))
    if not args.apply or not changed:
        return
    backup = (args.backup or target.parent/('pp-mtp2-backup-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    if backup.exists() or backup.is_relative_to(target) or target.is_relative_to(backup):
        raise SystemExit('Backup must be a new directory outside the runtime')
    backup.mkdir(parents=True)
    for rel, info in changed.items():
        if info['before'] is not None:
            dest = under(backup/'files',rel)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(under(target,rel),dest)
    (backup/'backup.json').write_text(json.dumps({'target':str(target),'files':changed},indent=2),encoding='utf-8')
    try:
        for patch in patches:
            git_apply(target, patch)
        for rel in copies:
            shutil.copy2(HERE/rel,under(target,rel))
        for rel, info in changed.items():
            if digest(under(target,rel)) != info['after']:
                raise RuntimeError('Installed checksum mismatch: '+rel)
    except BaseException:
        for rel, info in changed.items():
            p = under(target,rel)
            if info['before'] is None:
                if p.is_file():p.unlink()
            else:
                shutil.copy2(under(backup/'files',rel),p)
        raise
    print('Installed; backup:',backup)
    print('Use the environment example, then restart through your existing launcher.')

if __name__ == '__main__':
    main()
