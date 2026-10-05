#!/usr/bin/env python3
"""Merge existing preload manifests with this machine's compiled SM80 QSA cache.

No CUDA calls, model loads, downloads or service modifications. Generated cubins
may contain compiler paths; keep this output local rather than publishing it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def digest(data): return hashlib.sha256(data).hexdigest()

def build(bases, cache, output):
    if output.exists():raise ValueError('Output must be a new directory')
    records = {}
    def add(name, shared, file, expected=None):
        data=file.read_bytes();h=digest(data)
        if expected is not None and expected!=h:raise ValueError('Preload checksum mismatch')
        key=(name,h,int(shared))
        records.setdefault(key,(file,{'name':name,'sha256':h,'shared':int(shared),'file':h+'.cubin'}))
    for base in bases:
        base=base.resolve(strict=True)
        for row in json.loads((base/'manifest.json').read_text(encoding='utf-8')):
            p=(base/row['file']).resolve(strict=True)
            if not p.is_relative_to(base):raise ValueError('Preload file escapes its directory')
            add(row['name'],row['shared'],p,row['sha256'])
    initial=len(records)
    cache=cache.resolve(strict=True)
    for p in sorted(cache.rglob('*.json')):
        try:meta=json.loads(p.read_text(encoding='utf-8'))
        except (ValueError,UnicodeError):continue
        if not isinstance(meta,dict):continue
        target=meta.get('target') or {}
        if not isinstance(target,dict):continue
        if target.get('backend')!='cuda' or str(target.get('arch'))!='80':continue
        name=meta.get('name','')
        binary=p.with_suffix('.cubin')
        if not name.startswith('_qsa_') or not binary.is_file():continue
        if not binary.resolve().is_relative_to(cache):raise ValueError('Cache file escapes its directory')
        add(name,meta['shared'],binary)
    if len(records)==initial:raise ValueError('No additional compiled SM80 QSA kernels found')
    output.mkdir(parents=True)
    rows=[]
    for source,row in records.values():
        dest=output/row['file']
        if not dest.exists():shutil.copyfile(source,dest)
        rows.append(row)
    (output/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    return {'existing':initial,'added':len(records)-initial,'total':len(records)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,action='append',required=True,help='Existing model/vision preload; repeatable')
    p.add_argument('--cache',type=Path,required=True,help='Locally compiled Triton cache')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.base,a.cache,a.output)))

if __name__=='__main__':main()
