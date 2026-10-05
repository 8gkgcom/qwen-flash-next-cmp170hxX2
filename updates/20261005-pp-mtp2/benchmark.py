"""Fixed, sequential decode benchmark; uses actual API usage and engine counters."""
import argparse
import os
import concurrent.futures
import json
from pathlib import Path
import re
import time
import urllib.request

ROOT = Path(os.environ.get('BENCH_OUTPUT', './benchmark-output'))
URL = os.environ.get('BENCH_URL', 'http://127.0.0.1:8000').rstrip('/')
PROMPTS = {
    'code': '''Design and implement a complete Python in-memory transactional key-value store.
Include nested transactions, rollback, expiry, prefix iteration and thread safety.
Explain the invariants and then provide all implementation code and comprehensive
unittest tests. Think carefully about failure cases and concurrency. Be detailed.''',
    'reason': '''Solve carefully and justify every step: A fair six-sided die is rolled repeatedly.
Find the expected number of rolls until the consecutive pattern 1,2,1,2,1 first appears.
Then derive a general algorithm for any target pattern, implement it in exact rational
Python arithmetic, and explain how to validate the result independently.''',
}


def metrics():
    text = urllib.request.urlopen(URL+'/metrics', timeout=10).read().decode()
    vals = {}
    for line in text.splitlines():
        m = re.match(r'(vllm:[\w]+)(?:\{[^}]*\})?\s+([\d.eE+-]+)', line)
        if m:
            k,v=m.groups()
            vals[k]=vals.get(k,0)+float(v)
    return vals


def generate(prompt, limit, seed):
    payload = {'model':'qwen','messages':[{'role':'user','content':prompt}],
        'stream':True,'stream_options':{'include_usage':True},'max_tokens':limit,
        'seed':seed,'temperature':1.0,'top_p':0.95,'top_k':20,'presence_penalty':0.5,
        'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'xhigh'}}
    req=urllib.request.Request(URL+'/v1/chat/completions',data=json.dumps(payload).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ.get('BENCH_API_KEY','')})
    start=time.monotonic();first=last=None;usage=None;finish=None;parts=[]
    with urllib.request.urlopen(req,timeout=600) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]':continue
            obj=json.loads(line[6:])
            if obj.get('error'):raise RuntimeError(obj['error'])
            if obj.get('usage'):usage=obj['usage']
            for choice in obj.get('choices',[]):
                delta=choice.get('delta') or {}
                if any(delta.get(k) for k in ('content','reasoning','reasoning_content')):
                    last=time.monotonic()
                    if first is None:first=last
                    parts.append(delta)
                if choice.get('finish_reason'):finish=choice['finish_reason']
    elapsed=time.monotonic()-start
    if not usage or not first or last<=first:raise RuntimeError('Missing output/usage')
    return {'usage':usage,'finish':finish,'ttft':first-start,'seconds':elapsed,
        'decode_seconds':last-first,'decode_tps':(usage['completion_tokens']-1)/(last-first),
        'output':parts}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('label');ap.add_argument('--tokens',type=int,default=1536)
    ap.add_argument('--repeat',type=int,default=1);ap.add_argument('--concurrency',type=int,default=1)
    args=ap.parse_args();ROOT.mkdir(exist_ok=True,parents=True);rows=[]
    for name,prompt in [('warmup',PROMPTS['code'])]+list(PROMPTS.items())*args.repeat:
        before=metrics()
        if before.get('vllm:num_requests_running',0) or before.get('vllm:num_requests_waiting',0):
            raise RuntimeError('Unrelated requests are active; benchmark stopped')
        n=1 if name=='warmup' else args.concurrency
        started=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futures=[pool.submit(generate,prompt,128 if name=='warmup' else args.tokens,123+i) for i in range(n)]
            responses=[f.result() for f in futures]
        elapsed=time.monotonic()-started;after=metrics()
        keys=['vllm:generation_tokens_total','vllm:spec_decode_num_drafts_total',
            'vllm:spec_decode_num_accepted_tokens_total','vllm:num_preemptions_total']
        delta={k:after.get(k,0)-before.get(k,0) for k in keys}
        row={'name':name,'concurrency':n,'responses':responses,'wall_seconds':elapsed,'delta':delta,
            'aggregate_tps':sum(x['usage']['completion_tokens'] for x in responses)/elapsed}
        drafts=delta['vllm:spec_decode_num_drafts_total']
        row['accepted_per_round']=delta['vllm:spec_decode_num_accepted_tokens_total']/drafts if drafts else None
        row['estimated_round_ms']=sum(x['decode_seconds'] for x in responses)/drafts*1000 if drafts else None
        rows.append(row)
        (ROOT/(args.label+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='responses'}|{'responses':[
            {k:v for k,v in x.items() if k!='output'} for x in responses]},ensure_ascii=False),flush=True)
        time.sleep(.5)
    print('BENCHMARK_COMPLETE',args.label,flush=True)


if __name__=='__main__':main()
