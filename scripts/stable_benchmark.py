"""Fixed-length code/agent benchmark with repeated prompts and SSE accounting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('--repeats', type=int, default=2)
p.add_argument('--temperature', type=float, default=0)
args = p.parse_args()
root = Path(__file__).resolve().parent
base = os.environ.get('MODEL_BASE_URL', 'http://127.0.0.1:8000').rstrip('/')
api_key = os.environ.get('MODEL_API_KEY')
if not api_key:
    raise SystemExit('Set MODEL_API_KEY before benchmarking.')

def metrics():
    request = urllib.request.Request(base + '/metrics', headers={'Authorization': 'Bearer ' + api_key})
    s = urllib.request.urlopen(request, timeout=5).read().decode()
    result = {}
    for line in s.splitlines():
        if line.startswith(('vllm:spec_decode_', 'vllm:num_preemptions_total',
                            'vllm:num_requests_running', 'vllm:num_requests_waiting')) and '_bucket{' not in line:
            key, value = line.rsplit(' ', 1)
            result[key] = float(value)
    return result

def generate(prompt, label, thinking, count=2048):
    body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': args.temperature, 'seed': 0, 'max_tokens': count, 'min_tokens': count,
            'stream': True, 'stream_options': {'include_usage': True},
            'cache_salt': 'pp-draft-' + args.name + '-' + label,
            'reasoning_effort': 'xhigh' if thinking else 'none',
            'chat_template_kwargs': {'enable_thinking': thinking}}
    req = urllib.request.Request(base + '/v1/chat/completions',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key})
    before = metrics()
    start = time.perf_counter()
    first = last = None
    content, reasoning, usage, finish = [], [], None, None
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            if not raw.startswith(b'data: '): continue
            line = raw[6:].strip()
            if line == b'[DONE]': break
            data = json.loads(line)
            if data.get('error'): raise RuntimeError(data['error'])
            usage = data.get('usage') or usage
            for choice in data.get('choices', []):
                delta = choice.get('delta', {})
                text = delta.get('content') or ''
                thought = delta.get('reasoning') or delta.get('reasoning_content') or ''
                if text or thought:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    content.append(text); reasoning.append(thought)
                finish = choice.get('finish_reason') or finish
    elapsed = time.perf_counter() - start
    after = metrics()
    assert usage and usage['completion_tokens'] == count, (usage, count)
    assert first and last and last > first
    text, thought = ''.join(content), ''.join(reasoning)
    (root / (args.name + '.' + label + '.output.json')).write_text(
        json.dumps({'content': text, 'reasoning': thought}, ensure_ascii=False))
    delta = {k: after[k] - before.get(k, 0) for k in after}
    def sum_delta(prefix): return sum(v for k, v in delta.items() if k.startswith(prefix+'{'))
    drafts = sum_delta('vllm:spec_decode_num_drafts_total')
    accepted = sum_delta('vllm:spec_decode_num_accepted_tokens_total')
    result = {
        'label': label, 'thinking': thinking, 'temperature': args.temperature, 'usage': usage, 'finish_reason': finish,
        'decode_tps': count / (last-first), 'elapsed': elapsed, 'ttft': first-start,
        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'output_sha256': hashlib.sha256((thought + text).encode()).hexdigest(),
        'accepted_per_draft': accepted / drafts if drafts else None,
        'drafts': drafts, 'preemptions': sum_delta('vllm:num_preemptions_total'),
        'acceptance_by_position': {k:v for k,v in delta.items() if k.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total{')},
    }
    print('RESULT', json.dumps(result), flush=True)
    return result

initial = metrics()
assert not any(v for k, v in initial.items() if k.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))), 'Server busy'
samples, stop = [], threading.Event()
def monitor():
    while not stop.is_set():
        r = subprocess.run(['nvidia-smi', '--query-gpu=index,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        samples.append({'time': time.time(), 'gpu': r.stdout.strip()})
        stop.wait(5)
t = threading.Thread(target=monitor, daemon=True); t.start()
summary = {'name': args.name, 'started': time.time(), 'trials': []}
code = 'Write a complete, production-ready Python module for a dependency graph. Implement cycle detection, topological sorting, atomic incremental updates, serialization, and comprehensive unit tests. Output code directly; include detailed docstrings and tests. Do not abbreviate.'
agent = 'You are implementing a single-file HTML canvas animation of a pelican riding a bicycle. Plan physically consistent pedal, knee and hip positions, chain and spoke motion, then implement complete interactive HTML, CSS and JavaScript with pause, speed and resize controls. Think carefully about correctness, then write the whole file.'
try:
    generate('Write a simple Python queue with unit tests.', 'warmup', False, 128)
    for i in range(args.repeats):
        for label, prompt, thinking in [('code', code, False), ('agent', agent, True)]:
            summary['trials'].append(generate(prompt, f'{label}-{i}', thinking))
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        pair = list(pool.map(lambda i: generate(code + f' Use class DependencyGraph{i}.', f'pair-{i}', True, 1024), range(2)))
    elapsed = time.perf_counter()-start
    summary['pair'] = {'elapsed': elapsed, 'aggregate_tps': 2048/elapsed, 'requests': pair}
    summary['completed'] = time.time()
finally:
    stop.set(); t.join(timeout=6)
    summary['gpu_samples'] = samples
    (root/(args.name+'.summary.json')).write_text(json.dumps(summary, indent=2))
print('COMPLETE', args.name, flush=True)
