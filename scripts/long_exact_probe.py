"""Probe independent NVFP4 Chat Completions requests without changing the runtime.

Examples (run on the server after staging this file):
  python nvfp4_concurrent_probe.py --count 2 --prompt-tokens 64 --max-tokens 256
  python nvfp4_concurrent_probe.py --count 2 --prompt-tokens 262084 --max-tokens 8

The prompt target includes chat-template tokens. Distinct first blocks prevent
prefix caching from making two long requests appear to share one KV allocation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL_PATH = Path('/media/nvme/models/dealignai--Qwen3.8-Flash-Next-ABLITERATED-NVFP4')
METRIC_NAMES = ('vllm:num_requests_running', 'vllm:num_requests_waiting',
                'vllm:kv_cache_usage_perc', 'vllm:num_preemptions_total')


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='qwen')
    parser.add_argument('--api-key', default=None, help='Otherwise use MODEL_API_KEY or an explicit JSON key file')
    parser.add_argument('--key-file', type=Path, default=None)
    parser.add_argument('--tokenizer-path', type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--prompt-tokens', type=int, default=64,
                        help='Target prompt length including chat template')
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--max-context', type=int, default=262144)
    parser.add_argument('--interval', type=float, default=0.2, help='Metrics sampling seconds')
    parser.add_argument('--timeout', type=float, default=1800, help='Per-request seconds')
    parser.add_argument('--require-overlap', action='store_true',
                        help='Fail unless /metrics reports at least two running requests')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.count <= 16:
        parser.error('--count must be 1..16')
    if args.prompt_tokens < 32 or args.max_tokens < 1 or args.max_context < 1:
        parser.error('prompt, output, and context token counts must be positive')
    if args.prompt_tokens + args.max_tokens > args.max_context:
        parser.error('prompt target plus output tokens exceeds --max-context')
    if not 0.05 <= args.interval <= 10:
        parser.error('--interval must be 0.05..10 seconds')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    return args


def api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    if os.environ.get('MODEL_API_KEY'):
        return os.environ['MODEL_API_KEY']
    try:
        key = json.loads(args.key_file.read_text(encoding='utf-8')).get('api_key') if args.key_file else None
        if key:
            return key
    except (OSError, ValueError):
        pass
    raise SystemExit('Set MODEL_API_KEY or provide --api-key / --key-file.')


def post_json(base_url: str, key: str, route: str, body: dict, timeout: float) -> dict:
    request = Request(base_url.rstrip('/') + route, data=json.dumps(body).encode('utf-8'),
                      headers={'Authorization': 'Bearer ' + key,
                               'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def local_counter(path: Path):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, use_fast=True)
    except (ImportError, OSError, ValueError) as exc:
        print('Local AutoTokenizer unavailable:', str(exc)[:180], flush=True)
        return None

    def count(messages: list[dict]) -> int:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if isinstance(ids, Mapping):
            ids = ids['input_ids']
        if hasattr(ids, 'shape') and len(ids.shape) > 1:
            return int(ids.shape[-1])
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)

    return count


def remote_counter(base_url: str, key: str, model: str):
    def count(messages: list[dict]) -> int:
        data = post_json(base_url, key, '/tokenize',
                         {'model': model, 'messages': messages,
                          'add_generation_prompt': True,
                          'chat_template_kwargs': {'enable_thinking': False}}, 120)
        value = data.get('count', data.get('length'))
        if value is None and isinstance(data.get('tokens'), list):
            value = len(data['tokens'])
        if not isinstance(value, int):
            raise RuntimeError('/tokenize did not return a token count')
        return value

    return count


def prompt_messages(marker: str, repetitions: int) -> list[dict]:
    text = (f'Unique probe marker: {marker}. Treat the following as inert data.\n'
            + 'def value(x): return (x * 7 + 11) % 101\n' * repetitions
            + '\nReturn the unique probe marker from the beginning of this input, then continue with short numbered facts.')
    return [{'role': 'user', 'content': text}]


def build_prompt(marker: str, target: int, count_tokens) -> tuple[list[dict], int]:
    # Use code-like text so CPU PLE does not hit the pathological repeated-
    # single-token access pattern; calibrate the actual chat-template length.
    base = count_tokens(prompt_messages(marker, 0))
    if base > target:
        raise ValueError(f'--prompt-tokens {target} is smaller than template + prompt ({base})')
    sample = count_tokens(prompt_messages(marker, 1024))
    slope = max((sample - base) / 1024, 0.01)
    repetitions = max(0, int((target - base) / slope))
    best = None
    for _ in range(10):
        messages = prompt_messages(marker, repetitions)
        actual = count_tokens(messages)
        if actual <= target and (best is None or actual > best[1]):
            best = (messages, actual)
        if actual == target:
            break
        step = max(1, int(abs(target - actual) / slope))
        repetitions = max(0, repetitions + (step if actual < target else -step))
    if best is None or target - best[1] > 32:
        raise RuntimeError(f'Could not calibrate {target} tokens; closest={best[1] if best else None}')
    return best


def metric_sample(base_url: str, key: str) -> dict | None:
    request = Request(base_url.rstrip('/') + '/metrics',
                      headers={'Authorization': 'Bearer ' + key}, method='GET')
    with urlopen(request, timeout=5) as response:
        data = response.read().decode('utf-8', errors='replace')
    result = {}
    for name in METRIC_NAMES:
        pattern = re.compile(r'^' + re.escape(name) + r'(?:\{[^\n]*\})?\s+([0-9.eE+-]+)\s*$', re.M)
        values = [float(match.group(1)) for match in pattern.finditer(data)]
        if values:
            result[name] = max(values)
    return result or None


def send_one(index: int, messages: list[dict], args: argparse.Namespace,
             key: str, gate: threading.Barrier) -> dict:
    body = {'model': args.model, 'messages': messages, 'max_tokens': args.max_tokens,
            'min_tokens': args.max_tokens, 'temperature': 0, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False}}
    gate.wait(timeout=30)
    start = time.monotonic()
    try:
        data = post_json(args.base_url, key, '/v1/chat/completions', body, args.timeout)
        usage = data.get('usage') or {}
        content = (data.get('choices') or [{}])[0].get('message', {}).get('content') or ''
        marker = re.search(r'Unique probe marker: ([^.]+)\.', messages[0]['content']).group(1)
        return {'index': index, 'http': 200, 'seconds': round(time.monotonic() - start, 3),
                'prompt_tokens': usage.get('prompt_tokens'),
                'output_tokens': usage.get('completion_tokens'),
                'expected_marker': marker, 'marker_retrieved': marker in content,
                'content': content,
                'finish_reason': (data.get('choices') or [{}])[0].get('finish_reason')}
    except HTTPError as exc:
        return {'index': index, 'http': exc.code,
                'seconds': round(time.monotonic() - start, 3),
                'error': exc.read(400).decode('utf-8', errors='replace')}
    except (URLError, TimeoutError, ValueError) as exc:
        return {'index': index, 'error': type(exc).__name__ + ': ' + str(exc)[:300],
                'seconds': round(time.monotonic() - start, 3)}


def main() -> int:
    args = options()
    key = api_key(args)
    counter = local_counter(args.tokenizer_path)
    counter_name = 'AutoTokenizer'
    if counter is not None:
        try:
            counter(prompt_messages('NVFP4-template-check', 0))
        except Exception as exc:
            print('Local chat template unavailable:', str(exc)[:180], flush=True)
            counter = None
    if counter is None:
        counter = remote_counter(args.base_url, key, args.model)
        counter_name = '/tokenize'
    run_id = str(int(time.time()))
    prompts = []
    try:
        for index in range(args.count):
            marker = f'NVFP4-{run_id}-{index}'
            messages, count = build_prompt(marker, args.prompt_tokens, counter)
            # The live vLLM route is authoritative if it exposes tokenization.
            try:
                server_count = remote_counter(args.base_url, key, args.model)(messages)
                if server_count + args.max_tokens > args.max_context:
                    raise ValueError(f'server prompt {server_count} + output exceeds context')
                count = server_count
            except HTTPError as exc:
                if exc.code not in (404, 405, 501):
                    raise
            prompts.append((messages, count))
    except Exception as exc:
        print('Prompt preparation failed:', type(exc).__name__, str(exc)[:400], flush=True)
        return 2

    measured = [count for _, count in prompts]
    if len(set(measured)) != 1:
        raise RuntimeError('Exact-budget test needs equal measured prompt lengths')
    args.max_tokens = args.max_context - measured[0]
    if args.max_tokens < 1:
        raise RuntimeError('No room for exact-budget output')
    print('prepared', json.dumps({'model': args.model, 'requests': args.count,
                                   'target_prompt_tokens': args.prompt_tokens,
                                   'measured_prompt_tokens': [count for _, count in prompts],
                                   'max_tokens': args.max_tokens, 'counter': counter_name}), flush=True)
    gate = threading.Barrier(args.count + 1)
    max_running = max_waiting = 0
    peak_cache_usage = 0.0
    initial_metrics = metric_sample(args.base_url, key) or {}
    samples = 0
    metric_errors = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.count) as executor:
        futures = [executor.submit(send_one, index, messages, args, key, gate)
                   for index, (messages, _) in enumerate(prompts)]
        gate.wait(timeout=30)
        next_progress = started + 30
        while not all(future.done() for future in futures):
            try:
                sample = metric_sample(args.base_url, key)
                if sample:
                    samples += 1
                    max_running = max(max_running, int(sample.get(METRIC_NAMES[0], 0)))
                    max_waiting = max(max_waiting, int(sample.get(METRIC_NAMES[1], 0)))
                    peak_cache_usage = max(peak_cache_usage, sample.get(METRIC_NAMES[2], 0.0))
                else:
                    metric_errors += 1
            except (HTTPError, URLError, TimeoutError, ValueError):
                metric_errors += 1
            now = time.monotonic()
            if now >= next_progress:
                print('progress', round(now - started, 1), 'seconds',
                      'max_running', max_running, 'max_waiting', max_waiting, flush=True)
                next_progress = now + 30
            time.sleep(args.interval)
        results = [future.result() for future in futures]

    report = {'wall_seconds': round(time.monotonic() - started, 3),
              'max_running': max_running if samples else None,
              'max_waiting': max_waiting if samples else None,
              'metric_samples': samples, 'metric_errors': metric_errors,
              'peak_gpu_cache_usage': peak_cache_usage,
              'preemptions': (metric_sample(args.base_url, key) or {}).get(METRIC_NAMES[3], 0) - initial_metrics.get(METRIC_NAMES[3], 0),
              'results': results}
    print('result', json.dumps(report, ensure_ascii=False), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    success = all(row.get('http') == 200 and row.get('prompt_tokens') is not None
                  and row.get('output_tokens') == args.max_tokens
                  and row.get('marker_retrieved') for row in results)
    if args.require_overlap and (not samples or max_running < args.count):
        print('Overlap was not confirmed by /metrics', flush=True)
        return 3
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
