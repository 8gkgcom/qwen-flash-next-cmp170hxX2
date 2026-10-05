"""Qualify fusion against original HC operators, including graph replay."""
import json
import os
from pathlib import Path

import torch
import triton
from vllm.models.qwen3_8_flash_next.nvidia.ops.hc import hc_silu, hc_gate_mix
import m3_hc_runtime as fast

ROOT = Path(os.environ.get('BENCH_OUTPUT', '.'))


def error(actual, expected):
    a, b = actual.float(), expected.float()
    rel = ((a-b).square().sum()/b.square().sum().clamp(min=1.e-12)).sqrt().item()
    assert torch.isfinite(a).all() and rel < 0.005, rel
    return {'relative_l2': rel, 'max_abs': (a-b).abs().max().item()}


@torch.inference_mode()
def main():
    torch.cuda.set_device(0)
    rows = []
    for m in [1, 3, 6, 12]:
        for k, n in [(10240, 336), (10240, 320), (320, 10240)]:
            for scale in [.01, .2, 1.]:
                torch.manual_seed(123)
                # Row padding validates non-contiguous leading strides.
                x = (torch.randn(m, k+16, device='cuda', dtype=torch.bfloat16)*scale)[:, :k]
                weight = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)*.02
                wt = weight.T
                original_out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
                fused_out = torch.empty(m, n+16, device='cuda', dtype=torch.bfloat16)[:, :n]
                normalized = torch.randn(m, 10240+16, device='cuda', dtype=torch.bfloat16)[:, :10240]
                def reference():
                    torch.mm(x, wt, out=original_out)
                    return hc_silu(original_out[:, :320], 4) if k == 10240 else hc_gate_mix(normalized, original_out, 4)
                def optimized():
                    return fast.down_silu(x, wt, out=fused_out) if k == 10240 else fast.up_gate(x, wt, fused_out, normalized)
                expected, actual = reference(), optimized()
                row = {'m': m, 'k': k, 'n': n, 'scale': scale, **error(actual, expected)}
                if m != 3:
                    assert torch.equal(actual, expected), 'Fallback must be bitwise identical'
                if k == 10240:
                    row['intermediate'] = error(fused_out, original_out)
                if m == 3 and scale == .2:
                    for f in [reference, optimized]:
                        for _ in range(4): f()
                    row['original_us'] = triton.testing.do_bench_cudagraph(reference, rep=100)*1000
                    row['fused_us'] = triton.testing.do_bench_cudagraph(optimized, rep=100)*1000
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = optimized()
                    for _ in range(5):
                        x.normal_().mul_(scale)
                        graph.replay()
                        error(captured, reference())
                    row['changing_graph_replays'] = 5
                rows.append(row)
                print(json.dumps(row), flush=True)
    (ROOT/'m3-kernel-tests.json').write_text(json.dumps(rows, indent=2))
    print('M3_KERNEL_QUALIFICATION_PASS', len(rows), flush=True)


if __name__ == '__main__':
    main()
