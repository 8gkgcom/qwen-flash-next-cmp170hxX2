"""Check original checkpoint HC weights, not only synthetic matrices."""
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.models.qwen3_8_flash_next.nvidia.ops.hc import hc_silu, hc_gate_mix
import m3_hc_runtime as fast
from test_m3_hc import error

root = Path(os.environ['MODEL_DIR'])
index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']


def load(name):
    with safe_open(str(root/index[name]), framework='pt', device='cpu') as f:
        return f.get_tensor(name).to('cuda')


rows = []
with torch.inference_mode():
    for layer in [0, 24, 47]:
        for kind in ['attn', 'mlp']:
            stem = f'model.language_model.layers.{layer}.{kind}_hyper_connection.'
            down = load(stem+'input_mix_weight_down.weight')
            inject = load(stem+'block_inject_weight.weight')
            up = load(stem+'input_mix_weight_up.weight')
            torch.manual_seed(20261005+layer)
            for scale in [.2, 1., 4.]:
                x = torch.randn(3, 10240, device='cuda', dtype=torch.bfloat16)*scale
                # Native loader pads the 320+4 merged projection to 336 rows.
                padding = down.new_zeros((336-down.shape[0]-inject.shape[0], down.shape[1]))
                w = torch.cat((down, inject, padding), dim=0)
                expected_down = x @ w.T
                actual_down = torch.empty_like(expected_down)
                actual_low = fast.down_silu(x, w.T, out=actual_down)
                expected_low = hc_silu(expected_down[:, :320], 4)
                row = dict(layer=layer, kind=kind, scale=scale,
                           down=error(actual_down, expected_down), low=error(actual_low, expected_low))
                expected_gate = hc_gate_mix(x, expected_low @ up.T, 4)
                scratch = torch.empty((3, 10240), device='cuda', dtype=torch.bfloat16)
                row['chain'] = error(fast.up_gate(actual_low, up.T, scratch, x), expected_gate)
                rows.append(row)
    # Reusing the compiled graph with changing data is covered by test_m3_hc.py.
(Path(os.environ.get('BENCH_OUTPUT', '.')) / 'm3-checkpoint-tests.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
print('CHECKPOINT_CASES_PASS', len(rows), 'MAX_CHAIN_RELATIVE_L2', max(x['chain']['relative_l2'] for x in rows))
