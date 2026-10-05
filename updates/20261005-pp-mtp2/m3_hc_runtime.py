"""PP2 MTP2 HC fusion, adapted from the author's seven-row BF16 path.

Only three-row HC operations are replaced. Other batch sizes use the original
matmul/custom operators, including multi-request and prefill paths.
"""
import ast
from collections import Counter
import functools
import json
import os
from pathlib import Path

import torch
import triton
from m3_hc_kernels import mm3, finish3, up_gate3

ENABLED = os.getenv('Q38_PP_M3_HC', '0') == '1'
CAPTURED = Counter()
SHAPES = {(10240, 336), (10240, 320), (320, 10240)}


def valid(x, wt, out):
    return (ENABLED and x.ndim == wt.ndim == out.ndim == 2
            and x.shape == (3, wt.shape[0]) and out.shape == (3, wt.shape[1])
            and wt.stride() == (1, wt.shape[0])
            and all(t.is_cuda and t.device == x.device and t.dtype == torch.bfloat16
                    for t in (x, wt, out))
            and x.stride(1) == out.stride(1) == 1)


def mark(kind):
    if torch.cuda.is_current_stream_capturing():
        CAPTURED[kind] += 1


def down_silu(x, wt, *, out):
    if not valid(x, wt, out):
        torch.mm(x, wt, out=out)
        return torch.ops.vllm.qwen3_8_flash_next_hc_silu.default(out[:, :320], 4)
    n = wt.shape[1]
    assert wt.shape[0] == 10240 and n in (320, 336)
    low = torch.empty((3, 320), dtype=x.dtype, device=x.device)
    partial = torch.empty((16, 3, n), dtype=torch.float32, device=x.device)
    mm3[(triton.cdiv(n, 32), 16)](
        x, wt.T, out, partial, low, n, 10240, x.stride(0), out.stride(0),
        32, 128, 16, True, num_warps=4, num_stages=3)
    finish3[(triton.cdiv(3*n, 256),)](partial, out, low, n, out.stride(0), 16, True, 256)
    mark('down_silu')
    return low


def up_gate(low, wt, out, normalized):
    if not valid(low, wt, out):
        torch.mm(low, wt, out=out)
        return torch.ops.vllm.qwen3_8_flash_next_hc_gate_mix.default(normalized, out, 4)
    assert wt.shape == (320, 10240)
    assert normalized.shape == (3, 10240) and normalized.stride(1) == 1
    result = torch.empty((3, 2560), dtype=low.dtype, device=low.device)
    up_gate3[(40,)](low, wt.T, normalized, result, low.stride(0), normalized.stride(0),
                    2560, 64, 64, num_warps=4, num_stages=3)
    mark('up_gate')
    return result


def instrument(source):
    if 'qwen3_8_flash_next_hc_gate_mix.default' not in source:
        return source, 0
    tree = ast.parse(source)
    original_call = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'call'), None)
    if original_call is None:
        return source, 0
    lines = source.splitlines(keepends=True)
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if ast.unparse(call.func) != 'extern_kernels.mm' or len(call.args) != 2:
            continue
        wt = call.args[1]
        if not isinstance(wt, ast.Call) or ast.unparse(wt.func) != 'reinterpret_tensor':
            continue
        try:
            shape, stride = ast.literal_eval(wt.args[1]), ast.literal_eval(wt.args[2])
        except (ValueError, TypeError, IndexError):
            continue
        if shape not in SHAPES or stride != (1, shape[0]):
            continue
        assert node.lineno == node.end_lineno
        call.func = ast.parse('_hc_gemv.mm', mode='eval').body
        edits.append((node.lineno - 1, ast.unparse(node)))
    if not edits:
        return source, 0
    for i, body in sorted(edits, reverse=True):
        indent = lines[i][:len(lines[i])-len(lines[i].lstrip())]
        lines[i] = indent + body + '\n'
    # Reuse the pinned author's structural HC chain matcher; no generic GEMM rewrite.
    from m7_joint_runtime import instrument as fuse
    fused = fuse(''.join(lines))
    assert '_hc_gemv.mm' not in fused, 'Unmatched HC operation'
    fused = fused.replace('m7_joint_runtime as _m7_joint', 'm3_hc_runtime as _m3_hc')
    fused = fused.replace('_m7_joint.', '_m3_hc.')
    # Preserve the entire original compiled function for all non-M3 inputs.
    # In particular, prefill must retain Inductor's original operation order,
    # buffer lifetimes, and external GEMM dispatch, not a Python reconstruction.
    token_symbols = set()
    for node in ast.walk(original_call):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'empty_strided_cuda':
            shape = node.args[0]
            if (isinstance(shape, ast.Tuple) and len(shape.elts) == 2
                    and isinstance(shape.elts[1], ast.Constant)
                    and shape.elts[1].value in (320, 336, 10240)
                    and isinstance(shape.elts[0], ast.Name)):
                token_symbols.add(shape.elts[0].id)
    if len(token_symbols) != 1:
        return source, 0
    symbol = token_symbols.pop()
    arg_name = None
    unpacked = None
    for node in ast.walk(original_call):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == symbol and isinstance(node.value, ast.Name):
            arg_name = node.value.id
        if (isinstance(target, ast.Tuple) and isinstance(node.value, ast.Name)
                and node.value.id == original_call.args.args[-1].arg):
            unpacked = [n.id for n in target.elts if isinstance(n, ast.Name)]
    if arg_name is None or unpacked is None or arg_name not in unpacked:
        return source, 0
    index = unpacked.index(arg_name)
    fused_call = next(n for n in ast.walk(ast.parse(fused)) if isinstance(n, ast.FunctionDef) and n.name == 'call')
    assert not original_call.decorator_list and not fused_call.decorator_list
    original_lines = source.splitlines(keepends=True)
    at, end = original_call.lineno - 1, original_call.end_lineno
    original_body = ''.join(original_lines[at:end]).replace('def call(', 'def _m3_hc_original_call(', 1)
    fused_body = ''.join(fused.splitlines(keepends=True)[fused_call.lineno-1:fused_call.end_lineno])
    fused_body = fused_body.replace('def call(', 'def _m3_hc_fused_call(', 1)
    indent = original_lines[at][:len(original_lines[at])-len(original_lines[at].lstrip())]
    receiver = 'self.' if original_call.args.args[0].arg == 'self' else ''
    wrapper = (f'{indent}def call({ast.unparse(original_call.args)}):\n'
               f'{indent}    if _m3_hc.ENABLED and args[{index}] == 3:\n'
               f'{indent}        return {receiver}_m3_hc_fused_call(args)\n'
               f'{indent}    return {receiver}_m3_hc_original_call(args)\n')
    original_lines[at:end] = [original_body+'\n'+fused_body+'\n'+wrapper]
    output = 'import m3_hc_runtime as _m3_hc\n' + ''.join(original_lines)
    compile(output, '<m3_hc_instrument>', 'exec')
    return output, len(edits)


def install():
    if not ENABLED:
        return
    import torch._inductor.codecache as cc
    import torch._inductor.runtime.compile_tasks as ct
    previous = cc._reload_python_module
    seen = set()

    def reload(key, path, *args, **kwargs):
        if path not in seen:
            p = Path(path)
            original = p.read_text()
            if 'import m3_hc_runtime as _m3_hc' not in original:
                patched, count = instrument(original)
                if count:
                    backup = p.with_suffix('.before-round2-m3')
                    if not backup.exists():
                        backup.write_text(original)
                    p.write_text(patched)
                    print('PP_M3_HC_REWRITE', key, count, flush=True)
            seen.add(path)
        return previous(key, path, *args, **kwargs)

    cc._reload_python_module = reload
    ct._reload_python_module = reload
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    old_capture = CudaGraphManager.capture

    @functools.wraps(old_capture)
    def capture(self, *args, **kwargs):
        result = old_capture(self, *args, **kwargs)
        record = dict(pid=os.getpid(), device=str(self.device), manager=type(self).__name__,
                      counts=dict(CAPTURED), query_len=self.decode_query_len)
        folder = os.getenv('Q38_M3_EVIDENCE')
        if folder:
            Path(folder).mkdir(exist_ok=True, parents=True)
            (Path(folder)/f'm3-capture-{os.getpid()}-{type(self).__name__}.json').write_text(json.dumps(record))
        print('PP_M3_HC_CAPTURE', json.dumps(record), flush=True)
        return result

    CudaGraphManager.capture = capture
    print('PP_M3_HC_INSTALLED', flush=True)
