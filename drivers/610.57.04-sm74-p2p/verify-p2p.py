"""Small, bounded post-boot CUDA/NCCL correctness check; no model changes."""
import datetime
import json
import os
from pathlib import Path
import socket
import sys
import time

os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,GRAPH,P2P")
os.environ.setdefault("NCCL_P2P_DISABLE", "0")
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
os.environ.setdefault("NCCL_SHM_DISABLE", "1")
os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, port, result_dir):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    start = time.monotonic()
    props = torch.cuda.get_device_properties(rank)
    # Exactly representable BF16 inputs/output; tests CUDA math independently.
    a = torch.ones((1024, 1024), dtype=torch.bfloat16, device=rank)
    b = a @ a
    matrix_bad = int(torch.count_nonzero(b != 1024).item())
    del a, b
    torch.cuda.empty_cache()
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}",
        world_size=2, rank=rank, timeout=datetime.timedelta(seconds=20),
        device_id=torch.device(f"cuda:{rank}"),
    )
    # 1 MiB per payload, varying nonzero patterns in both directions.
    base = torch.arange(262144, dtype=torch.int32, device=rank) % 997
    errors = []
    for root in (0, 1):
        for salt in (17, 193, 65537):
            expected = base + salt + root * 101
            buf = expected.clone() if rank == root else torch.full_like(base, -1)
            dist.broadcast(buf, src=root)
            errors.append({
                "root": root, "salt": salt,
                "mismatches": int(torch.count_nonzero(buf != expected).item()),
            })
    buf = base + rank * 31
    dist.all_reduce(buf)
    reduction_bad = int(torch.count_nonzero(buf != base * 2 + 31).item())
    torch.cuda.synchronize(rank)
    result = {
        "rank": rank, "name": props.name, "sm_count": props.multi_processor_count,
        "matrix_mismatches": matrix_bad, "broadcast_checks": errors,
        "allreduce_mismatches": reduction_bad,
        "max_allocated_mib": torch.cuda.max_memory_allocated(rank) / 2**20,
        "elapsed_s": round(time.monotonic() - start, 3),
    }
    result["passed"] = matrix_bad == 0 and reduction_bad == 0 and all(
        x["mismatches"] == 0 for x in errors
    )
    Path(result_dir, f"postboot-p2p-rank{rank}.json").write_text(json.dumps(result, indent=2))
    print("RESULT " + json.dumps(result), flush=True)
    dist.destroy_process_group()
    if not result["passed"]:
        raise RuntimeError("GPU or collective data-integrity check failed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 verify-p2p.py RESULTS_DIRECTORY")
    Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port, sys.argv[1]), nprocs=2, join=True)
