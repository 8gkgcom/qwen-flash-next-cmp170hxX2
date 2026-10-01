#!/usr/bin/env bash
set -euo pipefail
: "${GPU0_UUID:?Set physical GPU0 UUID}"
: "${GPU1_UUID:?Set physical GPU1 UUID}"
: "${MODEL_API_KEY:?Set MODEL_API_KEY}"
if [[ "$GPU0_UUID" == "$GPU1_UUID" ]]; then
  echo 'GPU0_UUID and GPU1_UUID must differ.' >&2
  exit 1
fi

export SM80_DATA_DIR="${SM80_DATA_DIR:-/media/nvme2/qwen-flash-sm80-data}"
export SM80_SITE="${SM80_SITE:-$SM80_DATA_DIR/native-root/usr/local/lib/python3.12/dist-packages}"
SM80_PYTHON="${SM80_PYTHON:-/opt/qwen-flash/venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-/media/nvme/models/dealignai--Qwen3.8-Flash-Next-ABLITERATED-NVFP4}"
export PYTHONPATH="$SM80_SITE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
# Optional CUDA package path when the Python environment does not supply its own RPATH.
if [[ -n "${SM80_NVIDIA_ROOT:-}" ]]; then
  export CPATH="$SM80_NVIDIA_ROOT/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$SM80_NVIDIA_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$SM80_NVIDIA_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU1_UUID,$GPU0_UUID"
export CUDA_MODULE_LOADING=LAZY PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_LOGGING_LEVEL=INFO
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_PP_LAYER_PARTITION=24,24
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_NVFP4_GPU=0
export VLLM_PLE_MMAP=1 VLLM_PLE_MMAP_PREWARM=1 VLLM_PLE_MMAP_LOCK=1
export VLLM_PLE_MMAP_WORKERS=8 VLLM_PLE_MMAP_CHUNK=2048
export VLLM_PLE_RAM_GATHER=1 VLLM_PLE_RAM_GATHER_MAX_ROWS=262144
export VLLM_PLE_GDS=0 VLLM_PLE_GDS_RUNNER=0
export Q38_RAM_COMPUTE_OPT=1 Q38_RAM_PARALLEL=pp2 Q38_PLE_LOCAL=1
export Q38_PP_DRAFT_INT8=1 Q38_DRAFT_INT8=0
export Q38_PLE_GPU_CACHE_MIB=0 Q38_TP_PLE=0 Q38_HC_GEMV=0
export Q38_PLE_SERIALIZE_LARGE_INPUTS=1
export VLLM_PLE_MTP_MOE_BACKEND=triton
export QWEN_GDN_REPLAY=1 GDN_DIAG_DISABLE_JIT_MONITOR=1
export R38_PP1_FULL_DECODE=0 R38_TOKEN_DIAGNOSTICS=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_SHM_DISABLE=1 NCCL_DEBUG=INFO
export VLLM_CACHE_ROOT="$SM80_DATA_DIR/cache/vllm-ram/pp2-optimized"
export TRITON_CACHE_DIR="$SM80_DATA_DIR/cache/triton"
export Q38_PP_DRAFT_EVIDENCE="$SM80_DATA_DIR/logs/pp-draft-int8.json"

mtp_tokens="${MTP_TOKENS:-4}"
if [[ ! "$mtp_tokens" =~ ^[1-9][0-9]*$ ]]; then
  echo 'MTP_TOKENS must be a positive integer.' >&2
  exit 1
fi
args=(
  "$SM80_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_DIR"
  --served-model-name qwen --host "${MODEL_HOST:-0.0.0.0}" --port "${MODEL_PORT:-8000}"
  --load-format safetensors --safetensors-load-strategy lazy
  --distributed-executor-backend mp --tensor-parallel-size 1 --pipeline-parallel-size 2
  --quantization modelopt_fp4 --linear-backend marlin --dtype bfloat16
  --max-model-len "${MAX_MODEL_LEN:-262400}" --block-size 1632
  --max-num-seqs "${MAX_NUM_SEQS:-8}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}"
  --enable-prefix-caching --mamba-cache-mode align --enable-chunked-prefill
  --max-num-batched-tokens "${PREFILL_TOKENS:-8192}" --async-scheduling
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}'
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
  -cc.cudagraph_mode=FULL_AND_PIECEWISE
  '-cc.splitting_ops=["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup","vllm::ple_gds_lookup"]'
  '-cc.inductor_compile_config={"combo_kernels":false,"benchmark_combo_kernel":false}'
  --no-enable-flashinfer-autotune
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp_tokens,\"use_local_argmax_reduction\":true}"
  --limit-mm-per-prompt '{"image":0,"video":0}'
  --long-prefill-token-threshold "${LONG_PREFILL_THRESHOLD:-1280}"
  --disable-custom-all-reduce
)

# Print a shareable command without launching, loading a model or writing cache files.
if [[ "${SM80_PRINT_COMMAND:-0}" == 1 ]]; then
  printf '%q ' "${args[@]}"
  printf '%s\n' '--api-key <MODEL_API_KEY>'
  exit 0
fi
[[ -x "$SM80_PYTHON" ]] || { echo 'SM80_PYTHON is not executable.' >&2; exit 1; }
[[ -f "$SM80_SITE/vllm/__init__.py" && -f "$SM80_SITE/ple_ram_gather.so" ]] || {
  echo 'Prepare the pinned base and apply the runtime overlay first.' >&2; exit 1;
}
[[ -f "$MODEL_DIR/config.json" ]] || { echo 'Missing model config.json.' >&2; exit 1; }
[[ -d /evidence && -w /evidence ]] || { echo 'The runtime user needs a writable /evidence directory.' >&2; exit 1; }
mkdir -p "$SM80_DATA_DIR/logs" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"
exec "${args[@]}" --api-key "$MODEL_API_KEY"
