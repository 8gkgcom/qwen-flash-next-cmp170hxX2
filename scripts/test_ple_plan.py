"""Check EOS, padded graphs, uneven batches and signed hash overflow exactly."""
import json
from pathlib import Path
import numpy as np
import torch
from types import SimpleNamespace
from sm80_ple_local import plan_ids
from vllm_ple_gds import _plan_ngram_row_ids_reference
from vllm.models.qwen3_8_flash_next.nvidia.ple_layer import Qwen3_8FlashNextNGramEmbedding as P

rng = np.random.default_rng(123)
mul = np.array([123456781234567, 23456789234567, 3456789134567],dtype=np.int64)
sizes = np.array([20000003 + i*6 for i in range(16)],dtype=np.int64)
offsets = np.concatenate(([0],np.cumsum(sizes)[:-1])).astype(np.int64)
layer=SimpleNamespace(ngram_size=3,heads_per_ngram=8,eos_token_id=248044,
    _ple_gds_multipliers_numpy=mul,_ple_gds_vocab_sizes_numpy=sizes,_ple_gds_offsets_numpy=offsets,
    _ple_gds_layer_multipliers_cpu=torch.from_numpy(mul),
    _ple_gds_vocab_sizes_cpu=torch.from_numpy(sizes),_ple_gds_offsets_cpu=torch.from_numpy(offsets),
    _ple_gds_reader=SimpleNamespace(row_count=int(sizes.sum())),
    _shift_precompute=P._shift_precompute,_shift_apply=P._shift_apply)
count=0
for requests in [1,2,3,8]:
    for width in [0,1,2,5,6,7,16,128,1280]:
        for padding in [0,1,7]:
            lengths=rng.integers(0,width+1,requests)
            starts=np.concatenate(([0],np.cumsum(lengths))).astype(np.int64)
            n=max(1,int(starts[-1])+padding)
            tokens=rng.integers(0,248319,n,dtype=np.int64)
            history=rng.integers(0,248319,(requests,2),dtype=np.int64)
            tokens[rng.random(n)<0.2]=layer.eos_token_id
            history[rng.random(history.shape)<0.3]=layer.eos_token_id
            actual=plan_ids(tokens,starts,history,mul,sizes,offsets,layer.eos_token_id)
            expected=_plan_ngram_row_ids_reference(layer,torch.from_numpy(tokens),torch.from_numpy(starts),torch.from_numpy(history)).numpy()
            assert np.array_equal(actual,expected),(requests,width,padding)
            count+=1
print(json.dumps({'passed':count,'comparison':'exact n-gram row IDs, including EOS and padding'}))
