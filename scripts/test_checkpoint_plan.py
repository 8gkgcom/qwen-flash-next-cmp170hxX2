import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from safetensors import safe_open
from sm80_ple_local import plan_ids
from vllm.models.qwen3_8_flash_next.nvidia import ple_layer as mod

torch.set_num_threads(1)
root=Path('/media/nvme/models/dealignai--Qwen3.8-Flash-Next-ABLITERATED-NVFP4')
cfg=json.loads((root/'config.json').read_text())['text_config']
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
arrays={}
for leaf in ['layer_multipliers','ngram_heads_vocab_sizes','ngram_heads_offsets']:
    name=next(n for n in index if n.endswith('layers.1.ple.ple_embedding.'+leaf))
    with safe_open(str(root/index[name]),framework='pt',device='cpu') as f:
        arrays[leaf]=f.get_tensor(name)
class Capture:
    def __call__(self,ids):
        self.ids=ids.clone()
        return torch.empty((*ids.shape,160),dtype=torch.uint8)
embedding=Capture()
P=mod.Qwen3_8FlashNextNGramEmbedding
state=SimpleNamespace(ngram_size=3,heads_per_ngram=8,eos_token_id=cfg['eos_token_id'],
    positions_buffer=torch.arange(16384),padded_buffer=torch.empty((8,16384),dtype=torch.int64),
    ngram_embedding=embedding,_shift_precompute=P._shift_precompute,_shift_apply=P._shift_apply,
    **arrays)
mod.is_offload_process=lambda:True
rng=np.random.default_rng(83)
passed=0
for nr in [1,2,3,8]:
    for width in [0,1,2,5,6,7,16,128,1280]:
        for padding in [0,1,7]:
            lengths=rng.integers(0,width+1,nr)
            starts=np.concatenate(([0],np.cumsum(lengths))).astype(np.int64)
            n=max(1,int(starts[-1])+padding)
            tokens=rng.integers(0,248319,n,dtype=np.int64)
            history=rng.integers(0,248319,(nr,2),dtype=np.int64)
            tokens[rng.random(n)<0.2]=state.eos_token_id
            tokens[rng.random(n)<0.1]=-1
            history[rng.random(history.shape)<0.3]=state.eos_token_id
            P._ple_mmap_orig_forward_impl(state,None,torch.from_numpy(tokens).clamp_min(0),
                torch.from_numpy(starts),torch.from_numpy(history))
            actual=plan_ids(tokens,starts,history,arrays['layer_multipliers'].numpy(),
                arrays['ngram_heads_vocab_sizes'].numpy(),arrays['ngram_heads_offsets'].numpy(),state.eos_token_id)
            assert np.array_equal(actual,embedding.ids.numpy().reshape(-1)),(nr,width,padding)
            passed+=1
print(json.dumps({'passed':passed,'real_checkpoint_hash_buffers':True,'reference':'original model CPU forward row IDs','negative_placeholders':True}))
