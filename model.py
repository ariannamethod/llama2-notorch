"""
Llama 2 model — backed by notorch (ctypes to libnotorch).

Originally from llama2.c by Andrej Karpathy.
PyTorch removed. notorch is the only engine.

Architecture: RMSNorm, SwiGLU, RoPE, multi-head attention.
no import torch. no nn.Module. no F.softmax.
just ctypes to C. the way it should be.
"""

import math
import ctypes
import os
from dataclasses import dataclass
from typing import Optional

from ariannamethod.notorch_nn import (
    _lib, _get_tensor_struct, _NtTapeEntry, _NtTensor,
    Tensor, Parameter, Module, Linear, Embedding, RMSNorm,
    softmax, multinomial, seed as nt_seed,
)
from ariannamethod.chuck import ChuckOptimizer


@dataclass
class ModelArgs:
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 32
    norm_eps: float = 1e-5
    max_seq_len: int = 256
    dropout: float = 0.0


class Transformer(Module):
    """Llama 2 Transformer backed by notorch via ctypes."""

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.dim = params.dim
        self.n_heads = params.n_heads
        self.n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
        self.head_dim = params.dim // params.n_heads
        self.max_seq_len = params.max_seq_len

        # compute hidden dim (SwiGLU style: 2/3 * 4d rounded to multiple_of)
        hidden_dim = params.hidden_dim
        if hidden_dim is None:
            hidden_dim = 4 * params.dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = params.multiple_of * ((hidden_dim + params.multiple_of - 1) // params.multiple_of)
        self.hidden_dim = hidden_dim

        # token embedding
        self.tok_emb = Embedding(params.vocab_size, params.dim)

        # transformer layers
        self.layers = []
        for l in range(params.n_layers):
            layer = {
                'rms1': RMSNorm(params.dim),
                'wq': Linear(params.dim, params.n_heads * self.head_dim),
                'wk': Linear(params.dim, self.n_kv_heads * self.head_dim),
                'wv': Linear(params.dim, self.n_kv_heads * self.head_dim),
                'wo': Linear(params.n_heads * self.head_dim, params.dim),
                'rms2': RMSNorm(params.dim),
                'w_gate': Linear(params.dim, hidden_dim),
                'w_up': Linear(params.dim, hidden_dim),
                'w_down': Linear(hidden_dim, params.dim),
            }
            for k, v in layer.items():
                setattr(self, f'l{l}_{k}', v)
            self.layers.append(layer)

        # final norm and output head
        self.norm_f = RMSNorm(params.dim)
        self.head = Linear(params.dim, params.vocab_size)

    def param_list(self):
        """Ordered parameters matching the forward pass layout."""
        params = [self.tok_emb.weight]
        for l in self.layers:
            params.extend([
                l['rms1'].weight, l['wq'].weight, l['wk'].weight,
                l['wv'].weight, l['wo'].weight, l['rms2'].weight,
                l['w_gate'].weight, l['w_up'].weight, l['w_down'].weight,
            ])
        params.extend([self.norm_f.weight, self.head.weight])
        return params

    def count_params(self):
        return sum(p.numel for p in self.param_list())

    def forward_train(self, token_ids, target_ids):
        """Forward through notorch tape. Returns (loss_idx, loss_val)."""
        CTX = len(token_ids)
        DIM = self.dim
        HD = self.head_dim

        _lib.nt_tape_start()
        _lib.nt_train_mode(1)

        params = self.param_list()
        tape_ids = [_lib.nt_tape_param(p._ptr) for p in params]
        _lib.nt_tape_no_decay(tape_ids[0])  # embedding — no weight decay

        # Build token/target tensors
        tok_t = Tensor.zeros(CTX)
        tgt_t = Tensor.zeros(CTX)
        tok_t.set_data([float(x) for x in token_ids])
        tgt_t.set_data([float(x) for x in target_ids])
        tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tok_t._owns = False
        tgt_t._owns = False

        # Forward: embedding -> blocks -> rmsnorm -> lm_head -> cross_entropy
        pi = 0
        h = _lib.nt_seq_embedding(tape_ids[pi], -1, tok_idx, CTX, DIM); pi += 1

        for l in range(self.n_layers):
            rms1=tape_ids[pi]; pi+=1
            wq=tape_ids[pi]; pi+=1; wk=tape_ids[pi]; pi+=1
            wv=tape_ids[pi]; pi+=1; wo=tape_ids[pi]; pi+=1
            rms2=tape_ids[pi]; pi+=1
            wg=tape_ids[pi]; pi+=1; wu=tape_ids[pi]; pi+=1; wd=tape_ids[pi]; pi+=1

            xn = _lib.nt_seq_rmsnorm(h, rms1, CTX, DIM)
            q = _lib.nt_rope(_lib.nt_seq_linear(wq, xn, CTX), CTX, HD)
            k = _lib.nt_rope(_lib.nt_seq_linear(wk, xn, CTX), CTX, HD)
            v = _lib.nt_seq_linear(wv, xn, CTX)
            attn = _lib.nt_mh_causal_attention(q, k, v, CTX, HD)
            h = _lib.nt_add(h, _lib.nt_seq_linear(wo, attn, CTX))

            xn = _lib.nt_seq_rmsnorm(h, rms2, CTX, DIM)
            gate = _lib.nt_silu(_lib.nt_seq_linear(wg, xn, CTX))
            up = _lib.nt_seq_linear(wu, xn, CTX)
            h = _lib.nt_add(h, _lib.nt_seq_linear(wd, _lib.nt_mul(gate, up), CTX))

        rmsf=tape_ids[pi]; pi+=1; head_i=tape_ids[pi]; pi+=1
        hf = _lib.nt_seq_rmsnorm(h, rmsf, CTX, DIM)
        logits_idx = _lib.nt_seq_linear(head_i, hf, CTX)
        loss_idx = _lib.nt_seq_cross_entropy(logits_idx, tgt_idx, CTX, self.vocab_size)

        # Read loss value from tape
        tape_ptr = _lib.nt_tape_get()
        entry_size = ctypes.sizeof(_NtTapeEntry)
        tape_addr = ctypes.cast(tape_ptr, ctypes.c_void_p).value
        loss_entry = ctypes.cast(
            tape_addr + loss_idx * entry_size,
            ctypes.POINTER(_NtTapeEntry)
        ).contents
        loss_tensor = ctypes.cast(loss_entry.output, ctypes.POINTER(_NtTensor)).contents
        loss_val = loss_tensor.data[0]

        return loss_idx, loss_val

    def backward_step(self, loss_idx, loss_val, lr):
        """Backward + Chuck optimizer + clear tape."""
        _lib.nt_tape_backward(loss_idx)
        _lib.nt_tape_clip_grads(ctypes.c_float(1.0))
        _lib.nt_tape_chuck_step(ctypes.c_float(lr), ctypes.c_float(loss_val))
        _lib.nt_tape_clear()

    def generate(self, token_ids, max_new=200, temperature=0.8, top_k=40):
        """Autoregressive generation via notorch."""
        _lib.nt_train_mode(0)
        ctx = list(token_ids)

        for _ in range(max_new):
            if len(ctx) > self.max_seq_len:
                ctx = ctx[-self.max_seq_len:]
            CTX = len(ctx)

            _lib.nt_tape_start()
            params = self.param_list()
            tape_ids = [_lib.nt_tape_param(p._ptr) for p in params]

            tok_t = Tensor.zeros(CTX)
            tgt_t = Tensor.zeros(CTX)
            tok_t.set_data([float(x) for x in ctx])
            tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))
            tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, ctypes.c_float(0))
            tok_t._owns = False
            tgt_t._owns = False

            pi = 0
            h = _lib.nt_seq_embedding(tape_ids[pi], -1, tok_idx, CTX, self.dim); pi += 1
            for l in range(self.n_layers):
                rms1=tape_ids[pi]; pi+=1
                wq=tape_ids[pi]; pi+=1; wk=tape_ids[pi]; pi+=1
                wv=tape_ids[pi]; pi+=1; wo=tape_ids[pi]; pi+=1
                rms2=tape_ids[pi]; pi+=1
                wg=tape_ids[pi]; pi+=1; wu=tape_ids[pi]; pi+=1; wd=tape_ids[pi]; pi+=1
                xn = _lib.nt_seq_rmsnorm(h, rms1, CTX, self.dim)
                q = _lib.nt_rope(_lib.nt_seq_linear(wq, xn, CTX), CTX, self.head_dim)
                k = _lib.nt_rope(_lib.nt_seq_linear(wk, xn, CTX), CTX, self.head_dim)
                v = _lib.nt_seq_linear(wv, xn, CTX)
                attn = _lib.nt_mh_causal_attention(q, k, v, CTX, self.head_dim)
                h = _lib.nt_add(h, _lib.nt_seq_linear(wo, attn, CTX))
                xn = _lib.nt_seq_rmsnorm(h, rms2, CTX, self.dim)
                gate = _lib.nt_silu(_lib.nt_seq_linear(wg, xn, CTX))
                up = _lib.nt_seq_linear(wu, xn, CTX)
                h = _lib.nt_add(h, _lib.nt_seq_linear(wd, _lib.nt_mul(gate, up), CTX))

            rmsf=tape_ids[pi]; pi+=1; head_i=tape_ids[pi]; pi+=1
            hf = _lib.nt_seq_rmsnorm(h, rmsf, CTX, self.dim)
            logits_idx = _lib.nt_seq_linear(head_i, hf, CTX)

            # Read last-position logits from tape
            tape_ptr = _lib.nt_tape_get()
            entry_size = ctypes.sizeof(_NtTapeEntry)
            tape_addr = ctypes.cast(tape_ptr, ctypes.c_void_p).value
            logits_entry = ctypes.cast(
                tape_addr + logits_idx * entry_size,
                ctypes.POINTER(_NtTapeEntry)
            ).contents
            logits_t = ctypes.cast(logits_entry.output, ctypes.POINTER(_NtTensor)).contents
            offset = (CTX - 1) * self.vocab_size
            raw_logits = [logits_t.data[offset + i] / temperature for i in range(self.vocab_size)]

            if top_k > 0 and top_k < self.vocab_size:
                sorted_vals = sorted(raw_logits, reverse=True)
                threshold = sorted_vals[min(top_k - 1, len(sorted_vals) - 1)]
                raw_logits = [v if v >= threshold else -1e30 for v in raw_logits]

            probs = softmax(raw_logits)
            next_id = multinomial(probs)
            _lib.nt_tape_clear()
            ctx.append(next_id)

        return ctx[len(token_ids):]

    def save_weights(self, path):
        params = self.param_list()
        n = len(params)
        arr = (ctypes.c_void_p * n)(*[p._ptr for p in params])
        _lib.nt_save(path.encode(), arr, n)

    def load_weights(self, path):
        if not os.path.exists(path):
            return False
        n_loaded = ctypes.c_int(0)
        loaded = _lib.nt_load(path.encode(), ctypes.byref(n_loaded))
        if not loaded:
            return False
        params = self.param_list()
        for i in range(min(n_loaded.value, len(params))):
            src = _get_tensor_struct(loaded[i])
            dst = _get_tensor_struct(params[i]._ptr)
            if src.len == dst.len:
                ctypes.memmove(dst.data, src.data, dst.len * 4)
            _lib.nt_tensor_free(loaded[i])
        return True
