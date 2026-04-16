"""
notorch_nn.py — Python neural network API backed by libnotorch (ctypes)

Drop-in replacement for torch.nn / torch.nn.functional / torch.optim.
No PyTorch. No numpy. Just ctypes to the C library that started it all.

Usage:
    from ariannamethod.notorch_nn import *
    # or: from ariannamethod import Tensor, Linear, Module, ...
"""

import ctypes
import ctypes.util
import os
import math
import struct
import random

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD libnotorch
# ═══════════════════════════════════════════════════════════════════════════════

_dir = os.path.dirname(os.path.abspath(__file__))

# Try platform-specific extensions
for ext in ['.dylib', '.so', '.dll']:
    _libpath = os.path.join(_dir, f'libnotorch{ext}')
    if os.path.exists(_libpath):
        break
else:
    # Try building it
    _src = os.path.join(_dir, 'notorch.c')
    _libpath = os.path.join(_dir, 'libnotorch.so')
    if os.path.exists(_src):
        import subprocess
        subprocess.run(['cc', '-O2', '-std=c11', '-shared', '-fPIC',
                       '-o', _libpath, _src, '-lm'], check=True)

_lib = ctypes.CDLL(_libpath)

# ═══════════════════════════════════════════════════════════════════════════════
# C FUNCTION SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════════

# Tensor
_lib.nt_tensor_new.restype = ctypes.c_void_p
_lib.nt_tensor_new.argtypes = [ctypes.c_int]
_lib.nt_tensor_new2d.restype = ctypes.c_void_p
_lib.nt_tensor_new2d.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.nt_tensor_free.argtypes = [ctypes.c_void_p]
_lib.nt_tensor_xavier.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.nt_tensor_fill.argtypes = [ctypes.c_void_p, ctypes.c_float]

# Tape
_lib.nt_tape_start.restype = None
_lib.nt_tape_clear.restype = None
_lib.nt_tape_param.restype = ctypes.c_int
_lib.nt_tape_param.argtypes = [ctypes.c_void_p]
_lib.nt_tape_no_decay.argtypes = [ctypes.c_int]
_lib.nt_tape_backward.argtypes = [ctypes.c_int]
_lib.nt_tape_clip_grads.restype = ctypes.c_float
_lib.nt_tape_clip_grads.argtypes = [ctypes.c_float]
_lib.nt_tape_chuck_step.argtypes = [ctypes.c_float, ctypes.c_float]
_lib.nt_train_mode.argtypes = [ctypes.c_int]

# Ops
_lib.nt_seq_embedding.restype = ctypes.c_int
_lib.nt_seq_embedding.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_seq_rmsnorm.restype = ctypes.c_int
_lib.nt_seq_rmsnorm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_seq_linear.restype = ctypes.c_int
_lib.nt_seq_linear.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_mh_causal_attention.restype = ctypes.c_int
_lib.nt_mh_causal_attention.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_rope.restype = ctypes.c_int
_lib.nt_rope.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_silu.restype = ctypes.c_int
_lib.nt_silu.argtypes = [ctypes.c_int]
_lib.nt_add.restype = ctypes.c_int
_lib.nt_add.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.nt_mul.restype = ctypes.c_int
_lib.nt_mul.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.nt_seq_cross_entropy.restype = ctypes.c_int
_lib.nt_seq_cross_entropy.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

# Record
_lib.nt_tape_record.restype = ctypes.c_int
_lib.nt_tape_record.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]

# Save/Load
_lib.nt_save.restype = ctypes.c_int
_lib.nt_save.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
_lib.nt_load.restype = ctypes.POINTER(ctypes.c_void_p)
_lib.nt_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]

# Schedule
_lib.nt_seed.argtypes = [ctypes.c_uint64]

# ═══════════════════════════════════════════════════════════════════════════════
# TENSOR WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

# nt_tensor struct layout: float* data, int ndim, int shape[8], int stride[8], int len, int refcount
# We need to access data pointer and len

class _NtTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int * 8),
        ("stride", ctypes.c_int * 8),
        ("len", ctypes.c_int),
        ("refcount", ctypes.c_int),
    ]


def _get_tensor_struct(ptr):
    """Cast void* to _NtTensor*"""
    return ctypes.cast(ptr, ctypes.POINTER(_NtTensor)).contents


class Tensor:
    """Wrapper around nt_tensor*"""

    def __init__(self, ptr, owns=True):
        self._ptr = ptr
        self._owns = owns

    @staticmethod
    def zeros(size):
        if isinstance(size, int):
            ptr = _lib.nt_tensor_new(size)
        elif len(size) == 1:
            ptr = _lib.nt_tensor_new(size[0])
        else:
            ptr = _lib.nt_tensor_new2d(size[0], size[1])
        return Tensor(ptr)

    @staticmethod
    def ones(size):
        t = Tensor.zeros(size) if isinstance(size, int) else Tensor.zeros(size)
        _lib.nt_tensor_fill(t._ptr, 1.0)
        return t

    @property
    def data_ptr(self):
        return _get_tensor_struct(self._ptr).data

    @property
    def numel(self):
        return _get_tensor_struct(self._ptr).len

    @property
    def shape(self):
        s = _get_tensor_struct(self._ptr)
        return tuple(s.shape[i] for i in range(s.ndim))

    def fill_(self, val):
        _lib.nt_tensor_fill(self._ptr, ctypes.c_float(val))
        return self

    def xavier_(self, fan_in, fan_out):
        _lib.nt_tensor_xavier(self._ptr, fan_in, fan_out)
        return self

    def set_data(self, flat_list):
        """Set tensor data from a flat list of floats"""
        s = _get_tensor_struct(self._ptr)
        for i in range(min(len(flat_list), s.len)):
            s.data[i] = flat_list[i]

    def get_data(self):
        """Get tensor data as flat list"""
        s = _get_tensor_struct(self._ptr)
        return [s.data[i] for i in range(s.len)]

    def __del__(self):
        if self._owns and self._ptr:
            _lib.nt_tensor_free(self._ptr)


class Parameter(Tensor):
    """Trainable parameter — registers on tape"""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class Module:
    """Base module — like nn.Module but backed by notorch"""

    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._training = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault('_parameters', {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault('_modules', {})[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        """Yield all parameters recursively"""
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def train(self, mode=True):
        self._training = mode
        _lib.nt_train_mode(1 if mode else 0)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter.zeros((out_features, in_features))
        self.weight.xavier_(in_features, out_features)
        # bias not supported in notorch seq_linear (matches our models)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter.zeros((num_embeddings, embedding_dim))
        self.weight.xavier_(num_embeddings, embedding_dim)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = Parameter.ones(dim)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTIONAL API (replaces torch.nn.functional)
# ═══════════════════════════════════════════════════════════════════════════════

def softmax(logits_list, dim=-1):
    """Softmax over a list of floats"""
    mx = max(logits_list)
    exps = [math.exp(x - mx) for x in logits_list]
    s = sum(exps)
    return [e / s for e in exps]


def silu(x):
    """SiLU activation for a single float"""
    return x / (1 + math.exp(-x)) if abs(x) < 80 else (x if x > 0 else 0)


def cross_entropy(logits_list, target):
    """Cross entropy loss from logits list and target index"""
    mx = max(logits_list)
    lse = math.log(sum(math.exp(x - mx) for x in logits_list)) + mx
    return -(logits_list[target] - lse)


def multinomial(probs):
    """Sample one index from probability distribution"""
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if cum >= r:
            return i
    return len(probs) - 1


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE — runs forward/backward through notorch tape
# ═══════════════════════════════════════════════════════════════════════════════

class NotorchEngine:
    """
    Manages the notorch tape for a full forward-backward-update cycle.
    This is the bridge between Python Module API and C notorch tape.
    """

    def __init__(self, model, lr=3e-4):
        self.model = model
        self.lr = lr
        self.params = list(model.parameters())

    def forward(self, token_ids, target_ids, ctx_len, vocab_size, dim, n_layers, n_heads, head_dim):
        """
        Run full forward pass through notorch tape.
        token_ids: list of int, length ctx_len
        target_ids: list of int, length ctx_len
        Returns loss value (float)
        """
        _lib.nt_tape_start()

        # Register all parameters on tape
        tape_ids = []
        for p in self.params:
            idx = _lib.nt_tape_param(p._ptr)
            tape_ids.append(idx)

        # Build token/target tensors
        tok_t = Tensor.zeros(ctx_len)
        tgt_t = Tensor.zeros(ctx_len)
        tok_t.set_data([float(x) for x in token_ids])
        tgt_t.set_data([float(x) for x in target_ids])

        tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))  # NT_OP_NONE=0
        tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tok_t._owns = False  # tape owns it now
        tgt_t._owns = False

        # Forward: embedding → blocks → rmsnorm → lm_head → cross_entropy
        # Parameter order: wte, [rms1, wq, wk, wv, wo, rms2, w_gate, w_up, w_down] × L, rms_f, head
        pi = 0
        wte_i = tape_ids[pi]; pi += 1

        h = _lib.nt_seq_embedding(wte_i, -1, tok_idx, ctx_len, dim)

        for l in range(n_layers):
            rms1_i = tape_ids[pi]; pi += 1
            wq_i = tape_ids[pi]; pi += 1
            wk_i = tape_ids[pi]; pi += 1
            wv_i = tape_ids[pi]; pi += 1
            wo_i = tape_ids[pi]; pi += 1
            rms2_i = tape_ids[pi]; pi += 1
            wgate_i = tape_ids[pi]; pi += 1
            wup_i = tape_ids[pi]; pi += 1
            wdown_i = tape_ids[pi]; pi += 1

            xn = _lib.nt_seq_rmsnorm(h, rms1_i, ctx_len, dim)
            q = _lib.nt_seq_linear(wq_i, xn, ctx_len)
            k = _lib.nt_seq_linear(wk_i, xn, ctx_len)
            v = _lib.nt_seq_linear(wv_i, xn, ctx_len)
            q = _lib.nt_rope(q, ctx_len, head_dim)
            k = _lib.nt_rope(k, ctx_len, head_dim)
            attn = _lib.nt_mh_causal_attention(q, k, v, ctx_len, head_dim)
            proj = _lib.nt_seq_linear(wo_i, attn, ctx_len)
            h = _lib.nt_add(h, proj)

            xn = _lib.nt_seq_rmsnorm(h, rms2_i, ctx_len, dim)
            gate = _lib.nt_silu(_lib.nt_seq_linear(wgate_i, xn, ctx_len))
            up = _lib.nt_seq_linear(wup_i, xn, ctx_len)
            down = _lib.nt_seq_linear(wdown_i, _lib.nt_mul(gate, up), ctx_len)
            h = _lib.nt_add(h, down)

        rmsf_i = tape_ids[pi]; pi += 1
        head_i = tape_ids[pi]; pi += 1

        hf = _lib.nt_seq_rmsnorm(h, rmsf_i, ctx_len, dim)
        logits = _lib.nt_seq_linear(head_i, hf, ctx_len)
        loss_idx = _lib.nt_seq_cross_entropy(logits, tgt_idx, ctx_len, vocab_size)

        # Get loss value
        tape_ptr = _lib.nt_tape_get()
        # nt_tape_entry at loss_idx: output->data[0]
        # We need to read the loss from the tape
        loss_tensor_ptr = ctypes.cast(tape_ptr, ctypes.POINTER(_NtTapeEntry))[loss_idx].output
        loss_val = ctypes.cast(loss_tensor_ptr, ctypes.POINTER(_NtTensor)).contents.data[0]

        return loss_idx, loss_val

    def backward_and_step(self, loss_idx, loss_val):
        """Run backward pass and Chuck optimizer step"""
        _lib.nt_tape_backward(loss_idx)
        _lib.nt_tape_clip_grads(ctypes.c_float(1.0))
        _lib.nt_tape_chuck_step(ctypes.c_float(self.lr), ctypes.c_float(loss_val))
        _lib.nt_tape_clear()

    def save(self, path):
        """Save all parameters"""
        n = len(self.params)
        arr = (ctypes.c_void_p * n)(*[p._ptr for p in self.params])
        _lib.nt_save(path.encode(), arr, n)

    def load(self, path):
        """Load parameters from file"""
        n_loaded = ctypes.c_int(0)
        loaded = _lib.nt_load(path.encode(), ctypes.byref(n_loaded))
        if not loaded:
            return False
        for i in range(min(n_loaded.value, len(self.params))):
            src = _get_tensor_struct(loaded[i])
            dst = _get_tensor_struct(self.params[i]._ptr)
            ctypes.memmove(dst.data, src.data, dst.len * 4)
            _lib.nt_tensor_free(loaded[i])
        return True


# Tape entry struct for reading loss
class _NtTapeEntry(ctypes.Structure):
    _fields_ = [
        ("output", ctypes.c_void_p),
        ("grad", ctypes.c_void_p),
        ("op", ctypes.c_int),
        ("parent1", ctypes.c_int),
        ("parent2", ctypes.c_int),
        ("parent3", ctypes.c_int),
        ("aux", ctypes.c_float),
        ("aux2", ctypes.c_float),
        ("aux3", ctypes.c_float),
        ("aux4", ctypes.c_float),
        ("is_param", ctypes.c_int),
        ("no_decay", ctypes.c_int),
    ]

# Get tape pointer
_lib.nt_tape_get.restype = ctypes.c_void_p


def seed(s):
    _lib.nt_seed(ctypes.c_uint64(s))


__all__ = [
    'Tensor', 'Parameter', 'Module', 'Linear', 'Embedding', 'RMSNorm',
    'NotorchEngine', 'softmax', 'silu', 'cross_entropy', 'multinomial', 'seed',
    '_lib',
]
