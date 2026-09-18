"""Connectome-constrained rate model (torch).

W[post, pre] = exp(log_gain + post_scale[type(post)]) · w0 · exp(pre_scale[type(pre)]) keeps the sparse
matrix W0 fixed and puts every trainable gain on the diagonal, so backprop needs only W0ᵀ (≈20× faster
than per-synapse gradients on this circuit). See CLAUDE.md "Model contract".
"""
import math
import warnings

import numpy as np
import torch

from flybrain import config

INPUT_KIND = {name: i for i, name in enumerate(config.INPUT_KINDS)}
LOG_GAIN_MIN, LOG_GAIN_MAX = -6.0, 6.0  # per-factor exponent bounds; exp(6)·exp(6 + log 64) stays far below float32 max
LOG_POST_GAIN_MAX = LOG_GAIN_MAX + math.log(64.0)  # the post factor also carries the global log_gain


class _FixedSpMV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, Wt):
        ctx.Wt = Wt
        return W @ x

    @staticmethod
    def backward(ctx, grad):
        return ctx.Wt @ grad, None, None


def _csr_numpy(rows, cols, vals, n):
    order = np.lexsort((cols, rows))
    r, c, v = rows[order], cols[order], vals[order]
    return np.searchsorted(r, np.arange(n + 1)), c, v


def _csr_torch(rows, cols, vals, n, dtype):
    row_ptr, c, v = _csr_numpy(rows, cols, vals, n)
    with warnings.catch_warnings():  # torch flags sparse CSR as beta on every construction
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        return torch.sparse_csr_tensor(torch.as_tensor(row_ptr, dtype=torch.int64), torch.as_tensor(c, dtype=torch.int64),
                                       torch.as_tensor(v, dtype=dtype), (n, n), check_invariants=False)


class RateModel(torch.nn.Module):
    """CPU-only rate model with one fixed floating dtype (`dtype`) for parameters, W0 and state.

    `simulate` (also `forward`) casts drive and r0 to that dtype. W0/W0t are plain CSR tensors, not buffers,
    so `.to(device/dtype)` does not move them: build a new model instead.
    """

    def __init__(self, circuit, gain=1.0, bias=0.0, tau_ms=config.TAU_INIT_MS, readout_seed=0, dtype=torch.float32):
        super().__init__()
        if not config.TAU_MIN_MS < tau_ms < config.TAU_MAX_MS:
            raise ValueError(f"tau_ms must lie strictly between TAU_MIN_MS={config.TAU_MIN_MS} and "
                             f"TAU_MAX_MS={config.TAU_MAX_MS}, got {tau_ms}")
        self.n, self.dtype = len(circuit.body), dtype
        types, type_id = np.unique(np.asarray(circuit.type).astype(str), return_inverse=True)
        self.type_names = types.tolist()
        inputs = np.flatnonzero(circuit.role == 0)
        dns = np.flatnonzero(circuit.role == 2)
        self.register_buffer("type_id", torch.as_tensor(type_id, dtype=torch.long))
        self.register_buffer("input_idx", torch.as_tensor(inputs, dtype=torch.long))
        self.register_buffer("input_kind", torch.as_tensor([INPUT_KIND[str(t)] for t in circuit.type[inputs]],
                                                           dtype=torch.long))
        self.register_buffer("input_side", torch.as_tensor(circuit.side[inputs].astype(np.int64)))
        self.register_buffer("dn_idx", torch.as_tensor(dns, dtype=torch.long))
        self._pre = circuit.pre.astype(np.int64)
        self._post = circuit.post.astype(np.int64)
        self._w0 = circuit.w0.astype(np.float64)
        self.W0 = _csr_torch(self._post, self._pre, self._w0, self.n, dtype)
        self.W0t = _csr_torch(self._pre, self._post, self._w0, self.n, dtype)

        n_types = len(types)
        param = lambda x: torch.nn.Parameter(torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dtype))
        frac = (tau_ms - config.TAU_MIN_MS) / (config.TAU_MAX_MS - config.TAU_MIN_MS)
        self.log_gain = param(math.log(gain))
        self.pre_scale = param(np.zeros(n_types))
        self.post_scale = param(np.zeros(n_types))
        self.type_bias = param(np.full(n_types, bias))
        self.tau_raw = param(np.full(n_types, math.log(frac / (1 - frac))))
        self.log_input_gain = param(np.zeros(len(config.INPUT_KINDS)))
        gen = torch.Generator().manual_seed(readout_seed)
        self.W_out = torch.nn.Parameter(0.01 * torch.randn(len(config.CLASSES), len(dns), generator=gen, dtype=dtype))
        self.b_out = param(np.zeros(len(config.CLASSES)))

    # per-neuron views of the type-level parameters
    def neuron_bias(self):
        return self.type_bias[self.type_id]

    def neuron_alpha(self):
        tau = config.TAU_MIN_MS + (config.TAU_MAX_MS - config.TAU_MIN_MS) * torch.sigmoid(self.tau_raw[self.type_id])
        return config.DT_SUB * 1000.0 / tau

    def pre_gain(self):
        return torch.exp(torch.clamp(self.pre_scale[self.type_id], LOG_GAIN_MIN, LOG_GAIN_MAX))

    def post_gain(self):
        return torch.exp(torch.clamp(self.log_gain + self.post_scale[self.type_id], LOG_GAIN_MIN, LOG_POST_GAIN_MAX))

    def input_gain(self):
        return torch.exp(self.log_input_gain)[self.input_kind]

    def simulate(self, drive, r0=None):
        drive = drive.to(self.dtype)
        batch, frames, _ = drive.shape
        r = (torch.zeros(self.n, batch, dtype=self.dtype) if r0 is None
             else r0.to(self.dtype).reshape(self.n, 1).repeat(1, batch))
        bias, alpha = self.neuron_bias()[:, None], self.neuron_alpha()[:, None]
        pre_g, post_g = self.pre_gain()[:, None], self.post_gain()[:, None]
        in_g = self.input_gain()[:, None]
        logits = []
        for f in range(frames):
            u = torch.zeros(self.n, batch, dtype=self.dtype).index_add(0, self.input_idx, in_g * drive[:, f].T)
            for _ in range(config.SUBSTEPS):
                current = post_g * _FixedSpMV.apply(pre_g * r, self.W0, self.W0t) + bias + u
                r = (1 - alpha) * r + alpha * torch.clamp(current, min=0.0, max=config.R_MAX)
            logits.append(self.W_out @ r[self.dn_idx] + self.b_out[:, None])
        return torch.stack(logits, 0).permute(2, 0, 1), r.T

    forward = simulate

    @torch.no_grad()
    def export_arrays(self, dtype=np.float32):
        pre_g = self.pre_gain().double().numpy()
        post_g = self.post_gain().double().numpy()
        vals = post_g[self._post] * self._w0 * pre_g[self._pre]
        row_ptr, col, val = _csr_numpy(self._post, self._pre, vals, self.n)
        u32 = lambda x: np.asarray(x, dtype=np.uint32)
        f = lambda x: np.asarray(x.double().numpy() if torch.is_tensor(x) else x, dtype=dtype)
        out = {"row_ptr": u32(row_ptr), "col": u32(col), "val": f(val),
                "bias": f(self.neuron_bias()), "alpha": f(self.neuron_alpha()),
                "input_idx": u32(self.input_idx), "input_kind": u32(self.input_kind),
                "input_side": u32(self.input_side), "input_gain": f(torch.exp(self.log_input_gain)),
                "dn_idx": u32(self.dn_idx), "W_out": f(self.W_out), "b_out": f(self.b_out)}
        bad = [k for k, v in out.items() if not np.isfinite(v).all()]
        if bad:
            raise ValueError(f"export_arrays: non-finite values in {bad}")
        return out
