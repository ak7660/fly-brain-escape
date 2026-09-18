"""Float64 reference simulator over exported (folded) arrays; the parity oracle for torch and web/model.js."""
import numpy as np
import scipy.sparse as sp

from flybrain import config


class NumpySim:
    def __init__(self, arrays, dtype=np.float64):
        a = arrays
        self.n = len(a["bias"])
        self.W = sp.csr_matrix((a["val"].astype(dtype), a["col"].astype(np.int64), a["row_ptr"].astype(np.int64)),
                               shape=(self.n, self.n))
        self.bias = a["bias"].astype(dtype)
        self.alpha = a["alpha"].astype(dtype)
        self.input_idx = a["input_idx"].astype(np.int64)
        self.input_scale = a["input_gain"].astype(dtype)[a["input_kind"].astype(np.int64)]
        self.dn_idx = a["dn_idx"].astype(np.int64)
        self.W_out = a["W_out"].astype(dtype).reshape(-1, len(self.dn_idx))
        self.b_out = a["b_out"].astype(dtype)
        self.dtype = dtype

    def frame(self, r, drive_f):
        u = np.zeros(self.n, dtype=self.dtype)
        u[self.input_idx] = self.input_scale * drive_f
        for _ in range(config.SUBSTEPS):
            current = self.W @ r + self.bias + u
            r = (1 - self.alpha) * r + self.alpha * np.clip(current, 0.0, config.R_MAX)
        return r, self.W_out @ r[self.dn_idx] + self.b_out

    def run(self, drive, r0=None, record=False):
        r = np.zeros(self.n, dtype=self.dtype) if r0 is None else np.asarray(r0, dtype=self.dtype).copy()
        logits = np.zeros((len(drive), len(self.b_out)), dtype=self.dtype)
        rates = np.zeros((len(drive), self.n), dtype=self.dtype) if record else None
        for f, d in enumerate(drive):
            r, logits[f] = self.frame(r, d)
            if record:
                rates[f] = r
        out = {"logits": logits, "r": r}
        if record:
            out["rates"] = rates
        return out

    def steady_state(self, frames=config.WARMUP_FRAMES):
        return self.run(np.zeros((frames, len(self.input_idx))))["r"]
