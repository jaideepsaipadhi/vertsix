from __future__ import annotations
import json
import random

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

import numpy as np


class SixVertexSampler:
    def __init__(self, n: int, c_up: float = 1.0, c_down: float = 1.0,
                 device: str | None = None, seed: int | None = None):
        self.n = n
        self.c_up = float(c_up)
        self.c_down = float(c_down)
        self.p_up = self.c_up / (self.c_up + self.c_down)

        self.use_torch = _HAS_TORCH
        if self.use_torch:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            if seed is not None:
                torch.manual_seed(seed)
        else:
            self.device = "cpu (numpy fallback -- install torch for GPU support)"
            if seed is not None:
                np.random.seed(seed)

        self.H = self._init_height()

        i_idx, j_idx = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        interior = (i_idx > 0) & (i_idx < n) & (j_idx > 0) & (j_idx < n)
        parity = (i_idx + j_idx) % 2
        self._masks_np = [interior & (parity == 0), interior & (parity == 1)]
        if self.use_torch:
            self._masks = [torch.from_numpy(m).to(self.device) for m in self._masks_np]

    @staticmethod
    def extremal_height(n: int, kind: str):
        i = np.arange(n + 1).reshape(-1, 1).astype(np.float32)
        j = np.arange(n + 1).reshape(1, -1).astype(np.float32)
        corners = [(0, 0, 0), (n, 0, n), (0, n, n), (n, n, 0)]
        if kind == "lo":
            H = np.full((n + 1, n + 1), -1e9, dtype=np.float32)
            for a, b, hv in corners:
                dist = np.abs(i - a) + np.abs(j - b)
                H = np.maximum(H, hv - dist)
        else:
            H = np.full((n + 1, n + 1), 1e9, dtype=np.float32)
            for a, b, hv in corners:
                dist = np.abs(i - a) + np.abs(j - b)
                H = np.minimum(H, hv + dist)
        return H

    def _init_height(self):
        H = SixVertexSampler.extremal_height(self.n, "lo")
        if self.use_torch:
            import torch
            return torch.from_numpy(H).to(self.device)
        return H

    def step(self, sweeps: int = 1):
        if self.use_torch:
            self._step_torch(sweeps)
        else:
            self._step_numpy(sweeps)

    def _step_torch(self, sweeps):
        import torch
        H = self.H
        n = self.n
        for _ in range(sweeps):
            for mask in self._masks:
                N = torch.zeros_like(H); S = torch.zeros_like(H)
                E = torch.zeros_like(H); W = torch.zeros_like(H)
                N[1:, :] = H[:-1, :]
                S[:-1, :] = H[1:, :]
                E[:, :-1] = H[:, 1:]
                W[:, 1:] = H[:, :-1]
                same = (N == S) & (S == E) & (E == W)
                is_extremum = mask & same
                rnd = torch.rand(H.shape, device=H.device)
                target = torch.where(rnd < self.p_up, N + 1, N - 1)
                H = torch.where(is_extremum, target, H)
        self.H = H

    def _step_numpy(self, sweeps):
        H = self.H
        n = self.n
        for _ in range(sweeps):
            for mask in self._masks_np:
                N = np.zeros_like(H); S = np.zeros_like(H)
                E = np.zeros_like(H); W = np.zeros_like(H)
                N[1:, :] = H[:-1, :]
                S[:-1, :] = H[1:, :]
                E[:, :-1] = H[:, 1:]
                W[:, 1:] = H[:, :-1]
                same = (N == S) & (S == E) & (E == W)
                is_extremum = mask & same
                rnd = np.random.rand(*H.shape)
                target = np.where(rnd < self.p_up, N + 1, N - 1)
                H = np.where(is_extremum, target, H)
        self.H = H

    def height_array(self):
        if self.use_torch:
            return self.H.detach().cpu().numpy()
        return self.H

    def active_mask(self):
        H = self.height_array()
        N = np.zeros_like(H); S = np.zeros_like(H)
        E = np.zeros_like(H); W = np.zeros_like(H)
        N[1:, :] = H[:-1, :]
        S[:-1, :] = H[1:, :]
        E[:, :-1] = H[:, 1:]
        W[:, 1:] = H[:, :-1]
        same = (N == S) & (S == E) & (E == W)
        n = self.n
        i_idx, j_idx = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        interior = (i_idx > 0) & (i_idx < n) & (j_idx > 0) & (j_idx < n)
        return interior & same

    def to_json(self):
        Hn = self.height_array()
        active = self.active_mask()
        return json.dumps({
            "n": self.n,
            "height": Hn.astype(int).tolist(),
            "active": active.astype(int).tolist(),
            "min": float(Hn.min()),
            "max": float(Hn.max()),
        })
