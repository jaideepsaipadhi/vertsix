from __future__ import annotations
import json
import base64

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

import numpy as np


class SixVertexSampler:
    def __init__(self, n: int, c_up: float = 1.0, c_down: float = 1.0,
                 a1: float = 1.0, a2: float = 1.0,
                 b1: float = 1.0, b2: float = 1.0,
                 device: str | None = None, seed: int | None = None):
        self.n = n
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.c1 = float(c_up)
        self.c2 = float(c_down)

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
        color = (i_idx % 2) * 2 + (j_idx % 2)
        self._masks_np = [interior & (color == c) for c in range(4)]
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

    def is_symmetric_regime(self) -> bool:
        return self.a1 == 1.0 and self.a2 == 1.0 and self.b1 == 1.0 and self.b2 == 1.0

    def step(self, sweeps: int = 1):
        if self.use_torch:
            self._step_torch(sweeps)
        else:
            self._step_numpy(sweeps)

    def _classify_face_np(self, tl, tr, bl, br):
        top = tr - tl
        bottom = br - bl
        left = bl - tl
        right = br - tr
        t = (top == 1)
        b = (bottom == 1)
        l = (left == 1)
        r = (right == 1)
        w = np.ones_like(top, dtype=np.float64)
        w = np.where((~l) & (~t) & (~b) & (~r), self.a1, w)
        w = np.where(l & t & b & r, self.a2, w)
        w = np.where(l & t & (~b) & (~r), self.b1, w)
        w = np.where((~l) & (~t) & b & r, self.b2, w)
        w = np.where((~l) & t & b & (~r), self.c1, w)
        w = np.where(l & (~t) & (~b) & r, self.c2, w)
        return w

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

                NW = np.zeros_like(H); NE = np.zeros_like(H)
                SW = np.zeros_like(H); SE = np.zeros_like(H)
                NW[1:, 1:] = H[:-1, :-1]
                NE[1:, :-1] = H[:-1, 1:]
                SW[:-1, 1:] = H[1:, :-1]
                SE[:-1, :-1] = H[1:, 1:]

                H_after = np.where(H == N + 1, N - 1, N + 1)

                before = (self._classify_face_np(NW, N, W, H)
                          * self._classify_face_np(N, NE, H, E)
                          * self._classify_face_np(W, H, SW, S)
                          * self._classify_face_np(H, E, S, SE))
                after = (self._classify_face_np(NW, N, W, H_after)
                         * self._classify_face_np(N, NE, H_after, E)
                         * self._classify_face_np(W, H_after, SW, S)
                         * self._classify_face_np(H_after, E, S, SE))

                ratio = np.where(is_extremum, after / np.maximum(before, 1e-300), 1.0)
                p_accept = ratio / (1.0 + ratio)
                rnd = np.random.rand(*H.shape)
                do_flip = is_extremum & (rnd < p_accept)
                H = np.where(do_flip, H_after, H)
        self.H = H

    def _classify_face_torch(self, tl, tr, bl, br):
        import torch
        top = tr - tl
        bottom = br - bl
        left = bl - tl
        right = br - tr
        t = (top == 1)
        b = (bottom == 1)
        l = (left == 1)
        r = (right == 1)
        w = torch.ones_like(top)
        w = torch.where((~l) & (~t) & (~b) & (~r), torch.tensor(self.a1, device=w.device, dtype=w.dtype), w)
        w = torch.where(l & t & b & r, torch.tensor(self.a2, device=w.device, dtype=w.dtype), w)
        w = torch.where(l & t & (~b) & (~r), torch.tensor(self.b1, device=w.device, dtype=w.dtype), w)
        w = torch.where((~l) & (~t) & b & r, torch.tensor(self.b2, device=w.device, dtype=w.dtype), w)
        w = torch.where((~l) & t & b & (~r), torch.tensor(self.c1, device=w.device, dtype=w.dtype), w)
        w = torch.where(l & (~t) & (~b) & r, torch.tensor(self.c2, device=w.device, dtype=w.dtype), w)
        return w

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

                NW = torch.zeros_like(H); NE = torch.zeros_like(H)
                SW = torch.zeros_like(H); SE = torch.zeros_like(H)
                NW[1:, 1:] = H[:-1, :-1]
                NE[1:, :-1] = H[:-1, 1:]
                SW[:-1, 1:] = H[1:, :-1]
                SE[:-1, :-1] = H[1:, 1:]

                H_after = torch.where(H == N + 1, N - 1, N + 1)

                before = (self._classify_face_torch(NW, N, W, H)
                          * self._classify_face_torch(N, NE, H, E)
                          * self._classify_face_torch(W, H, SW, S)
                          * self._classify_face_torch(H, E, S, SE))
                after = (self._classify_face_torch(NW, N, W, H_after)
                         * self._classify_face_torch(N, NE, H_after, E)
                         * self._classify_face_torch(W, H_after, SW, S)
                         * self._classify_face_torch(H_after, E, S, SE))

                ratio = torch.where(is_extremum, after / torch.clamp(before, min=1e-300), torch.ones_like(before))
                p_accept = ratio / (1.0 + ratio)
                rnd = torch.rand(H.shape, device=H.device)
                do_flip = is_extremum & (rnd < p_accept)
                H = torch.where(do_flip, H_after, H)
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

    def to_binary_frame(self):
        Hn = self.height_array().astype(np.int16)
        active = self.active_mask().astype(np.uint8)
        height_bytes = Hn.tobytes()
        active_bytes = active.tobytes()
        return {
            "n": self.n,
            "height_b64": base64.b64encode(height_bytes).decode("ascii"),
            "active_b64": base64.b64encode(active_bytes).decode("ascii"),
            "min": float(Hn.min()),
            "max": float(Hn.max()),
        }
