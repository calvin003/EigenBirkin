from __future__ import annotations

import numpy as np

from .base_layer import BaseLayer


class Conv2D(BaseLayer):
    """A simple 2D convolution layer (N, C, H, W) -> (N, F, H_out, W_out)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.w = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels, dtype=np.float32)

        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.cache = None

    def forward(self, input_x: np.ndarray) -> np.ndarray:
        n, c, h, w = input_x.shape
        k = self.kernel_size
        p = self.padding
        s = self.stride

        x_padded = np.pad(input_x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

        h_out = (h + 2 * p - k) // s + 1
        w_out = (w + 2 * p - k) // s + 1
        out = np.zeros((n, self.out_channels, h_out, w_out), dtype=input_x.dtype)

        for i in range(h_out):
            hs = i * s
            for j in range(w_out):
                ws = j * s
                patch = x_padded[:, :, hs : hs + k, ws : ws + k]
                out[:, :, i, j] = np.tensordot(patch, self.w, axes=([1, 2, 3], [1, 2, 3])) + self.b

        self.cache = (input_x, x_padded)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        input_x, x_padded = self.cache
        n, _, h, w = input_x.shape
        k = self.kernel_size
        p = self.padding
        s = self.stride

        _, _, h_out, w_out = dout.shape

        self.dw.fill(0.0)
        self.db = dout.sum(axis=(0, 2, 3))
        dx_padded = np.zeros_like(x_padded)

        for i in range(h_out):
            hs = i * s
            for j in range(w_out):
                ws = j * s
                patch = x_padded[:, :, hs : hs + k, ws : ws + k]
                for f in range(self.out_channels):
                    self.dw[f] += np.sum(patch * dout[:, f : f + 1, i : i + 1, j : j + 1], axis=0)
                for sample_idx in range(n):
                    dx_padded[sample_idx, :, hs : hs + k, ws : ws + k] += np.sum(
                        self.w * dout[sample_idx, :, i, j][:, None, None, None], axis=0
                    )

        if p == 0:
            return dx_padded
        return dx_padded[:, :, p:-p, p:-p]

    def zero_grad(self):
        self.dw.fill(0.0)
        self.db.fill(0.0)

    @property
    def parameters(self):
        return [self.w, self.b]

    @property
    def grads(self):
        return [self.dw, self.db]
