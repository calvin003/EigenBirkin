from __future__ import annotations

import numpy as np

from .base_layer import BaseLayer


class MaxPool2D(BaseLayer):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None

    def forward(self, input_x: np.ndarray) -> np.ndarray:
        n, c, h, w = input_x.shape
        k = self.kernel_size
        s = self.stride

        h_out = (h - k) // s + 1
        w_out = (w - k) // s + 1

        out = np.zeros((n, c, h_out, w_out), dtype=input_x.dtype)
        max_indices = np.zeros((n, c, h_out, w_out, 2), dtype=np.int64)

        for i in range(h_out):
            hs = i * s
            for j in range(w_out):
                ws = j * s
                window = input_x[:, :, hs : hs + k, ws : ws + k]
                window_flat = window.reshape(n, c, -1)
                max_pos = np.argmax(window_flat, axis=2)
                out[:, :, i, j] = np.max(window_flat, axis=2)
                max_indices[:, :, i, j, 0] = max_pos // k
                max_indices[:, :, i, j, 1] = max_pos % k

        self.cache = (input_x.shape, max_indices)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        input_shape, max_indices = self.cache
        n, c, h, w = input_shape
        k = self.kernel_size
        s = self.stride
        _, _, h_out, w_out = dout.shape

        dx = np.zeros(input_shape, dtype=dout.dtype)

        for i in range(h_out):
            hs = i * s
            for j in range(w_out):
                ws = j * s
                row_idx = max_indices[:, :, i, j, 0]
                col_idx = max_indices[:, :, i, j, 1]
                for sample_idx in range(n):
                    for ch in range(c):
                        dx[sample_idx, ch, hs + row_idx[sample_idx, ch], ws + col_idx[sample_idx, ch]] += dout[
                            sample_idx, ch, i, j
                        ]

        return dx

    def zero_grad(self):
        return
