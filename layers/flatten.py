from __future__ import annotations

import numpy as np

from .base_layer import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        self.input_shape = None

    def forward(self, input_x: np.ndarray) -> np.ndarray:
        self.input_shape = input_x.shape
        return input_x.reshape(input_x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self.input_shape)

    def zero_grad(self):
        return
