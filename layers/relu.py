from .base_layer import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        self.cache = None

    def forward(self, input_x):
        out = np.maximum(0, input_x)
        self.cache = input_x
        return out

    def backward(self, dout):
        x = self.cache
        dx = dout.copy()
        dx[x <= 0] = 0
        return dx
