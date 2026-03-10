from .base_layer import BaseLayer
import numpy as np

class Sequential(BaseLayer):
    def __init__(self, modules=None):
        self._modules = modules if modules is not None else []

    def forward(self, input_x):
        inter_x = input_x
        for module in self._modules:
            inter_x = module(inter_x)
        return inter_x

    def backward(self, dx):
        for module in reversed(self._modules):
            dx = module.backward(dx)
        return dx   # important for chaining

    def zero_grad(self):
        for module in self._modules:
            module.zero_grad()

    def predict(self, input_x):
        out = self.forward(input_x)
        return np.argmax(out, axis=-1)
