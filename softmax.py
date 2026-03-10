from .base_layer import BaseLayer
import numpy as np


class Softmax(BaseLayer):
    """Implement the softmax layer
    Output of the softmax passes to the Cross entropy loss function
    The gradient for the softmax is calculated in the loss_func backward pass
    Thus the backward function here should be an empty pass"""

    def __init__(self):
        pass

    def forward(self, input_x: np.ndarray):
        (
            N,
            C,
        ) = (
            input_x.shape
        )
        # ---- Numerically stable softmax ----
        # Subtract max per row to prevent overflow
        shifted_logits = input_x - np.max(input_x, axis=1, keepdims=True)

        exp_scores = np.exp(shifted_logits)

        # Normalize each row
        scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return scores

    def backward(self, dout):
        # Nothing to do here, pass. The gradient are calculated in the cross entropy loss backward function itself
        return dout

