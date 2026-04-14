import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Numerically stable softmax: subtract max before exponentiation.
        shifted = Z - np.max(Z, axis=self.dim, keepdims=True)
        exp_shifted = np.exp(shifted)
        self.A = exp_shifted / np.sum(exp_shifted, axis=self.dim, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        # Jacobian-vector product for softmax in a vectorized form.
        dot = np.sum(dLdA * self.A, axis=self.dim, keepdims=True)
        dLdZ = self.A * (dLdA - dot)
        return dLdZ
 

    