import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Cache input activation to be utilized in backward step
        self.A = A

        # Perform the affine projection across the trailing dimension
        Z = np.matmul(A, self.W.T) + self.b
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        # Compute the gradient w.r.t. the incoming input features
        self.dLdA = np.matmul(dLdZ, self.W)

        # Flatten leading dimensions to accumulate parameter gradients correctly
        A_2d = self.A.reshape(-1, self.A.shape[-1])
        dLdZ_2d = dLdZ.reshape(-1, dLdZ.shape[-1])

        # Accumulate gradients for weights and biases
        self.dLdW = np.matmul(dLdZ_2d.T, A_2d)
        self.dLdb = np.sum(dLdZ_2d, axis=0)
        
        return self.dLdA
