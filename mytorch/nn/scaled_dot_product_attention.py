import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask
        self.d_k = Q.shape[-1]
        
        # Derive unnormalized attention scores scaled by dimensionality
        scaled_dp = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(self.d_k)
        
        # Incorporate masking preceding softmax normalization
        if mask is not None:
            scaled_dp = np.where(mask, -self.eps, scaled_dp)

        self.scaled_dot_product = scaled_dp

        # Normalize across the relevant target sequence axis via Softmax
        self.attention_scores = self.softmax.forward(scaled_dp)

        # Extract features scaled by attention weights
        out = np.matmul(self.attention_scores, V)

        return out
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass

        # Gradients w.r.t the value representations
        grad_V = np.matmul(np.swapaxes(self.attention_scores, -1, -2), d_output)
        
        # Derive loss gradients prior to softmax
        grad_attention_score = np.matmul(d_output, np.swapaxes(self.V, -1, -2))
        grad_scaled_dp = self.softmax.backward(grad_attention_score)

        # Mask elements contribute zero to gradient signals
        if self.mask is not None:
            grad_scaled_dp = np.where(self.mask, 0.0, grad_scaled_dp)
        
        # Apply differentiation of scale factor
        grad_scaled_dp = grad_scaled_dp / np.sqrt(self.d_k)
        
        # Backpropagate through matrix multiplications for query and key
        grad_Q = np.matmul(grad_scaled_dp, self.K)
        grad_K = np.matmul(np.swapaxes(grad_scaled_dp, -1, -2), self.Q)
        
        return grad_Q, grad_K, grad_V

