from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    """
    Multi-Head Attention Layer
    """ 
    def __init__(self, embed_dim, num_heads):
        """
        Initializer for Multi-Head Attention layer.
        :param embed_dim: Dimensionality of the input embeddings.
        :param num_heads: Amount of parallel attention heads.
        """
        if embed_dim % num_heads != 0:
            raise ValueError("The embed_dim variable must be perfectly divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Prepare the scaled dot-product attention core element
        self.attention = ScaledDotProductAttention()
        
        # Instantiate dense projections for Query, Key, Value, and Output
        # Mapping: embed_dim -> embed_dim
        self.q_proj   = Linear(embed_dim, embed_dim)
        self.k_proj   = Linear(embed_dim, embed_dim)
        self.v_proj   = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """
        
        # TODO: Implement forward pass

        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]
        
        # Project inputs
        q = self.q_proj.forward(query)
        k = self.k_proj.forward(key)
        v = self.v_proj.forward(value)

        # Reshape for multiple heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Combine padding and causal masks
        mask = self._merge_masks(key_padding_mask, attn_mask)

        # Apply attention
        attn_outputs = self.attention.forward(q, k, v, mask=mask)

        # Merge heads
        attn_output = self._concat_heads(attn_outputs)

        # Final projection
        output = self.out_proj.forward(attn_output)

        return output

    def backward(self, d_output):
        """
        Backward pass implementation for the multi-head attention.
        """

        # Project backwards through output dense layer
        grad_attn_out = self.out_proj.backward(d_output)

        # Restore head dimensionality for gradients
        grad_attn_out_split = self._split_heads(grad_attn_out)

        # Backward propagation across SDPA
        grad_q_split, grad_k_split, grad_v_split = self.attention.backward(grad_attn_out_split)

        # Stitch head gradients back together 
        grad_q = self._concat_heads(grad_q_split)
        grad_k = self._concat_heads(grad_k_split)
        grad_v = self._concat_heads(grad_v_split)

        # Pass representations through respective input projection backwards
        d_query = self.q_proj.backward(grad_q)
        d_key = self.k_proj.backward(grad_k)
        d_value = self.v_proj.backward(grad_v)

        return d_query, d_key, d_value

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge two mask types into a single mask.
        """
        if key_padding_mask is None and attn_mask is None:
            return None

        # Expand masks for broadcasting
        key_mask = None
        attention_mask = None

        if key_padding_mask is not None:
            # (N, S) -> (N, 1, 1, S)
            key_mask = key_padding_mask[:, None, None, :].astype(bool)

        if attn_mask is not None:
            # (L, S) -> (1, 1, L, S)
            attention_mask = attn_mask[None, None, :, :].astype(bool)
        
        # Combine masks
        if key_mask is None:
            combined_mask = attention_mask
        elif attention_mask is None:
            combined_mask = key_mask
        else:
            combined_mask = np.logical_or(key_mask, attention_mask)
        
        return combined_mask

    def _split_heads(self, x):
        """
        Reshape tensor for multi-head attention.
        """
        # Reshape and transpose for heads
        head_dim = self.embed_dim // self.num_heads
        x = x.reshape(self.N, -1, self.num_heads, head_dim)
        x = np.transpose(x, (0, 2, 1, 3))
        
        return x

    def _concat_heads(self, x):
        """
        Collapses the internal multi-head dimension into (N, L, embed_dim).
        Performs a continuous transpose placing the num_heads back.
        :param x: Array shaped as (N, num_heads, L, embed_dim // num_heads)
        :return: Array shaped as (N, L, embed_dim)
        """
        # Execute matrix axis swap and fold 
        x = np.transpose(x, (0, 2, 1, 3))
        x = x.reshape(x.shape[0], x.shape[1], self.embed_dim)
        
        return x
