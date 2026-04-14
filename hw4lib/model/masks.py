import torch

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """ 
    Constructs a mask identifying padded positions. 
    Parameters:
        padded_input: Tensor representing input sequences, shape (N, T, ...).
        input_lengths: 1D tensor of accurate pre-padding lengths, shape (N,).
    Returns:
        Boolean tensor defining padding positions with shape (N, T).
    """
    # Instantiate time indices [0, 1, ..., T-1] to cross-check valid extents.
    sequence_length = padded_input.size(1)
    time_indices = torch.arange(sequence_length, device=padded_input.device).unsqueeze(0)  # (1, T)
    actual_lens = input_lengths.to(padded_input.device).unsqueeze(1)                       # (N, 1)

    # Identifiers exceed valid sequence lengths represent padded elements.
    return time_indices >= actual_lens

''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """ 
    Constructs an upper-triangular self-attention causal mask. 
    Parameters:
        padded_input: Batch sequence tensor, shape (N, T, ...).
    Returns:
        Boolean tensor blocking future attention with shape (T, T).
    """
    # Upper-triangular mask (excluding diagonal) blocks attention to future positions.
    seq_len = padded_input.size(1)
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=padded_input.device),
        diagonal=1,
    )

