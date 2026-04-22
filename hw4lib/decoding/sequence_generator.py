import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Constructor for the sequence generator using varied decoding strategies.
        
        Parameters:
            score_fn: The model's scoring function returning next-token logits.
            tokenizer: Tokenizer instance for sequence handling.
            max_length: The length capacity limit for generation.
            device: Computing device ('cuda' or 'cpu').
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Optimize by avoiding full_like allocation
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                idx_logits = logits[idx, unique_tokens]
                logits[idx, unique_tokens] = torch.where(
                    idx_logits > 0,
                    idx_logits / penalty,
                    idx_logits * penalty
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    idx_logits = logits[batch_idx, beam_idx, unique_tokens]
                    logits[batch_idx, beam_idx, unique_tokens] = torch.where(
                        idx_logits > 0,
                        idx_logits / penalty,
                        idx_logits * penalty
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Input assertions to ensure correctly formatted tensors
        if not torch.is_tensor(x):
            raise TypeError("Expected input x to be a PyTorch tensor.")
        if x.dim() != 2:
            raise ValueError("Input x must have shape (batch_size, seq_len).")
        if self.max_length < x.size(1):
            raise ValueError("Configured max_length must not be less than input sequence length.")
        if temperature <= 0.0:
            raise ValueError("Temperature must be strictly positive.")
        if repeat_penalty <= 0.0:
            raise ValueError("Repeat penalty must be strictly positive.")
        
        # Track cumulative log-prob scores for each sequence in the batch.
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Stop early if every sequence already reached EOS.
            if finished.all():
                break

            # Score next-token logits for current prefixes.
            next_logits = self.score_fn(x)  # (batch_size, vocab_size)
            next_logits = next_logits / temperature
            next_logits = self._apply_repeat_penalty(next_logits, x, repeat_penalty)
            log_probs = torch.log_softmax(next_logits, dim=-1)

            # Execute greedy selection by picking the highest-probability token.
            next_tokens = torch.argmax(log_probs, dim=-1)  # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            # Freeze sequences that have already hit EOS.
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, self.tokenizer.eos_id),
                next_tokens,
            )
            token_scores = torch.where(finished, torch.zeros_like(token_scores), token_scores)

            # Accumulate scores and concatenate generated tokens
            scores += token_scores
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            # Check for EOS tokens and update finished flag
            finished |= (next_tokens == self.tokenizer.eos_id)

        return x, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, beam_width, sequence_length) where each sequence in a beam set is sorted by score
             - scores is of shape (batch_size, beam_width)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be > 0")
        
        # Beam width 1 is equivalent to greedy decoding.
        if beam_width == 1:
            seqs, seq_scores = self.generate_greedy(x, temperature, repeat_penalty)
            return seqs.unsqueeze(1), seq_scores.unsqueeze(1)

        batch_size = x.size(0)
        vocab_size = None

        # First expansion from the prompt to initialize beam candidates.
        init_logits = self.score_fn(x)  # (batch_size, vocab_size)
        init_logits = init_logits / temperature
        init_logits = self._apply_repeat_penalty(init_logits, x, repeat_penalty)
        init_log_probs = torch.log_softmax(init_logits, dim=-1)
        vocab_size = init_log_probs.size(-1)

        top_log_probs, top_tokens = torch.topk(init_log_probs, k=beam_width, dim=-1)

        # Create initial beam sequences by appending the chosen token to each beam.
        sequences = x.unsqueeze(1).expand(-1, beam_width, -1).contiguous()
        sequences = torch.cat([sequences, top_tokens.unsqueeze(-1)], dim=-1)
        scores = top_log_probs
        finished = top_tokens == self.tokenizer.eos_id

        # Continue decoding until max_length or until all beams are finished.
        for _ in range(self.max_length - sequences.size(-1)):
            if finished.all():
                break

            # Score each beam separately so score_fn always receives (batch_size, seq_len).
            # This keeps compatibility with score functions that are batch-index-aware.
            beam_logits = []
            for beam_idx in range(beam_width):
                beam_logits.append(self.score_fn(sequences[:, beam_idx, :]))
            next_logits = torch.stack(beam_logits, dim=1)  # (B, beam, V)
            next_logits = next_logits / temperature
            next_logits = self._apply_repeat_penalty(next_logits, sequences, repeat_penalty)
            log_probs = torch.log_softmax(next_logits, dim=-1)

            # For finished beams, only allow EOS with zero additional score.
            if finished.any():
                forced_log_probs = torch.full_like(log_probs, float('-inf'))
                forced_log_probs[..., self.tokenizer.eos_id] = 0.0
                log_probs = torch.where(finished.unsqueeze(-1), forced_log_probs, log_probs)

            # Candidate score for each (beam, vocab) continuation.
            candidate_scores = scores.unsqueeze(-1) + log_probs  # (B, beam, V)
            candidate_scores = candidate_scores.view(batch_size, beam_width * vocab_size)

            # Keep the top-k candidates for each item in batch.
            top_scores, top_indices = torch.topk(candidate_scores, k=beam_width, dim=-1)
            next_beam_idx = top_indices // vocab_size
            next_tokens = top_indices % vocab_size

            # Select parent beams, then append the selected token.
            gather_idx = next_beam_idx.unsqueeze(-1).expand(-1, -1, sequences.size(-1))
            selected_sequences = torch.gather(sequences, dim=1, index=gather_idx)
            sequences = torch.cat([selected_sequences, next_tokens.unsqueeze(-1)], dim=-1)

            # Update beam scores and finished flags.
            scores = top_scores
            selected_finished = torch.gather(finished, dim=1, index=next_beam_idx)
            finished = selected_finished | (next_tokens == self.tokenizer.eos_id)

        # Beams are kept sorted by score (descending) via topk at every step.
        return sequences, scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break

            # Get logits and apply filtering
            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            # We need probabilities for multinomial sampling
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)

            # Check for EOS tokens and update finished flag
            finished |= (next_tokens == self.tokenizer.eos_id)

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]