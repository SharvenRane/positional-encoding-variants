"""Positional encoding variants implemented in PyTorch.

This module contains three families of positional encodings used in
transformer style models:

* Sinusoidal encodings from the original attention paper. They are fixed
  (not learned) and produced from sine and cosine functions of the position.
* Learned encodings, a plain embedding table indexed by position.
* Rotary encodings (RoPE), which rotate query and key vectors by an angle
  that depends on the position. Rotations preserve vector norm, which is the
  property the tests verify.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    For a model dimension ``d_model`` and a position ``pos``, even channels
    hold ``sin(pos / 10000^(2i/d))`` and odd channels hold the matching cosine
    term. The buffer is registered so it moves with the module to any device
    but never receives gradients.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal encoding")
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape (1, max_len, d_model) so it broadcasts over the batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def encoding(self, seq_len: int) -> torch.Tensor:
        """Return the encoding table for the first ``seq_len`` positions."""
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_len {self.max_len}"
            )
        return self.pe[:, :seq_len, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the positional encoding to an input of shape (B, T, d_model)."""
        if x.dim() != 3:
            raise ValueError("expected input of shape (batch, seq_len, d_model)")
        seq_len = x.size(1)
        return x + self.encoding(seq_len)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding backed by an embedding table."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(max_len, d_model)

    def encoding(self, seq_len: int, device=None) -> torch.Tensor:
        """Return the learned encoding for the first ``seq_len`` positions."""
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_len {self.max_len}"
            )
        if device is None:
            device = self.embedding.weight.device
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the learned encoding to an input of shape (B, T, d_model)."""
        if x.dim() != 3:
            raise ValueError("expected input of shape (batch, seq_len, d_model)")
        seq_len = x.size(1)
        return x + self.encoding(seq_len, device=x.device)


def _build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    """Precompute cos and sin tables for rotary embedding.

    Returns two tensors of shape (seq_len, head_dim) where each consecutive
    pair of channels shares the same angle.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)  # (seq_len, half)
    # interleave so channel 2i and 2i+1 share an angle
    cos = torch.repeat_interleave(torch.cos(freqs), 2, dim=1)
    sin = torch.repeat_interleave(torch.sin(freqs), 2, dim=1)
    return cos.to(dtype), sin.to(dtype)


def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """Map (..., x0, x1, x2, x3, ...) to (..., -x1, x0, -x3, x2, ...)."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(-2)


def apply_rotary_embedding(
    x: torch.Tensor, base: float = 10000.0
) -> torch.Tensor:
    """Apply rotary position embedding to a tensor of shape (B, T, H, D).

    Each (B, H) sequence of D dimensional vectors is rotated by a position
    dependent angle. Because the operation is a rotation in each 2D subspace,
    the L2 norm of every vector is preserved.
    """
    if x.dim() != 4:
        raise ValueError("expected input of shape (batch, seq_len, heads, head_dim)")
    head_dim = x.size(-1)
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for rotary embedding")
    seq_len = x.size(1)
    cos, sin = _build_rope_cache(seq_len, head_dim, base, x.device, x.dtype)
    # reshape cos/sin to (1, seq_len, 1, head_dim) for broadcasting
    cos = cos.view(1, seq_len, 1, head_dim)
    sin = sin.view(1, seq_len, 1, head_dim)
    return x * cos + _rotate_pairs(x) * sin


class RotaryPositionalEncoding(nn.Module):
    """Module wrapper around :func:`apply_rotary_embedding`."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embedding")
        self.head_dim = head_dim
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.head_dim:
            raise ValueError(
                f"last dim {x.size(-1)} does not match head_dim {self.head_dim}"
            )
        return apply_rotary_embedding(x, base=self.base)
