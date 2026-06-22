import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.encodings import (  # noqa: E402
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    apply_rotary_embedding,
)


def test_sinusoidal_shape():
    d_model, seq_len = 16, 10
    enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=64)
    table = enc.encoding(seq_len)
    assert table.shape == (1, seq_len, d_model)

    x = torch.zeros(4, seq_len, d_model)
    out = enc(x)
    assert out.shape == x.shape


def test_sinusoidal_bounded():
    enc = SinusoidalPositionalEncoding(d_model=32, max_len=128)
    table = enc.encoding(128)
    assert torch.all(table <= 1.0 + 1e-6)
    assert torch.all(table >= -1.0 - 1e-6)


def test_sinusoidal_known_values():
    # position 0 has sin(0)=0 on even channels and cos(0)=1 on odd channels
    enc = SinusoidalPositionalEncoding(d_model=8, max_len=4)
    table = enc.encoding(4)[0]
    assert torch.allclose(table[0, 0::2], torch.zeros(4), atol=1e-6)
    assert torch.allclose(table[0, 1::2], torch.ones(4), atol=1e-6)


def test_learned_shape():
    d_model, seq_len, batch = 24, 7, 3
    enc = LearnedPositionalEncoding(d_model=d_model, max_len=50)
    table = enc.encoding(seq_len)
    assert table.shape == (1, seq_len, d_model)

    x = torch.randn(batch, seq_len, d_model)
    out = enc(x)
    assert out.shape == (batch, seq_len, d_model)


def test_learned_is_trainable():
    enc = LearnedPositionalEncoding(d_model=8, max_len=16)
    params = list(enc.parameters())
    assert len(params) == 1
    assert params[0].requires_grad
    assert params[0].shape == (16, 8)


def test_rope_preserves_norm():
    torch.manual_seed(0)
    batch, seq_len, heads, head_dim = 2, 12, 3, 16
    x = torch.randn(batch, seq_len, heads, head_dim)
    out = apply_rotary_embedding(x)
    assert out.shape == x.shape

    norm_before = x.norm(dim=-1)
    norm_after = out.norm(dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-5)


def test_rope_position_zero_is_identity():
    # at position 0 every angle is zero so the vector is unchanged
    torch.manual_seed(1)
    x = torch.randn(1, 5, 2, 8)
    out = apply_rotary_embedding(x)
    assert torch.allclose(out[:, 0], x[:, 0], atol=1e-6)


def test_rope_changes_nonzero_positions():
    torch.manual_seed(2)
    x = torch.randn(1, 6, 1, 8)
    out = apply_rotary_embedding(x)
    # a later position should actually be rotated
    assert not torch.allclose(out[:, 3], x[:, 3], atol=1e-4)


def test_rope_module_matches_function():
    torch.manual_seed(3)
    x = torch.randn(2, 9, 4, 16)
    mod = RotaryPositionalEncoding(head_dim=16)
    assert torch.allclose(mod(x), apply_rotary_embedding(x), atol=1e-6)


def test_rope_preserves_inner_product_relativity():
    # RoPE is built so the dot product of a query and key depends on the
    # difference of their positions. Shifting both by the same offset leaves
    # the inner product unchanged.
    torch.manual_seed(4)
    head_dim = 16
    q = torch.randn(1, 1, head_dim)
    k = torch.randn(1, 1, head_dim)

    def rope_at(vec, pos):
        # place the single token at absolute position `pos` by padding
        padded = torch.zeros(1, pos + 1, 1, head_dim)
        padded[:, pos, 0] = vec.squeeze()
        return apply_rotary_embedding(padded)[:, pos, 0]

    q2, k2 = rope_at(q, 2), rope_at(k, 5)
    q4, k4 = rope_at(q, 4), rope_at(k, 7)
    dot_a = (q2 * k2).sum()
    dot_b = (q4 * k4).sum()
    assert torch.allclose(dot_a, dot_b, atol=1e-4)
