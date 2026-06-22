# positional-encoding-variants

Three ways to tell a transformer where each token sits in a sequence, implemented in PyTorch with property based tests. The point of this repo is to keep the math honest: each variant lives in one small module and each test checks a behavior that the math actually guarantees.

## What is included

**Sinusoidal encoding.** The fixed scheme from the original attention paper. Even channels carry a sine wave, odd channels carry the matching cosine, and the wavelength grows geometrically across the channel dimension. Nothing here is learned, so the encoding is the same every run and every value stays inside the range minus one to one.

**Learned encoding.** A plain embedding table indexed by position. The model learns one vector per position during training, which is the approach used by models like BERT. It is flexible but it cannot generalize past the longest position it was trained on.

**Rotary encoding (RoPE).** Instead of adding a vector, RoPE rotates each query and key inside a set of two dimensional subspaces by an angle that depends on the position. Because a rotation never changes the length of a vector, the norm of every token is preserved, and the dot product of a rotated query and key ends up depending only on the difference between their positions. That relative property is what makes RoPE attractive.

## Layout

```
src/encodings.py     the three encodings plus a functional RoPE helper
tests/               pytest property and behavior checks
requirements.txt     torch and pytest
```

## Usage

```python
import torch
from src.encodings import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    apply_rotary_embedding,
)

x = torch.randn(2, 10, 32)          # batch, sequence, model dim

sin_enc = SinusoidalPositionalEncoding(d_model=32)
x_sin = sin_enc(x)                  # adds the fixed table

learned = LearnedPositionalEncoding(d_model=32)
x_learned = learned(x)             # adds the learned table

# RoPE works on attention heads, shape batch, sequence, heads, head_dim
q = torch.randn(2, 10, 4, 16)
q_rotated = apply_rotary_embedding(q)
```

## What the tests check

These are behavior checks rather than trivial asserts.

* Sinusoidal output has the right shape and every entry lands in minus one to one.
* At position zero the sine channels are zero and the cosine channels are one, which is a closed form value the formula must produce.
* The learned encoding returns the right shape and exposes exactly one trainable embedding table.
* RoPE preserves the L2 norm of every vector after rotation.
* RoPE leaves position zero untouched and does change later positions.
* The functional helper and the module wrapper agree.
* Rotating a query and key by the same positional shift leaves their inner product unchanged, which is the relative position property RoPE is designed for.

## Running

```
pip install -r requirements.txt
python -m pytest tests/ -q
```
