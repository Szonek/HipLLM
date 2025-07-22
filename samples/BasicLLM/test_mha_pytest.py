import pytest
import numpy as np
from custom_mha import MultiHeadAttention

def test_mha_forward_shape():
    # Example input: batch_size=2, seq_len=4, embed_dim=8
    batch_size, seq_len, embed_dim, num_heads = 2, 4, 8, 2
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    output = mha.forward(x)
    # Output shape should match input shape
    assert output.shape == (batch_size, seq_len, embed_dim)
