import pytest
import torch
from custom_mha import MultiHeadAttention

@pytest.mark.parametrize("batch_size, seq_len, embed_dim, num_heads", [
    (8, 1024, 768, 12),
])
def test_mha_forward_shape(batch_size, seq_len, embed_dim, num_heads):
    torch.manual_seed(1337)
    mha = MultiHeadAttention(d_in=embed_dim, d_out=embed_dim, num_heads=num_heads, context_length=seq_len, dropout=0.0)
    x = torch.rand(batch_size, seq_len, embed_dim)
    output = mha.forward(x)
    # Output shape should match input shape
    assert output.shape == (batch_size, seq_len, embed_dim)
