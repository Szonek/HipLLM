import torch
torch.manual_seed(123)
from custom_mha import MultiHeadAttention


inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]])
batch = torch.stack([inputs, inputs], dim=0)
print("batch shape:", batch.shape)

d_in = 3
d_out = 2
num_heads = 2
batch_size, context_length, d_in = batch.shape
mha = MultiHeadAttention(d_in=d_in, d_out=d_out, context_length=context_length,
                          dropout=0.0, num_heads=num_heads)
context_vecs = mha(batch)
print("context_vecs.shape: ", context_vecs.shape)
print(context_vecs)