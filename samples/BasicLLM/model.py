import torch


GPT_CONFIG_124M = {
    "vocab_size":     50257,  # BPE tokenizer
    "context_length": 1024,   # max number of tokens model can handle
    "emb_dim":        768,    # each token is transformed into embedding dimension vector
    "n_heads":        12,     # heads in MHAs
    "n_layers":       12,     # number of transformer blocks
    "drop_rate":      0.1,    # 0.1 = 10%; used to prevent overfitting
    "qkv_bias":       False   # wheter to include bias in Linear layers of MHA
}