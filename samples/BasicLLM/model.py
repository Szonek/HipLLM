import torch
import torch.nn as nn

import tiktoken

GPT_CONFIG_124M = {
    "vocab_size":     50257,  # BPE tokenizer
    "context_length": 1024,   # max number of tokens model can handle
    "emb_dim":        768,    # each token is transformed into embedding dimension vector
    "n_heads":        12,     # heads in MHAs
    "n_layers":       12,     # number of transformer blocks
    "drop_rate":      0.1,    # 0.1 = 10%; used to prevent overfitting
    "qkv_bias":       False   # wheter to include bias in Linear layers of MHA
}

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
    def forward(self, x):
        return x

# goal of the layer norm is to improve training stability and weights convergance 
# by making mean = 0 and var = 1
# example usage:
#batch_example = torch.rand(2, 5)
#ln = LayerNorm(5)
#out_Example = ln(batch_example)
#torch.set_printoptions(sci_mode=False)
#print(out_Example)
#print(out_Example.mean(dim=-1, keepdim=True))
#print(out_Example.var(dim=-1, keepdim=True, unbiased=False))
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x : torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_nom = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx : torch.Tensor):
        batch_size, seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_nom(x)
        logits = self.out_head(x)
        return logits


tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape: ", logits.shape)
print(logits)
